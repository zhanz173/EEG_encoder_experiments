"""Audit amplitudes in the exact fixed windows saved by an EEG checkpoint.

The high-amplitude threshold is a dataset-relative quality-control flag, not a
clinical abnormality threshold. HDF5 signal units are reported as stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.eeg_dataset import EEGWindowDataset
from utils.experiment_utils import read_manifest


def channel_names(value, n_channels: int) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    names = list(value)
    if len(names) != n_channels:
        raise ValueError(f"Expected {n_channels} channel names, got {len(names)}")
    return [str(name) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--peak-multiple", type=float, default=20.0,
                        help="Flag a window when its absolute peak exceeds this times the median peak")
    args = parser.parse_args()
    if args.peak_multiple <= 1:
        parser.error("peak-multiple must be greater than one")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    manifest = read_manifest(config["manifest"], config["sfreq"], config["channels"],
                             config["window_sec"])
    manifest = manifest.set_index("sha256_id", drop=False)
    bad_path = Path(config["manifest"]).with_name("bad_channels.parquet")
    bad_channels: dict[str, set[str]] = {}
    if bad_path.exists():
        bad_table = pd.read_parquet(bad_path)
        bad_channels = {
            str(recording_id): set(group["channel_name"].astype(str))
            for recording_id, group in bad_table.groupby("sha256_id")
        }

    rows = []
    channel_peaks = []
    for split in args.splits:
        ids = checkpoint["split_ids"][split]
        records = manifest.loc[ids].reset_index(drop=True)
        dataset = EEGWindowDataset(
            records, config["shards_dir"], config["window_sec"],
            config["eval_windows_per_recording"], False, config["channels"],
            config["sfreq"], eval_start_sec=config.get("eval_start_sec", 0.0),
        )
        for index in range(len(dataset)):
            item = dataset[index]
            recording_id = item["sha256_id"]
            record = manifest.loc[recording_id]
            names = channel_names(record["channel_names"], config["channels"])
            x = item["x"].numpy()
            absolute = np.abs(x)
            peak_channel_index, peak_sample = np.unravel_index(
                int(absolute.argmax()), absolute.shape
            )
            per_channel_peak = absolute.max(axis=1)
            start_sample = int(item["crop_start_sample"])
            peak_channel = names[peak_channel_index]
            marked_bad = bad_channels.get(recording_id, set())
            row = {
                "split": split,
                "sha256_id": recording_id,
                "site": str(record["site"]) if "site" in record else "",
                "start_sec": start_sample / config["sfreq"],
                "stop_sec": (start_sample + x.shape[-1]) / config["sfreq"],
                "peak_time_sec": (start_sample + int(peak_sample)) / config["sfreq"],
                "peak_abs_raw": float(absolute[peak_channel_index, peak_sample]),
                "peak_channel": peak_channel,
                "peak_channel_index": int(peak_channel_index),
                "peak_channel_marked_bad": peak_channel in marked_bad,
                "rms_raw": float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))),
                "signal_energy_raw": float(np.square(x, dtype=np.float64).sum()),
                "n_bad_channels": len(marked_bad),
                "bad_channel_names": ",".join(sorted(marked_bad)),
                "n_manifest_annotations": int(record["num_annotations"]) if "num_annotations" in record else -1,
            }
            rows.append(row)
            for channel_index, peak in enumerate(per_channel_peak):
                channel_peaks.append({
                    "split": split, "sha256_id": recording_id,
                    "start_sec": row["start_sec"], "channel": names[channel_index],
                    "peak_abs_raw": float(peak),
                    "marked_bad": names[channel_index] in marked_bad,
                })
        dataset.close()

    all_windows = pd.DataFrame(rows)
    median_peak = float(all_windows["peak_abs_raw"].median())
    threshold = median_peak * args.peak_multiple
    all_windows["peak_over_median"] = all_windows["peak_abs_raw"] / median_peak
    all_windows["high_amplitude_flag"] = all_windows["peak_abs_raw"] > threshold
    all_windows["energy_fraction_of_split"] = all_windows["signal_energy_raw"] / (
        all_windows.groupby("split")["signal_energy_raw"].transform("sum")
    )
    high = all_windows[all_windows["high_amplitude_flag"]].sort_values(
        "peak_abs_raw", ascending=False
    )
    high_keys = set(zip(high["split"], high["sha256_id"], high["start_sec"]))
    high_channel_rows = [row for row in channel_peaks
                         if (row["split"], row["sha256_id"], row["start_sec"]) in high_keys]
    recording_summary = (all_windows.groupby(["split", "sha256_id", "site"], as_index=False)
                         .agg(n_windows=("peak_abs_raw", "size"),
                              n_flagged_windows=("high_amplitude_flag", "sum"),
                              max_peak_abs_raw=("peak_abs_raw", "max"),
                              total_signal_energy_raw=("signal_energy_raw", "sum"))
                         .sort_values("max_peak_abs_raw", ascending=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_windows.sort_values("signal_energy_raw", ascending=False).to_csv(
        output_dir / "all_eval_windows.csv", index=False
    )
    high.to_csv(output_dir / "high_amplitude_windows.csv", index=False)
    pd.DataFrame(high_channel_rows).to_csv(output_dir / "high_amplitude_channels.csv", index=False)
    recording_summary.to_csv(output_dir / "recording_summary.csv", index=False)
    report = {
        "checkpoint": str(args.checkpoint),
        "splits": args.splits,
        "eval_start_sec": config.get("eval_start_sec", 0.0),
        "signal_units": "as stored; no unit metadata found in the HDF5 signals dataset",
        "median_window_peak_raw": median_peak,
        "flag_threshold_raw": threshold,
        "flag_rule": f"peak > {args.peak_multiple:g} times the median evaluation-window peak",
        "n_windows": len(all_windows),
        "n_flagged_windows": len(high),
        "n_flagged_recordings": int(high["sha256_id"].nunique()),
        "flagged_windows_by_split": high["split"].value_counts().to_dict(),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(high[["split", "sha256_id", "site", "start_sec", "peak_time_sec",
                "peak_abs_raw", "peak_channel", "peak_channel_marked_bad",
                "energy_fraction_of_split"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
