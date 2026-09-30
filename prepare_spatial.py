"""Prepare fixed patient splits, QC windows and one training-only amplitude scale."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_utils import read_manifest, split_records
from spatial_data import SpatialDataset, write_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--qc-dir")
    p.add_argument("--allow-no-qc", action="store_true")
    p.add_argument("--patient-column", default="Hashed_PatientURN")
    p.add_argument("--scan-column", default="ScanID")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-recordings", type=int, default=2560)
    p.add_argument("--val-recordings", type=int, default=256)
    p.add_argument("--test-recordings", type=int, default=256)
    p.add_argument("--window-sec", type=float, default=8)
    p.add_argument("--eval-start-sec", type=float, default=16)
    p.add_argument("--sfreq", type=float, default=256)
    p.add_argument("--channels", type=int, default=20)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error("Use a new output directory; prepared splits are immutable")
    length = round(args.window_sec*args.sfreq)
    if length < 16 or length % 16 or args.eval_start_sec < 0:
        p.error("Window samples must be a positive multiple of 16; start must be nonnegative")
    if not args.qc_dir and not args.allow_no_qc:
        p.error("Supply --qc-dir, or explicitly use --allow-no-qc for a separate raw-data experiment")
    records = read_manifest(args.manifest, args.sfreq, args.channels, args.window_sec)
    meta = pd.read_csv(args.metadata, usecols=[args.scan_column, args.patient_column], dtype=str)
    meta = meta.dropna().rename(columns={args.scan_column: "sha256_id", args.patient_column: "patient_id"})
    for col in ("sha256_id", "patient_id"):
        meta[col] = meta[col].str.strip()
        meta = meta[~meta[col].str.lower().isin(["", "nan", "none", "null"])]
    if (meta.groupby("sha256_id").patient_id.nunique() > 1).any():
        raise ValueError("Conflicting patient mappings; resolve before splitting")
    n_original = len(records)
    records = records.merge(meta.drop_duplicates("sha256_id"), on="sha256_id", validate="one_to_one")
    matched = len(records)
    splits = split_records(records, "patient_id", args.split_seed,
                           args.train_recordings, args.val_recordings, args.test_recordings)
    selected = pd.concat([v.assign(split=k) for k, v in splits.items()], ignore_index=True)
    if "channel_names" in selected:
        orders = selected.channel_names.map(lambda v: v if isinstance(v, str) else tuple(v))
        if orders.nunique() != 1:
            raise ValueError("Selected recordings have inconsistent channel ordering; harmonize before preparing")
    events = pd.DataFrame(columns=["sha256_id", "channel_index", "start_sample", "stop_sample"])
    qc_config, pad = None, 0
    if args.qc_dir:
        qc = Path(args.qc_dir)
        qc_config = json.loads((qc/"qc_config.json").read_text())
        if not np.isclose(qc_config["window_sec"], args.window_sec):
            raise ValueError("QC and experiment window lengths differ")
        filt = [("sha256_id", "in", selected.sha256_id.tolist())]
        windows = pd.read_parquet(qc/"window_qc.parquet", filters=filt)
        selected_without_cache = set(selected.sha256_id)-set(windows.sha256_id)
        if selected_without_cache:
            raise ValueError(f"QC cache is missing {len(selected_without_cache)} selected recordings; finish the cache first")
        windows = windows[windows.status.eq("accept")][["sha256_id", "start_sample"]].copy()
        all_events = pd.read_parquet(qc/"events.parquet", filters=filt)
        if "masked" in all_events:
            events = all_events[all_events.masked][list(events.columns)].copy()
        pad = round(qc_config.get("mask_pad_ms", 0)*args.sfreq/1000)
    else:
        windows = pd.DataFrame([(r.sha256_id, s) for r in selected.itertuples()
                                for s in range(0, int(r.n_samples)-length+1, length)],
                               columns=["sha256_id", "start_sample"])
    windows = windows.merge(selected[["sha256_id", "split", "n_samples"]], on="sha256_id")
    windows = windows[(windows.start_sample+length <= windows.n_samples) &
                      (windows.start_sample >= args.eval_start_sec*args.sfreq)].copy()
    if (windows.start_sample % length != 0).any() or windows.duplicated(["sha256_id", "start_sample"]).any():
        raise ValueError("QC windows must be unique and grid aligned")
    event_groups = {k: g for k, g in events.groupby("sha256_id")}
    def masked_count(row):
        mask = np.zeros((args.channels, length), bool)
        if row.sha256_id in event_groups:
            for e in event_groups[row.sha256_id].itertuples():
                lo = max(e.start_sample-pad, row.start_sample)-row.start_sample
                hi = min(e.stop_sample+pad, row.start_sample+length)-row.start_sample
                if lo < hi:
                    mask[int(e.channel_index), int(lo):int(hi)] = True
        return int(mask.sum())
    windows["masked_samples"] = [masked_count(r) for r in windows.itertuples()]
    selected = selected[selected.sha256_id.isin(windows.sha256_id)]
    if set(selected.split) != {"train", "val", "test"}:
        raise ValueError("QC left an empty split")
    patient_sets = [set(selected[selected.split.eq(s)].patient_id) for s in ("train", "val", "test")]
    assert all(not patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i))
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out/"records.csv", index=False)
    windows[["sha256_id", "start_sample", "masked_samples"]].to_csv(out/"windows.csv", index=False)
    events.to_csv(out/"masked_events.csv", index=False)
    config = dict(vars(args), window_samples=length, n_channels=args.channels,
                  mask_pad_samples=pad, qc_config=qc_config, normalization="training_global_p75",
                  original_recordings=n_original, matched_recordings=matched,
                  selected_counts=selected.groupby("split").size().to_dict(),
                  patient_counts=selected.groupby("split").patient_id.nunique().to_dict(),
                  manifest_sha256=fingerprint(args.manifest), metadata_sha256=fingerprint(args.metadata))
    write_json(out/"data.json", config)
    ds = SpatialDataset(out, args.shards_dir, "train", max_windows=2)
    indices = np.linspace(0, len(ds)-1, min(128, len(ds)), dtype=int)
    values = []
    for i in indices:
        item = ds[int(i)]
        values.append(item["x"][~item["mask"]].numpy()[::8])
    scale = float(np.percentile(np.abs(np.concatenate(values)), 75))
    ds.close()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training amplitude scale")
    config["scale"] = scale
    config["files_sha256"] = {name: fingerprint(out/name) for name in ("records.csv", "windows.csv", "masked_events.csv")}
    write_json(out/"data.json", config)
    print(json.dumps({"prepared": str(out), "scale": scale, "counts": config["selected_counts"]}))


if __name__ == "__main__":
    main()
