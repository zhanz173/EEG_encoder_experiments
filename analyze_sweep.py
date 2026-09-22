"""Re-evaluate a saved rate sweep with recording-level and EEG-band metrics.

This is descriptive analysis of fixed checkpoint evaluation windows. The
amplitude flag is a relative QC flag, not a clinical diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eeg_dataset import EEGWindowDataset
from experiment_utils import read_manifest
from model import EEGRateDistortionAE


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "beta": (13, 30),
}


def load_windows(checkpoint: dict, split: str, manifest: pd.DataFrame):
    config = checkpoint["config"]
    selected = manifest.set_index("sha256_id").loc[checkpoint["split_ids"][split]].reset_index()
    dataset = EEGWindowDataset(
        selected, config["shards_dir"], config["window_sec"],
        config["eval_windows_per_recording"], False, config["channels"],
        config["sfreq"], eval_start_sec=config["eval_start_sec"],
    )
    try:
        items = [dataset[i] for i in range(len(dataset))]
    finally:
        dataset.close()
    return {
        "x": np.stack([item["x"].numpy() for item in items]),
        "ids": [item["sha256_id"] for item in items],
        "start_sec": [item["crop_start_sample"] / config["sfreq"] for item in items],
    }


def evaluate_checkpoint(checkpoint: dict, windows: dict, device: torch.device,
                        batch_size: int, split: str, flagged_ids: set[str]):
    model = EEGRateDistortionAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    config = checkpoint["config"]
    n_time = windows["x"].shape[-1]
    taper = torch.hann_window(n_time, device=device)
    freqs = torch.fft.rfftfreq(n_time, d=1 / config["sfreq"]).to(device)
    weights = torch.ones_like(freqs)
    weights[1:-1] = 2
    masks = {name: ((freqs >= low) & (freqs < high)) * weights
             for name, (low, high) in BANDS.items()}
    psd_taper = torch.hann_window(256, device=device)

    def psd(signal: torch.Tensor) -> torch.Tensor:
        frames = signal.unfold(-1, 256, 128) * psd_taper
        return torch.fft.rfft(frames, dim=-1).abs().square().mean(dim=-2)

    def spatial_correlation(signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return normalized @ normalized.transpose(-1, -2)

    rows = []
    with torch.inference_mode():
        for first in range(0, len(windows["ids"]), batch_size):
            last = min(first + batch_size, len(windows["ids"]))
            x = torch.from_numpy(windows["x"][first:last]).to(device) / checkpoint["scale"]
            z_hat = torch.round(model.encoder(x))
            y = model.decoder(z_hat)
            rate_by_window = (model.prior.bits(z_hat).sum(dim=(-1, -2)) /
                              (x.shape[1] * x.shape[2]))
            px, py = psd(x), psd(y)
            cx, cy = spatial_correlation(x), spatial_correlation(y)
            lx = x.diff(dim=-1).abs().mean(dim=(-1, -2))
            ly = y.diff(dim=-1).abs().mean(dim=(-1, -2))
            peak_x = x.abs().amax(dim=(-1, -2))
            peak_y = y.abs().amax(dim=(-1, -2))
            batch = {
                "wave_energy": x.square().sum(dim=(-1, -2)).cpu().numpy(),
                "wave_error": (x - y).square().sum(dim=(-1, -2)).cpu().numpy(),
                "estimated_bits_per_channel_sample": rate_by_window.cpu().numpy(),
                "log_psd_mae": (torch.log10(px.clamp_min(1e-8)) -
                                torch.log10(py.clamp_min(1e-8))).abs().mean(dim=(-1, -2)).cpu().numpy(),
                "spatial_corr_error": (((cx - cy).square().sum(dim=(-1, -2)).sqrt()) /
                                       cx.square().sum(dim=(-1, -2)).sqrt().clamp_min(1e-8)).cpu().numpy(),
                "line_length_rel_error": ((lx - ly).abs() / lx.clamp_min(1e-8)).cpu().numpy(),
                "peak_rel_error": ((peak_x - peak_y).abs() /
                                   peak_x.clamp_min(1e-8)).cpu().numpy(),
            }
            x_fft = torch.fft.rfft(x * taper, dim=-1)
            error_fft = x_fft - torch.fft.rfft(y * taper, dim=-1)
            x_power = x_fft.abs().square()
            error_power = error_fft.abs().square()
            for name, mask in masks.items():
                batch[f"{name}_energy"] = (x_power * mask).sum(dim=(-1, -2)).cpu().numpy()
                batch[f"{name}_error"] = (error_power * mask).sum(dim=(-1, -2)).cpu().numpy()
            for offset, recording_id in enumerate(windows["ids"][first:last]):
                row = {
                    "split": split, "sha256_id": recording_id,
                    "start_sec": windows["start_sec"][first + offset],
                    "amplitude_flagged_recording": recording_id in flagged_ids,
                }
                row.update({key: float(value[offset]) for key, value in batch.items()})
                rows.append(row)
    del model
    return rows


def summarize(rows: pd.DataFrame, samples_per_second: float):
    components = ("wave", *BANDS)
    sum_cols = [f"{name}_{suffix}" for name in components for suffix in ("energy", "error")]
    average_cols = ("estimated_bits_per_channel_sample", "log_psd_mae",
                    "spatial_corr_error", "line_length_rel_error", "peak_rel_error")
    grouping = rows.groupby(["lambda_rate", "split", "sha256_id",
                             "amplitude_flagged_recording"], as_index=False)
    per_recording = grouping[sum_cols + list(average_cols)].sum()
    per_recording["n_windows"] = grouping.size()["size"]
    for name in average_cols:
        per_recording[name] /= per_recording["n_windows"]
    for name in components:
        per_recording[f"{name}_nmse"] = (per_recording[f"{name}_error"] /
                                          per_recording[f"{name}_energy"].clip(lower=1e-12))
    summaries = []
    for (rate, split), group in per_recording.groupby(["lambda_rate", "split"]):
        clean = group[~group["amplitude_flagged_recording"]]
        row = {
            "lambda_rate": rate, "split": split,
            "n_recordings": len(group),
            "n_amplitude_flagged_recordings": int(group["amplitude_flagged_recording"].sum()),
            "wave_nmse_pooled": group["wave_error"].sum() / group["wave_energy"].sum(),
            "wave_nmse_pooled_unflagged": clean["wave_error"].sum() / clean["wave_energy"].sum(),
            "wave_nmse_mean_recording": group["wave_nmse"].mean(),
            "wave_nmse_median_recording": group["wave_nmse"].median(),
            "wave_nmse_p90_recording": group["wave_nmse"].quantile(0.9),
            "wave_nmse_mean_recording_unflagged": clean["wave_nmse"].mean(),
            "wave_nmse_median_recording_unflagged": clean["wave_nmse"].median(),
            "wave_nmse_p90_recording_unflagged": clean["wave_nmse"].quantile(0.9),
        }
        row["snr_db_pooled"] = -10 * np.log10(row["wave_nmse_pooled"])
        row["snr_db_pooled_unflagged"] = -10 * np.log10(row["wave_nmse_pooled_unflagged"])
        for name in average_cols:
            row[name] = np.average(group[name], weights=group["n_windows"])
            row[f"{name}_unflagged"] = np.average(clean[name], weights=clean["n_windows"])
        row["estimated_bits_per_second"] = row["estimated_bits_per_channel_sample"] * samples_per_second
        row["estimated_bits_per_second_unflagged"] = (
            row["estimated_bits_per_channel_sample_unflagged"] * samples_per_second
        )
        for name in BANDS:
            row[f"{name}_nmse_pooled"] = group[f"{name}_error"].sum() / group[f"{name}_energy"].sum()
            row[f"{name}_nmse_mean_recording"] = group[f"{name}_nmse"].mean()
            row[f"{name}_nmse_median_recording"] = group[f"{name}_nmse"].median()
            row[f"{name}_nmse_pooled_unflagged"] = (clean[f"{name}_error"].sum() /
                                                     clean[f"{name}_energy"].sum())
            row[f"{name}_nmse_mean_recording_unflagged"] = clean[f"{name}_nmse"].mean()
            row[f"{name}_nmse_median_recording_unflagged"] = clean[f"{name}_nmse"].median()
        summaries.append(row)
    return per_recording, pd.DataFrame(summaries).sort_values(["split", "lambda_rate"])


def draw_plots(metrics: pd.DataFrame, output_dir: Path):
    test = metrics[metrics["split"] == "test"].sort_values("lambda_rate")
    val = metrics[metrics["split"] == "val"].sort_values("lambda_rate")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    axes[0].plot(test["estimated_bits_per_channel_sample"],
                 test["wave_nmse_pooled"], "o-", label="Pooled test")
    axes[0].plot(val["estimated_bits_per_channel_sample"],
                 val["wave_nmse_pooled"], "o--", label="Pooled validation")
    axes[0].set(xlabel="Estimated bits / channel-sample", ylabel="Waveform NMSE",
                title="Pooled waveform error")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(test["estimated_bits_per_channel_sample_unflagged"],
                 test["wave_nmse_pooled_unflagged"], "o-", label="Unflagged test recordings")
    axes[1].plot(test["estimated_bits_per_channel_sample_unflagged"],
                 test["wave_nmse_median_recording_unflagged"], "o-", label="Median unflagged recording")
    axes[1].set(xlabel="Estimated bits / channel-sample", ylabel="Waveform NMSE",
                title="Typical recording")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for name, label in (("delta", "Delta 1–4 Hz"), ("alpha", "Alpha 8–13 Hz"),
                        ("low_beta", "Low beta 13–20 Hz"), ("beta", "Beta 13–30 Hz")):
        axes[2].plot(test["estimated_bits_per_channel_sample_unflagged"],
                     test[f"{name}_nmse_median_recording_unflagged"], "o-", label=label)
    axes[2].set(xlabel="Estimated bits / channel-sample",
                ylabel="Median recording band NMSE", title="Band reconstruction")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(output_dir / "rate_distortion.png", dpi=180)
    plt.close(fig)


def draw_training(paths: list[Path], checkpoints: list[dict], output_dir: Path):
    histories = {}
    for path, checkpoint in zip(paths, checkpoints):
        rate = checkpoint["config"]["lambda_rate"]
        histories[rate] = [json.loads(line) for line in
                           (path.parent / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    available_rates = sorted(histories)
    rates = [available_rates[0], available_rates[len(available_rates) // 2], available_rates[-1]]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for rate in rates:
        history = histories[rate]
        ax.plot([row["epoch"] for row in history], [row["val_nmse"] for row in history],
                label=f"λ = {rate:g}")
    ax.set(xlabel="Epoch", ylabel="Pooled validation waveform NMSE",
           title="Validation error through epoch 30")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "training_convergence.png", dpi=180)
    plt.close(fig)


def paired_bootstrap(per_recording: pd.DataFrame, comparisons: tuple[float, ...],
                     n_bootstrap: int = 5000) -> pd.DataFrame:
    """Resample recording IDs in pairs; this does not capture training-seed variance."""
    rng = np.random.default_rng(12345)
    baseline_rate = float(per_recording["lambda_rate"].min())
    rows = []
    for split in ("val", "test"):
        baseline = per_recording[(per_recording["split"] == split) &
                                 (per_recording["lambda_rate"] == baseline_rate)]
        for comparison_rate in comparisons:
            comparison = per_recording[(per_recording["split"] == split) &
                                       (per_recording["lambda_rate"] == comparison_rate)]
            paired = baseline.merge(comparison, on="sha256_id", suffixes=("_base", "_comparison"),
                                    validate="one_to_one")
            indices = rng.integers(0, len(paired), size=(n_bootstrap, len(paired)))
            for band in ("wave", "delta", "alpha", "low_beta", "beta"):
                before = paired[f"{band}_nmse_base"].to_numpy()
                after = paired[f"{band}_nmse_comparison"].to_numpy()
                differences = np.median(after[indices], axis=1) - np.median(before[indices], axis=1)
                rows.append({
                    "split": split, "baseline_lambda": baseline_rate,
                    "comparison_lambda": comparison_rate, "band": band,
                    "n_recordings": len(paired),
                    "median_nmse_difference": float(np.median(after) - np.median(before)),
                    "bootstrap_ci_low": float(np.quantile(differences, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(differences, 0.975)),
                    "n_bootstrap": n_bootstrap,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    paths = sorted(args.run_dir.glob("rate_*/best.pt"))
    if not paths:
        parser.error("No rate_*/best.pt checkpoints found")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = checkpoints[0]
    config = base["config"]
    compare_keys = set(config) - {"lambda_rate", "output_dir"}
    for path, checkpoint in zip(paths, checkpoints):
        if any(checkpoint["config"].get(key) != config[key] for key in compare_keys):
            raise ValueError(f"Configuration differs from first checkpoint: {path}")
        if checkpoint["split_ids"] != base["split_ids"]:
            raise ValueError(f"Split IDs differ from first checkpoint: {path}")
    reference_match = None
    if args.reference_checkpoint:
        reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
        reference_match = {
            "split_ids_match": reference["split_ids"] == base["split_ids"],
            "scale_match": reference["scale"] == base["scale"],
            "config_match_except_lambda_and_output": all(
                reference["config"].get(key) == config[key] for key in compare_keys
            ),
        }
    manifest = read_manifest(config["manifest"], config["sfreq"], config["channels"],
                             config["window_sec"])
    windows = {split: load_windows(base, split, manifest) for split in ("val", "test")}
    all_peaks = np.concatenate([np.abs(windows[split]["x"]).max(axis=(1, 2))
                                for split in ("val", "test")])
    threshold = float(np.median(all_peaks) * 20)
    flagged_ids = {
        split: {recording_id for recording_id, peak in zip(windows[split]["ids"],
                    np.abs(windows[split]["x"]).max(axis=(1, 2))) if peak > threshold}
        for split in ("val", "test")
    }
    output_dir = args.run_dir / "report_analysis"
    output_dir.mkdir(exist_ok=True)
    all_rows = []
    for path, checkpoint in zip(paths, checkpoints):
        rate = checkpoint["config"]["lambda_rate"]
        for split in ("val", "test"):
            evaluated = evaluate_checkpoint(checkpoint, windows[split], device,
                                            args.batch_size, split, flagged_ids[split])
            all_rows.extend({"lambda_rate": rate, **row} for row in evaluated)
        print(f"Evaluated lambda={rate:g} on {device}", flush=True)
    per_recording, metrics = summarize(pd.DataFrame(all_rows), config["channels"] * config["sfreq"])
    source_summary = pd.read_csv(args.run_dir / "summary.csv")
    comparison = source_summary.merge(
        metrics[metrics["split"] == "test"].drop(columns=["split"]),
        on="lambda_rate", validate="one_to_one",
    )
    per_recording.to_csv(output_dir / "per_recording_metrics.csv", index=False)
    metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    comparison.to_csv(output_dir / "test_comparison.csv", index=False)
    rates_sorted = sorted(per_recording["lambda_rate"].unique())
    paired_bootstrap(per_recording[~per_recording["amplitude_flagged_recording"]],
                     (rates_sorted[len(rates_sorted) // 2], rates_sorted[-1])).to_csv(
        output_dir / "paired_bootstrap.csv", index=False
    )
    draw_plots(metrics, output_dir)
    draw_training(paths, checkpoints, output_dir)
    excluded_rows = []
    for split, window_set in windows.items():
        peaks = np.abs(window_set["x"]).max(axis=(1, 2))
        energies = np.square(window_set["x"]).sum(axis=(1, 2), dtype=np.float64)
        qc = pd.DataFrame({"sha256_id": window_set["ids"], "peak_abs_raw": peaks,
                           "energy_raw": energies, "above_threshold": peaks > threshold})
        grouped = qc.groupby("sha256_id", as_index=False).agg(
            max_peak_abs_raw=("peak_abs_raw", "max"),
            n_windows=("peak_abs_raw", "size"),
            n_flagged_windows=("above_threshold", "sum"),
            total_energy_raw=("energy_raw", "sum"),
        )
        grouped = grouped[grouped["n_flagged_windows"] > 0].copy()
        grouped["split"] = split
        grouped["energy_fraction_of_split"] = grouped["total_energy_raw"] / qc["energy_raw"].sum()
        excluded_rows.append(grouped)
    excluded = pd.concat(excluded_rows, ignore_index=True).sort_values(
        ["split", "energy_fraction_of_split"], ascending=[True, False]
    )
    excluded.to_csv(output_dir / "excluded_recordings.csv", index=False)
    checks = {
        "device": str(device),
        "run_dir": str(args.run_dir),
        "n_checkpoints": len(paths),
        "seed": config["seed"],
        "evaluation_windows_per_split": {split: len(windows[split]["ids"]) for split in windows},
        "amplitude_flag_threshold_raw": threshold,
        "flagged_recordings_per_split": {split: len(ids) for split, ids in flagged_ids.items()},
        "reference_match": reference_match,
        "max_abs_test_nmse_difference_from_saved": float(
            (comparison["wave_nmse_pooled"] - comparison["test_nmse"]).abs().max()
        ),
        "max_abs_test_rate_difference_from_saved": float(
            (comparison["estimated_bits_per_channel_sample"] -
             comparison["test_estimated_bits_per_channel_sample"]).abs().max()
        ),
        "excluded_energy_fraction_by_split": excluded.groupby("split")[
            "energy_fraction_of_split"].sum().to_dict(),
    }
    (output_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
