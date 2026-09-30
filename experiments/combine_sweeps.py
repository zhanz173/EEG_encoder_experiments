"""Combine two analyzed EEG rate sweeps on the same unflagged evaluation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_run(run_dir: Path, order: int):
    analysis = run_dir / "report_analysis"
    summary = pd.read_csv(run_dir / "summary.csv")
    metrics = pd.read_csv(analysis / "split_metrics.csv")
    recordings = pd.read_csv(analysis / "per_recording_metrics.csv")
    excluded = pd.read_csv(analysis / "excluded_recordings.csv")
    checks = json.loads((analysis / "checks.json").read_text(encoding="utf-8"))
    clean_columns = [name for name in metrics if name.endswith("_unflagged")]
    cleaned = metrics[["lambda_rate", "split", "n_recordings",
                       "n_amplitude_flagged_recordings", *clean_columns]].copy()
    cleaned["n_recordings"] -= cleaned["n_amplitude_flagged_recordings"]
    cleaned = cleaned.rename(columns={name: name.removesuffix("_unflagged")
                                      for name in clean_columns})
    cleaned = cleaned.rename(columns={"n_amplitude_flagged_recordings": "n_excluded_recordings"})
    windows_per_recording = checks["evaluation_windows_per_split"]["test"] // 256
    cleaned["n_windows"] = cleaned["n_recordings"] * windows_per_recording
    cleaned = cleaned.merge(summary[["lambda_rate", "run_dir", "best_epoch", "epochs_ran",
                                     "stopped_early"]], on="lambda_rate", validate="many_to_one")
    cleaned.insert(0, "source_run", run_dir.name)
    cleaned.insert(1, "run_order", order)
    recordings = recordings[~recordings["amplitude_flagged_recording"]].copy()
    recordings.insert(0, "source_run", run_dir.name)
    recordings.insert(1, "run_order", order)
    return cleaned, recordings, excluded, checks


def endpoint_bootstrap(recordings: pd.DataFrame, first_name: str, second_name: str,
                       n_bootstrap: int = 5000):
    rng = np.random.default_rng(20260921)
    rows = []
    for split in ("val", "test"):
        first = recordings[(recordings["source_run"] == first_name) &
                           (recordings["split"] == split) &
                           (recordings["lambda_rate"] == 0.01)]
        second = recordings[(recordings["source_run"] == second_name) &
                            (recordings["split"] == split) &
                            (recordings["lambda_rate"] == 1.0)]
        paired = first.merge(second, on="sha256_id", suffixes=("_first", "_second"),
                             validate="one_to_one")
        if len(paired) not in (250, 252):
            raise ValueError(f"Unexpected paired cleaned cohort size for {split}: {len(paired)}")
        indices = rng.integers(0, len(paired), size=(n_bootstrap, len(paired)))
        for name in ("wave_nmse", "delta_nmse", "alpha_nmse", "low_beta_nmse", "beta_nmse"):
            before = paired[f"{name}_first"].to_numpy()
            after = paired[f"{name}_second"].to_numpy()
            differences = np.median(after[indices], axis=1) - np.median(before[indices], axis=1)
            rows.append({"split": split, "metric": name, "aggregation": "median recording",
                         "n_recordings": len(paired),
                         "estimate_difference": float(np.median(after) - np.median(before)),
                         "ci_low": float(np.quantile(differences, 0.025)),
                         "ci_high": float(np.quantile(differences, 0.975)),
                         "n_bootstrap": n_bootstrap})
        name = "estimated_bits_per_channel_sample"
        before = paired[f"{name}_first"].to_numpy()
        after = paired[f"{name}_second"].to_numpy()
        differences = (after[indices] - before[indices]).mean(axis=1)
        rows.append({"split": split, "metric": name, "aggregation": "mean recording",
                     "n_recordings": len(paired),
                     "estimate_difference": float(after.mean() - before.mean()),
                     "ci_low": float(np.quantile(differences, 0.025)),
                     "ci_high": float(np.quantile(differences, 0.975)),
                     "n_bootstrap": n_bootstrap})
    return pd.DataFrame(rows)


def plot_results(cleaned: pd.DataFrame, output_dir: Path):
    runs = list(cleaned.sort_values("run_order")["source_run"].drop_duplicates())
    styles = {runs[0]: ("o", "tab:blue"), runs[1]: ("s", "tab:orange")}
    labels = {runs[0]: "λ 0.01–0.50", runs[1]: "λ 0.50–1.00"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for run in runs:
        marker, color = styles[run]
        for split, linestyle in (("test", "-"), ("val", "--")):
            frame = cleaned[(cleaned["source_run"] == run) &
                            (cleaned["split"] == split)].sort_values("lambda_rate")
            axes[0].plot(frame["estimated_bits_per_channel_sample"],
                         frame["wave_nmse_pooled"], marker=marker,
                         linestyle=linestyle, color=color,
                         label=f"{labels[run]}: {split}", markersize=4)
        frame = cleaned[(cleaned["source_run"] == run) &
                        (cleaned["split"] == "test")].sort_values("lambda_rate")
        axes[1].plot(frame["estimated_bits_per_channel_sample"],
                     frame["wave_nmse_median_recording"], marker=marker,
                     color=color, label=labels[run], markersize=4)
    axes[0].set(xlabel="Estimated bits / channel-sample", ylabel="Pooled waveform NMSE",
                title="Validation and test")
    axes[1].set(xlabel="Estimated bits / channel-sample", ylabel="Median recording waveform NMSE",
                title="Test recordings")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(output_dir / "clean_rate_distortion.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bands = (("delta", "Delta 1–4 Hz"), ("alpha", "Alpha 8–13 Hz"),
             ("low_beta", "Low beta 13–20 Hz"), ("beta", "Beta 13–30 Hz"))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for (name, label), color in zip(bands, colors):
        for run in runs:
            frame = cleaned[(cleaned["source_run"] == run) &
                            (cleaned["split"] == "test")].sort_values("lambda_rate")
            marker, _ = styles[run]
            axes[0].plot(frame["estimated_bits_per_channel_sample"],
                         frame[f"{name}_nmse_median_recording"], marker=marker,
                         color=color, markersize=4)
            axes[1].plot(frame["lambda_rate"],
                         frame[f"{name}_nmse_median_recording"], marker=marker,
                         color=color, markersize=4)
        axes[1].plot([], [], color=color, label=label)
    axes[0].set(xlabel="Estimated bits / channel-sample", ylabel="Median recording band NMSE",
                title="Band error versus rate")
    axes[1].set(xlabel="λ", ylabel="Median recording band NMSE",
                title="Band error versus rate penalty")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_dir / "clean_band_reconstruction.png", dpi=180)
    plt.close(fig)

    test = cleaned[cleaned["split"] == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    measures = (("log_psd_mae", "Log PSD MAE"),
                ("spatial_corr_error", "Spatial correlation error"),
                ("line_length_rel_error", "Line length relative error"),
                ("peak_rel_error", "Peak relative error"))
    for ax, (column, title) in zip(axes.flat, measures):
        for run in runs:
            frame = test[test["source_run"] == run].sort_values("lambda_rate")
            marker, color = styles[run]
            ax.plot(frame["lambda_rate"], frame[column], marker=marker, color=color,
                    markersize=4, label=labels[run])
        ax.set(xlabel="λ", ylabel=title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output_dir / "clean_secondary_metrics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    one = read_run(args.first, 0)
    two = read_run(args.second, 1)
    for key in ("split_ids_match", "scale_match", "config_match_except_lambda_and_output"):
        if one[3]["reference_match"][key] is not True or two[3]["reference_match"][key] is not True:
            raise ValueError(f"The two runs are not directly comparable: {key}")
    if not np.isclose(one[3]["amplitude_flag_threshold_raw"],
                      two[3]["amplitude_flag_threshold_raw"], atol=1e-12, rtol=0):
        raise ValueError("Outlier thresholds differ between runs")
    excluded_one = one[2].sort_values(["split", "sha256_id"]).reset_index(drop=True)
    excluded_two = two[2].sort_values(["split", "sha256_id"]).reset_index(drop=True)
    if not excluded_one[["split", "sha256_id"]].equals(excluded_two[["split", "sha256_id"]]):
        raise ValueError("Excluded recording IDs differ between runs")
    cleaned = pd.concat([one[0], two[0]], ignore_index=True).sort_values(
        ["lambda_rate", "run_order", "split"]
    )
    recordings = pd.concat([one[1], two[1]], ignore_index=True).sort_values(
        ["lambda_rate", "run_order", "split", "sha256_id"]
    )
    if cleaned.groupby(["source_run", "lambda_rate", "split"]).size().ne(1).any():
        raise ValueError("Duplicate or missing sweep points")
    if cleaned.groupby("split")["n_recordings"].nunique().ne(1).any():
        raise ValueError("Clean cohort size varies across checkpoints")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "clean_split_results.csv", index=False)
    recordings.to_csv(output_dir / "clean_per_recording_results.csv", index=False)
    excluded_one.to_csv(output_dir / "excluded_recordings.csv", index=False)
    val = cleaned[cleaned["split"] == "val"][[
        "source_run", "lambda_rate", "estimated_bits_per_channel_sample",
        "wave_nmse_pooled", "alpha_nmse_median_recording", "low_beta_nmse_median_recording"
    ]].rename(columns={
        "estimated_bits_per_channel_sample": "val_bits_per_channel_sample",
        "wave_nmse_pooled": "val_wave_nmse_pooled",
        "alpha_nmse_median_recording": "val_alpha_nmse_median_recording",
        "low_beta_nmse_median_recording": "val_low_beta_nmse_median_recording",
    })
    test = cleaned[cleaned["split"] == "test"].drop(columns=["split"])
    compact = test.merge(val, on=["source_run", "lambda_rate"], validate="one_to_one")
    compact = compact.rename(columns={
        "estimated_bits_per_channel_sample": "test_bits_per_channel_sample",
        "estimated_bits_per_second": "test_bits_per_second",
        "wave_nmse_pooled": "test_wave_nmse_pooled",
        "wave_nmse_median_recording": "test_wave_nmse_median_recording",
        "snr_db_pooled": "test_snr_db_pooled",
    })
    compact.to_csv(output_dir / "clean_test_summary.csv", index=False)
    endpoint_bootstrap(recordings, args.first.name, args.second.name).to_csv(
        output_dir / "endpoint_bootstrap.csv", index=False
    )
    plot_results(cleaned, output_dir)
    checks = {
        "first_run": str(args.first), "second_run": str(args.second),
        "n_checkpoints": int(len(cleaned) // 2),
        "n_clean_recordings_by_split": cleaned.groupby("split")["n_recordings"].first().to_dict(),
        "n_clean_windows_by_split": cleaned.groupby("split")["n_windows"].first().to_dict(),
        "n_excluded_recordings_by_split": excluded_one.groupby("split").size().to_dict(),
        "amplitude_flag_threshold_raw": one[3]["amplitude_flag_threshold_raw"],
        "max_source_test_nmse_reconciliation_error": max(
            one[3]["max_abs_test_nmse_difference_from_saved"],
            two[3]["max_abs_test_nmse_difference_from_saved"],
        ),
        "max_source_test_rate_reconciliation_error": max(
            one[3]["max_abs_test_rate_difference_from_saved"],
            two[3]["max_abs_test_rate_difference_from_saved"],
        ),
    }
    (output_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
