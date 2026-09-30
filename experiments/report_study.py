"""Build a reproducible summary of the saved latent and rate sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "study_report"
BANDS = ("delta", "alpha", "low_beta", "beta")
METRICS = ("wave", "delta", "theta", "alpha", "low_beta", "high_beta", "beta")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    excluded = pd.read_csv(RUNS / "combined_30_epochs_clean" / "excluded_recordings.csv")
    excluded_ids = set(excluded.sha256_id)
    validation = pd.read_csv(RUNS / "latent_sweep" / "validation_by_recording.csv")
    test = pd.read_csv(RUNS / "latent_sweep" / "test_selected_by_recording.csv")
    original_validation = pd.read_csv(RUNS / "latent_sweep" / "validation.csv")
    original_test = pd.read_csv(RUNS / "latent_sweep" / "test_selected.csv")
    clean_val = validation.loc[~validation.sha256_id.isin(excluded_ids)].copy()
    clean_test = test.loc[~test.sha256_id.isin(excluded_ids)].copy()
    clean_val.to_csv(OUT / "latent_clean_validation_by_recording.csv", index=False)
    clean_test.to_csv(OUT / "latent_clean_test_by_recording.csv", index=False)

    rows = []
    for split, frame in (("validation", clean_val), ("test_selected", clean_test)):
        for (dim, seed), group in frame.groupby(["latent_dim", "seed"], sort=True):
            row = {"split": split, "latent_dim": dim, "seed": seed,
                   "n_recordings": group.sha256_id.nunique(),
                   "n_windows": int(group.n_windows.sum())}
            for metric in METRICS:
                for kind in ("nmse", "power_ratio"):
                    values = group[f"{metric}_{kind}"]
                    row[f"{metric}_{kind}_mean_recording"] = values.mean()
                    row[f"{metric}_{kind}_median_recording"] = values.median()
            rows.append(row)
    clean_seed = pd.DataFrame(rows)
    clean_seed.to_csv(OUT / "latent_clean_seed_summary.csv", index=False)
    clean_dim = clean_seed.groupby(["split", "latent_dim"], as_index=False).mean(numeric_only=True)
    clean_dim = clean_dim.drop(columns=["seed"])
    clean_dim.to_csv(OUT / "latent_clean_dimension_summary.csv", index=False)

    paired = clean_val.groupby(["sha256_id", "latent_dim"])[
        [f"{m}_nmse" for m in ("wave", *BANDS)]].mean()
    rng = np.random.default_rng(42)
    comparisons = []
    for lower, higher in ((8, 16), (16, 32), (32, 64), (64, 128), (128, 256)):
        left = paired.xs(lower, level="latent_dim").sort_index()
        right = paired.xs(higher, level="latent_dim").sort_index()
        if not left.index.equals(right.index):
            raise ValueError("Latent widths have different recording IDs")
        for metric in ("wave", *BANDS):
            difference = (left[f"{metric}_nmse"] - right[f"{metric}_nmse"]).to_numpy()
            samples = rng.integers(0, len(difference), (3000, len(difference)))
            low, high = np.quantile(difference[samples].mean(axis=1), [.025, .975])
            comparisons.append({"lower_dim": lower, "higher_dim": higher, "metric": metric,
                                "mean_nmse_reduction": difference.mean(),
                                "ci95_low": low, "ci95_high": high,
                                "n_recordings": len(difference)})
    pd.DataFrame(comparisons).to_csv(OUT / "latent_paired_bootstrap.csv", index=False)

    val_dim = clean_dim.loc[clean_dim.split.eq("validation")].set_index("latent_dim")
    reference = {b: val_dim[f"{b}_nmse_mean_recording"].min() for b in ("delta", "alpha", "low_beta")}
    eligible = [int(dim) for dim, row in val_dim.iterrows()
                if all(row[f"{band}_nmse_mean_recording"] <= reference[band] + .05
                       for band in reference)]
    posthoc_selected = min(eligible) if eligible else None

    bitrate = pd.read_csv(RUNS / "combined_30_epochs_clean" / "clean_test_summary.csv")
    bitrate.to_csv(OUT / "bitrate_clean_test_summary.csv", index=False)

    colors = {"delta": "#2b6cb0", "alpha": "#d69e2e", "low_beta": "#b83280", "beta": "#805ad5"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for band in BANDS:
        y = val_dim[f"{band}_nmse_mean_recording"]
        axes[0].plot(val_dim.index, y, marker="o", label=band.replace("_", " "), color=colors[band])
        p = val_dim[f"{band}_power_ratio_median_recording"]
        axes[1].plot(val_dim.index, p, marker="o", label=band.replace("_", " "), color=colors[band])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(val_dim.index, [str(x) for x in val_dim.index])
        ax.axvline(32, color="0.5", linestyle="--", linewidth=1)
        ax.grid(alpha=.25)
        ax.set_xlabel("Continuous latent channels")
    axes[0].set_ylabel("Mean recording band NMSE")
    axes[1].set_ylabel("Median recording reconstructed / original power")
    axes[0].legend(ncol=2, frameon=False)
    fig.suptitle("Latent width: retained validation recordings (252 EEGs, 3 seeds)")
    fig.savefig(OUT / "latent_clean_validation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bitrate = bitrate.sort_values("test_bits_per_channel_sample")
    axes[0].plot(bitrate.test_bits_per_channel_sample, bitrate.test_wave_nmse_pooled,
                 "o-", markersize=3, color="#2b6cb0")
    axes[0].set_ylabel("Pooled test waveform NMSE")
    for band in BANDS:
        axes[1].plot(bitrate.test_bits_per_channel_sample,
                     bitrate[f"{band}_nmse_median_recording"], "o-", markersize=3,
                     label=band.replace("_", " "), color=colors[band])
    axes[1].set_ylabel("Median test recording band NMSE")
    axes[1].legend(ncol=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=.25)
        ax.set_xlabel("Estimated bits per channel-sample")
    fig.suptitle("Quantized rate sweep: retained test recordings (250 EEGs)")
    fig.savefig(OUT / "bitrate_clean_test.png", dpi=180)
    plt.close(fig)

    print("CLEAN VALIDATION DIMENSIONS")
    print(val_dim[[f"{m}_nmse_mean_recording" for m in ("wave", *BANDS)]].round(4).to_string())
    print("POSTHOC SELECTED", posthoc_selected, "REFERENCES", reference)
    print("CLEAN TEST SELECTED")
    print(clean_dim.loc[clean_dim.split.eq("test_selected"),
                        ["latent_dim"] + [f"{m}_nmse_mean_recording" for m in ("wave", *BANDS)]].round(4).to_string(index=False))
    print("ORIGINAL TEST SELECTED")
    print(original_test[["seed"] + [f"{m}_nmse" for m in ("wave", *BANDS)]].round(4).to_string(index=False))
    print("VALIDATION SEED RANGE")
    print(clean_seed.loc[clean_seed.split.eq("validation")].groupby("latent_dim")[[f"{b}_nmse_mean_recording" for b in BANDS]].agg(["min", "max"]).round(4).to_string())
    print("ORIGINAL VALIDATION")
    print(original_validation.groupby("latent_dim")[[f"{m}_nmse" for m in ("wave", *BANDS)]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
