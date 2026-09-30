"""Collect validation/test curves; never automatically select a test operating point."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.spatial_data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, originals = [], []
    for path in Path(args.root).rglob("comparison.csv"):
        config_path = path.parent/"evaluation_config.json"
        config = json.loads(config_path.read_text())
        df = pd.read_csv(path)
        df["source"] = str(path)
        df["split"] = config["split"]
        df["checkpoint_sha256"] = config["checkpoint_sha256"]
        df["untouched_only"] = config["untouched_only"]
        df["data_hash"] = config.get("data_hash", str(path))
        df["architecture"] = config.get("model_config", {}).get("architecture", "unknown")
        df["rank"] = config.get("model_config", {}).get("rank", -1)
        df["training_seed"] = config.get("training_seed", -1)
        df["lambda_rate"] = config.get("lambda_rate", float("nan"))
        frames.append(df)
        original = json.loads((path.parent/"original/summary.json").read_text())
        original.update({k: df.iloc[0][k] for k in ("source", "split", "checkpoint_sha256", "untouched_only",
                                                   "data_hash", "architecture", "rank", "training_seed", "lambda_rate")})
        originals.append(original)
    if not frames:
        raise ValueError("No post-hoc comparison.csv files found")
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out/"all_comparisons.csv", index=False)
    original_frame = pd.DataFrame(originals)
    original_frame.to_csv(out/"original_models.csv", index=False)
    # Never combine different prepared datasets, evaluation splits, or QC subsets.
    for panel, ((data_hash, split, untouched), group) in enumerate(all_rows.groupby(["data_hash", "split", "untouched_only"])):
        reference = original_frame[(original_frame.data_hash == data_hash) & (original_frame.split == split) &
                                   (original_frame.untouched_only == untouched)].drop_duplicates("checkpoint_sha256")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for (architecture, rank, seed), g in reference.groupby(["architecture", "rank", "training_seed"]):
            g = g.sort_values("rate_total")
            label = f"{architecture}, K={rank}, seed={seed}: original"
            ax.plot(g.rate_total, g.wave_nmse, "o-", label=label)
        for architecture, g in group.groupby("architecture"):
            ax.scatter(g.rate_total, g.wave_nmse, s=12, alpha=.35, label=f"{architecture}: dictionary variants")
        ax.set(xlabel="Estimated total bits/channel-sample (shared dictionary excluded)",
               ylabel="Mean patient waveform NMSE", title=f"{split}: original models and frozen dictionary replacements")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.savefig(out/f"combined_rate_distortion_{panel:03d}.png", dpi=150); plt.close(fig)
    selections = []
    for source, group in all_rows.groupby("source", sort=False):
        dest = out/f"figure_{len(selections):03d}_{group.iloc[0]['checkpoint_sha256'][:8]}_{group.iloc[0]['split']}"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for m, g in group.groupby("requested_m"):
            g = g.sort_values("update_ms")
            axes[0].scatter(g.rate_total, g.wave_nmse, label=f"M={m}", s=20)
            held = g[g.rule.eq("distortion")]
            axes[1].plot(held.update_ms, held.delta_wave_nmse, "o-", label=f"M={m}")
        nearest = group[group.rule.eq("nearest")].sort_values("requested_m")
        for band in ("delta", "alpha", "low_beta", "high_beta"):
            axes[2].plot(nearest.requested_m, nearest[f"delta_{band}_nmse"], "o-", label=band)
        axes[0].set(xlabel="Estimated total bits/channel-sample", ylabel="Mean patient waveform NMSE")
        axes[1].set(xlabel="Spatial hold interval (ms)", ylabel="Increase in waveform NMSE", xscale="log")
        axes[1].axhline(.01, linestyle="--", color="gray")
        axes[2].set(xlabel="Requested dictionary size", ylabel="Increase in band NMSE", xscale="log",)
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.savefig(str(dest)+".png", dpi=150); plt.close(fig)
        # One inspectable failure example per checkpoint/evaluation, in source units.
        worst_case = group.sort_values("delta_wave_nmse", ascending=False).iloc[0]["case"]
        example_path = Path(source).parent/worst_case/"worst_waveform.npz"
        if example_path.exists():
            with np.load(example_path, allow_pickle=False) as example:
                x, original, replacement = example["x"], example["original"], example["replacement"]
                channels = np.argsort(((x-replacement)**2).mean(1))[-3:][::-1]
                time_axis = np.arange(x.shape[1])/float(example["sfreq"])
                fig, axes = plt.subplots(len(channels), 1, figsize=(12, 6), sharex=True, constrained_layout=True)
                for ax, channel in zip(np.atleast_1d(axes), channels):
                    ax.plot(time_axis, x[channel], color=".35", label="Input", linewidth=.8)
                    ax.plot(time_axis, original[channel], label="Original reconstruction", linewidth=.8)
                    ax.plot(time_axis, replacement[channel], label="Dictionary replacement", linewidth=.8)
                    ax.set_ylabel(f"Channel {channel}")
                axes[0].legend(ncol=3, fontsize=8)
                axes[0].set_title(f"Largest mean NMSE increase: {worst_case}; worst window")
                axes[-1].set_xlabel("Seconds within crop")
                fig.savefig(str(dest)+"_waveform.png", dpi=150); plt.close(fig)
        # Only validation produces a candidate, and never mixes checkpoints or splits.
        selection = dict(source=source, split=group.iloc[0]["split"], selected_case=None)
        if selection["split"] == "val":
            valid = group[group.passes_proposed_tolerance & group.rule.eq("nearest")]
            if len(valid):
                selection["selected_case"] = valid.sort_values(["effective_m", "rate_total"]).iloc[0]["case"]
        selections.append(selection)
    write_json(out/"validation_candidates.json", selections)
    print(f"Saved {out/'all_comparisons.csv'} and {len(selections)} figures")


if __name__ == "__main__":
    main()
