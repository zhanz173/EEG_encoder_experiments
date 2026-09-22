"""Sweep continuous-AE latent width and analyze EEG frequency-band retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from band_metrics import BANDS, evaluate_continuous_ae, evaluate_wave_nmse
from eeg_dataset import EEGWindowDataset
from experiment_utils import (
    estimate_input_scale, make_loader, read_manifest, seed_everything, split_records,
)
from model import EEGContinuousAE, waveform_nmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", default="runs/latent_sweep")
    parser.add_argument("--group-col", default="sha256_id",
                        help="Use a patient ID column if available; default splits recordings")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Independent training seeds; use at least 3 for a serious comparison")
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--train-windows-per-recording", type=int, default=4)
    parser.add_argument("--eval-windows-per-recording", type=int, default=4)
    parser.add_argument("--eval-start-sec", type=float, default=0.0)
    parser.add_argument("--max-train-recordings", type=int, default=256)
    parser.add_argument("--max-val-recordings", type=int, default=64)
    parser.add_argument("--max-test-recordings", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--nmse-tolerance", type=float, default=0.05,
                        help="Absolute NMSE margin above the best observed validation result per band")
    parser.add_argument("--max-acceptable-band-nmse", type=float, default=0.8,
                        help="Do not select a final width if the best model exceeds this NMSE in a required band")
    parser.add_argument("--selection-bands", nargs="+", default=["delta", "alpha", "low_beta"],
                        choices=list(BANDS))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if (not args.dims or any(dim < 1 for dim in args.dims) or
            not args.seeds or args.epochs < 1 or args.batch_size < 1 or args.nmse_tolerance < 0 or
            args.max_acceptable_band_nmse <= 0):
        parser.error("dims, seeds, epochs, and batch-size must be positive; tolerance must be nonnegative")
    if len(set(args.dims)) != len(args.dims) or len(set(args.seeds)) != len(args.seeds):
        parser.error("dims and seeds must not contain duplicates")
    if round(args.window_sec * args.sfreq) % 16:
        parser.error("window-sec × sfreq must be divisible by 16")
    if args.eval_start_sec < 0:
        parser.error("eval-start-sec must be nonnegative")
    if args.sfreq / 2 <= max(high for _, high in BANDS.values()):
        parser.error("Sampling rate is too low to evaluate the requested bands")
    args.dims.sort()
    return args


def train_one(
    dim: int, seed: int, args: argparse.Namespace, train_ds: EEGWindowDataset,
    val_loader, scale: float, device: torch.device, run_dir: Path,
) -> tuple[dict, list[dict]]:
    seed_everything(seed)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True)
    model = EEGContinuousAE(args.channels, dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_nmse = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        steps = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True) / scale
            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            loss = waveform_nmse(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at dimension {dim}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.detach().item()
            steps += 1
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
        val_nmse = evaluate_wave_nmse(model, val_loader, scale, device)
        history.append({"epoch": epoch, "train_wave_nmse": train_total / steps,
                        "val_wave_nmse": val_nmse})
        print(f"dim={dim:3d} seed={seed:3d} epoch={epoch:2d} "
              f"train={train_total / steps:.4f} val={val_nmse:.4f}", flush=True)
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            temporary = run_dir / "best.tmp"
            torch.save({"model": model.state_dict(), "latent_dim": dim, "seed": seed,
                        "epoch": epoch, "scale": scale}, temporary)
            temporary.replace(run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val, recording_rows = evaluate_continuous_ae(model, val_loader, scale, device)
    return {"latent_dim": dim, "seed": seed, "best_epoch": checkpoint["epoch"], **val}, [
        {"latent_dim": dim, "seed": seed, **row} for row in recording_rows
    ]


def analyze(validation: pd.DataFrame, args: argparse.Namespace) -> dict:
    means = validation.groupby("latent_dim", sort=True).mean(numeric_only=True)
    reference = {band: float(means[f"{band}_nmse"].min()) for band in args.selection_bands}
    eligible = [int(dim) for dim, row in means.iterrows()
                if all(row[f"{band}_nmse"] <= reference[band] + args.nmse_tolerance
                       for band in args.selection_bands)]
    candidate = min(eligible) if eligible else None
    undertrained = any(value >= args.max_acceptable_band_nmse for value in reference.values())
    selected = None if undertrained else candidate
    cliffs = {}
    ordered = means.sort_index(ascending=False)
    for band in BANDS:
        candidates = []
        dimensions = list(ordered.index)
        for higher, lower in zip(dimensions[:-1], dimensions[1:]):
            increase = float(ordered.loc[lower, f"{band}_nmse"] -
                             ordered.loc[higher, f"{band}_nmse"])
            candidates.append({"from_dim": int(higher), "to_dim": int(lower),
                               "nmse_increase": increase})
        largest = max(candidates, key=lambda row: row["nmse_increase"]) if candidates else None
        cliffs[band] = largest if largest and largest["nmse_increase"] > 0 else None

    # Normalize each band by the reconstruction gain of its own best dimension.
    # A value near one retains the best observed gain; zero is a zero-signal decoder.
    retention = []
    all_bands = ("delta", "alpha", "low_beta", "beta")
    best_per_band = {band: float(means[f"{band}_nmse"].min()) for band in all_bands}
    for dim, row in means.iterrows():
        item = {"latent_dim": int(dim)}
        for band in all_bands:
            gain = 1.0 - best_per_band[band]
            item[f"{band}_relative_gain"] = (
                float((1.0 - row[f"{band}_nmse"]) / gain) if gain > 0.05 else None
            )
        if item["delta_relative_gain"] is not None and item["alpha_relative_gain"] is not None:
            item["delta_minus_alpha_relative_gain"] = (
                item["delta_relative_gain"] - item["alpha_relative_gain"])
        if item["delta_relative_gain"] is not None and item["low_beta_relative_gain"] is not None:
            item["delta_minus_low_beta_relative_gain"] = (
                item["delta_relative_gain"] - item["low_beta_relative_gain"])
        retention.append(item)
    return {
        "selection_rule": "Smallest dimension within tolerance of the best observed validation NMSE in every selected band",
        "selection_bands": args.selection_bands,
        "nmse_tolerance": args.nmse_tolerance,
        "max_acceptable_band_nmse": args.max_acceptable_band_nmse,
        "best_observed_band_nmse": reference,
        "candidate_dim": candidate,
        "selected_dim": selected,
        "largest_adjacent_drops_candidate": cliffs,
        "relative_gain_by_dim": retention,
        "interpretation_warning": (
            "The best model still exceeds the acceptable NMSE in at least one selection band; "
            "train longer before interpreting a cliff or selected width."
            if undertrained else None
        ),
    }


def plot_validation(validation: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = validation.groupby("latent_dim", sort=True).mean(numeric_only=True)
    std = validation.groupby("latent_dim", sort=True).std(numeric_only=True).fillna(0)
    colors = {"delta": "#275DAB", "alpha": "#DF783B", "low_beta": "#8B55A4", "beta": "#2A946B"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for band, color in colors.items():
        for axis, suffix in zip(axes, ("nmse", "power_ratio")):
            key = f"{band}_{suffix}"
            axis.errorbar(means.index, means[key], yerr=std[key], marker="o",
                          capsize=3, color=color, label=band.replace("_", " "))
    axes[0].axhline(1.0, color="0.7", linestyle="--", linewidth=1)
    axes[1].axhline(1.0, color="0.7", linestyle="--", linewidth=1)
    any_gain = False
    for band, color in colors.items():
        best = float(means[f"{band}_nmse"].min())
        if 1.0 - best > 0.05:
            any_gain = True
            relative_gain = (1.0 - means[f"{band}_nmse"]) / (1.0 - best)
            axes[2].plot(means.index, relative_gain, marker="o", color=color,
                         label=band.replace("_", " "))
    axes[2].axhline(1.0, color="0.7", linestyle="--", linewidth=1)
    if not any_gain:
        axes[2].text(0.5, 0.5, "Gain undefined: models have not learned\nthese bands yet",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Band-limited waveform NMSE (lower is better)")
    axes[1].set_ylabel("Reconstructed / original band power")
    axes[2].set_ylabel("Fraction of best band gain retained")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(means.index)
        axis.set_xticklabels([str(int(dim)) for dim in means.index])
        axis.set_xlabel("Continuous latent channels at 16× temporal stride")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Validation band retention across latent dimensions")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.manifest = str(Path(args.manifest).resolve())
    args.shards_dir = str(Path(args.shards_dir).resolve())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.group_col == "sha256_id":
        print("WARNING: sha256_id separates recordings, not patients. Pass a patient ID column for patient-level conclusions.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new directory for this sweep.")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.split_seed)
    records = read_manifest(args.manifest, args.sfreq, args.channels, args.window_sec)
    splits = split_records(records, args.group_col, args.split_seed,
                           args.max_train_recordings, args.max_val_recordings,
                           args.max_test_recordings)
    train_ds = EEGWindowDataset(splits["train"], args.shards_dir, args.window_sec,
                                args.train_windows_per_recording, True, args.channels, args.sfreq)
    val_ds = EEGWindowDataset(splits["val"], args.shards_dir, args.window_sec,
                              args.eval_windows_per_recording, False, args.channels, args.sfreq,
                              eval_start_sec=args.eval_start_sec)
    test_ds = EEGWindowDataset(splits["test"], args.shards_dir, args.window_sec,
                               args.eval_windows_per_recording, False, args.channels, args.sfreq,
                               eval_start_sec=args.eval_start_sec)
    scale = estimate_input_scale(train_ds)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, False)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, False)
    config = {**vars(args), "input_scale": scale, "device_used": str(device),
              "split_ids": {name: split["sha256_id"].astype(str).tolist()
                            for name, split in splits.items()}}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device={device}; scale={scale:.6g}; split recordings="
          f"{ {name: len(split) for name, split in splits.items()} }", flush=True)

    validation_rows = []
    recording_rows = []
    for dim in args.dims:
        for seed in args.seeds:
            run_dir = output_dir / f"dim_{dim:03d}" / f"seed_{seed}"
            summary, by_recording = train_one(dim, seed, args, train_ds, val_loader,
                                              scale, device, run_dir)
            validation_rows.append(summary)
            recording_rows.extend(by_recording)
            pd.DataFrame(validation_rows).to_csv(output_dir / "validation.csv", index=False)
            pd.DataFrame(recording_rows).to_csv(output_dir / "validation_by_recording.csv", index=False)

    validation = pd.DataFrame(validation_rows)
    result = analyze(validation, args)
    (output_dir / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_validation(validation, output_dir / "validation_bands.png")
    print("Selection:", json.dumps(result, indent=2), flush=True)

    if result["selected_dim"] is not None:
        test_rows = []
        test_recording_rows = []
        for seed in args.seeds:
            path = output_dir / f"dim_{result['selected_dim']:03d}" / f"seed_{seed}" / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = EEGContinuousAE(args.channels, result["selected_dim"]).to(device)
            model.load_state_dict(checkpoint["model"])
            summary, by_recording = evaluate_continuous_ae(model, test_loader, scale, device)
            test_rows.append({"latent_dim": result["selected_dim"], "seed": seed, **summary})
            test_recording_rows.extend(
                {"latent_dim": result["selected_dim"], "seed": seed, **row}
                for row in by_recording
            )
        pd.DataFrame(test_rows).to_csv(output_dir / "test_selected.csv", index=False)
        pd.DataFrame(test_recording_rows).to_csv(output_dir / "test_selected_by_recording.csv", index=False)
    train_ds.close()
    val_ds.close()
    test_ds.close()


if __name__ == "__main__":
    main()
