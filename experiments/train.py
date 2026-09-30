"""Train one EEG rate–distortion model on a small recording subset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from utils.eeg_dataset import EEGWindowDataset
from utils.experiment_utils import (
    estimate_input_scale, evaluate, make_loader, read_manifest, seed_everything,
    split_records,
)
from models.model import waveform_huber
from models.temporal_model import build_rate_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True, help="CSV or Parquet recordings manifest")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", default="runs/quick")
    parser.add_argument("--group-col", default="sha256_id",
                        help="Patient ID column if available; sha256_id gives a recording-level split")
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--train-windows-per-recording", type=int, default=4)
    parser.add_argument("--eval-windows-per-recording", type=int, default=4)
    parser.add_argument("--eval-start-sec", type=float, default=0.0,
                        help="First allowed validation/test window start, in seconds")
    parser.add_argument("--max-train-recordings", type=int, default=256)
    parser.add_argument("--max-val-recordings", type=int, default=64)
    parser.add_argument("--max-test-recordings", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--architecture", choices=["baseline", "slow-fast"], default="baseline")
    parser.add_argument("--channel-mode", choices=["joint", "independent"], default="joint",
                        help="Slow-fast channel handling; baseline always uses joint channels")
    parser.add_argument("--slow-dim", type=int, default=4, help="Slow features, joint or per channel according to channel-mode")
    parser.add_argument("--fast-dim", type=int, default=4, help="Fast features, joint or per channel according to channel-mode")
    parser.add_argument("--slow-stride", type=int, default=64)
    parser.add_argument("--fast-stride", type=int, default=16)
    parser.add_argument("--temporal-width", type=int, default=32)
    parser.add_argument("--slow-loss-weight", type=float, default=0.25,
                        help="Auxiliary slow-only Huber weight for slow-fast models")
    parser.add_argument("--lambda-rate", type=float, default=0.01)
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber transition in units of each input window's RMS")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many epochs without meaningful validation improvement; 0 disables")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001,
                        help="Minimum decrease in validation objective that resets patience")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="0 means the entire selected training subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args(argv)
    if not math.isfinite(args.lambda_rate) or args.lambda_rate < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("lambda-rate must be nonnegative; epochs and batch-size must be positive")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0 or args.eval_start_sec < 0:
        parser.error("early-stop-patience, early-stop-min-delta, and eval-start-sec must be nonnegative")
    if not math.isfinite(args.huber_delta) or args.huber_delta <= 0:
        parser.error("huber-delta must be positive and finite")
    if args.architecture == "baseline" and round(args.window_sec * args.sfreq) % 16:
        parser.error("window-sec × sfreq must be divisible by 16")
    if not math.isfinite(args.slow_loss_weight) or args.slow_loss_weight < 0:
        parser.error("slow-loss-weight must be finite and nonnegative")
    if args.architecture == "slow-fast":
        if (min(args.slow_dim, args.fast_dim, args.temporal_width) < 1
                or args.temporal_width % 8
                or any(s < 2 or s & (s - 1) for s in (args.slow_stride, args.fast_stride))
                or args.slow_stride <= args.fast_stride):
            parser.error("Positive dimensions, width divisible by 8, and power-of-two strides slow > fast >= 2 required")
        if round(args.window_sec * args.sfreq) < args.slow_stride or round(args.window_sec * args.sfreq) % args.slow_stride:
            parser.error("window-sec × sfreq must be a positive multiple of slow-stride")
    return args


def main() -> None:
    args = parse_args()
    args.manifest = str(Path(args.manifest).resolve())
    args.shards_dir = str(Path(args.shards_dir).resolve())
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.group_col == "sha256_id":
        print("WARNING: sha256_id splits recordings, not patients. Supply --group-col PATIENT_COLUMN for patient-level evaluation.")
    if args.lambda_rate == 0:
        print("WARNING: lambda-rate=0 does not train the entropy prior; reported rate is uncalibrated.")

    records = read_manifest(args.manifest, args.sfreq, args.channels, args.window_sec)
    splits = split_records(records, args.group_col, args.seed,
                           args.max_train_recordings, args.max_val_recordings,
                           args.max_test_recordings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["distortion_loss"] = "normalized_huber"
    config["device_used"] = str(device)
    config["split_recording_counts"] = {key: len(value) for key, value in splits.items()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = EEGWindowDataset(splits["train"], args.shards_dir, args.window_sec,
                                args.train_windows_per_recording, True, args.channels, args.sfreq)
    val_ds = EEGWindowDataset(splits["val"], args.shards_dir, args.window_sec,
                              args.eval_windows_per_recording, False, args.channels, args.sfreq,
                              eval_start_sec=args.eval_start_sec)
    test_ds = EEGWindowDataset(splits["test"], args.shards_dir, args.window_sec,
                               args.eval_windows_per_recording, False, args.channels, args.sfreq,
                               eval_start_sec=args.eval_start_sec)
    scale = estimate_input_scale(train_ds)
    print(f"Device={device}; recordings={config['split_recording_counts']}; fixed input scale={scale:.6g}")
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, False)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, False)

    model_config = {"n_channels": args.channels, "latent_dim": args.latent_dim}
    if args.architecture == "slow-fast":
        model_config = dict(architecture="slow-fast", n_channels=args.channels,
                            slow_dim=args.slow_dim, fast_dim=args.fast_dim,
                            slow_stride=args.slow_stride, fast_stride=args.fast_stride,
                            width=args.temporal_width, channel_mode=args.channel_mode)
    model = build_rate_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    split_ids = {key: value["sha256_id"].astype(str).tolist() for key, value in splits.items()}
    best_score = float("inf")
    best_epoch = 0
    progress_score = float("inf")
    stale_epochs = 0
    epochs_ran = 0
    stopped_early = False
    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        steps = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True) / scale
            optimizer.zero_grad(set_to_none=True)
            slow_loss = x.new_zeros(())
            if args.architecture == "slow-fast":
                components = model.forward_components(x)
                reconstructed, rate = components["reconstruction"], components["rate"]
                slow_loss = waveform_huber(x, components["slow"], delta=args.huber_delta)
            else:
                reconstructed, rate = model(x)
            distortion = waveform_huber(x, reconstructed, delta=args.huber_delta)
            loss = distortion + args.lambda_rate * rate + args.slow_loss_weight * slow_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.detach().item()
            steps += 1
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
        validation = evaluate(model, val_loader, scale, device, huber_delta=args.huber_delta)
        score = validation["huber"] + args.lambda_rate * validation["estimated_bits_per_channel_sample"]
        if args.architecture == "slow-fast":
            score += args.slow_loss_weight * validation["slow_huber"]
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation objective")
        improved_checkpoint = score < best_score
        if improved_checkpoint:
            best_score = score
            best_epoch = epoch
        if score < progress_score - args.early_stop_min_delta:
            progress_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        epochs_ran = epoch
        row = {"epoch": epoch, "train_loss": loss_total / steps, "val_objective": score,
               "improved_checkpoint": improved_checkpoint,
               "epochs_without_significant_improvement": stale_epochs,
               **{f"val_{key}": value for key, value in validation.items()}}
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        if improved_checkpoint:
            checkpoint = {
                "model": model.state_dict(), "model_config": model_config,
                "config": config, "scale": scale, "split_ids": split_ids, "epoch": epoch,
            }
            temp_path = output_dir / "best.tmp"
            torch.save(checkpoint, temp_path)
            temp_path.replace(output_dir / "best.pt")
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch}; best validation objective "
                  f"{best_score:.6f} at epoch {best_epoch}")
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "best_epoch": checkpoint["epoch"],
        "epochs_ran": epochs_ran,
        "stopped_early": stopped_early,
        "best_val_objective": best_score,
        "val": evaluate(model, val_loader, scale, device, huber_delta=args.huber_delta),
        "test": evaluate(model, test_loader, scale, device, huber_delta=args.huber_delta),
        "split_group_column": args.group_col,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Final results:", json.dumps(result))
    train_ds.close()
    val_ds.close()
    test_ds.close()


if __name__ == "__main__":
    main()
