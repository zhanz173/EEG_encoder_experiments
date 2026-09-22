"""Re-evaluate a saved EEG model on its fixed validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eeg_dataset import EEGWindowDataset
from experiment_utils import evaluate, make_loader, read_manifest
from model import EEGRateDistortionAE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--manifest", help="Override manifest path saved in checkpoint")
    parser.add_argument("--shards-dir", help="Override shard directory saved in checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    manifest = args.manifest or config["manifest"]
    shards_dir = args.shards_dir or config["shards_dir"]
    records = read_manifest(manifest, config["sfreq"], config["channels"], config["window_sec"])
    ids = checkpoint["split_ids"][args.split]
    selected = records.set_index("sha256_id").loc[ids].reset_index()
    dataset = EEGWindowDataset(selected, shards_dir, config["window_sec"],
                               config["eval_windows_per_recording"], False,
                               config["channels"], config["sfreq"],
                               eval_start_sec=config.get("eval_start_sec", 0.0))
    loader = make_loader(dataset, args.batch_size, args.num_workers, False)
    model = EEGRateDistortionAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    result = {"split": args.split, "checkpoint": str(args.checkpoint),
              "split_group_column": config["group_col"],
              **evaluate(model, loader, checkpoint["scale"], device,
                         huber_delta=config.get("huber_delta", 1.0))}
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    dataset.close()


if __name__ == "__main__":
    main()
