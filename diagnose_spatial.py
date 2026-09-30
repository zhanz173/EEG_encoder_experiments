"""Rotation-invariant spatial-subspace diagnostics; does not define a codec."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from spatial_data import SpatialDataset, loader, fingerprint, write_json
from train_spatial import load_checkpoint, amp_context, runtime


def projectors(a, rtol=1e-5):
    u, singular, _ = torch.linalg.svd(a.float(), full_matrices=False)
    keep = singular > (singular[..., :1]*rtol).clamp_min(1e-8)
    return (u*keep.unsqueeze(-2)) @ u.transpose(-1, -2), keep.sum(-1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--windows-per-recording", type=int, default=1)
    p.add_argument("--lags", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()): p.error("Use a new output directory")
    if min(args.lags) < 1 or args.rtol <= 0: p.error("Positive lags/tolerance required")
    device = torch.device(args.device)
    runtime(device, args.precision)
    model, ck = load_checkpoint(args.checkpoint, device)
    if ck["model_config"]["architecture"] != "factorized": p.error("Requires a factorized model")
    if fingerprint(Path(args.prepared)/"data.json") != ck["data_hash"]: p.error("Prepared data mismatch")
    ds = SpatialDataset(args.prepared, args.shards_dir, args.split, max_windows=args.windows_per_recording)
    rows = []
    for batch in loader(ds, args.batch_size, args.workers):
        x = batch["x"].to(device)/ck["scale"]
        with amp_context(device, args.precision): o = model(x)
        a = o["a"]
        projection, rank = projectors(a, args.rtol)
        for lag in args.lags:
            if lag >= a.shape[1]: continue
            matrix_d = (a[:, lag:]-a[:, :-lag]).square().sum((-1, -2)).mean(1)
            subspace_d = (projection[:, lag:]-projection[:, :-lag]).square().sum((-1, -2)).mean(1)
            for i in range(len(x)):
                rows.append(dict(patient_id=batch["patient_id"][i], sha256_id=batch["sha256_id"][i],
                                 start=int(batch["start"][i]), lag_frames=lag,
                                 lag_ms=1000*lag*ck["model_config"]["spatial_stride"]/ds.config["sfreq"],
                                 matrix_distance_squared=float(matrix_d[i]), projector_distance_squared=float(subspace_d[i]),
                                 mean_rank=float(rank[i].float().mean()), min_rank=int(rank[i].min())))
    if not rows: raise ValueError("No valid lag/window combinations")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"windows.csv", index=False)
    cols = ["matrix_distance_squared", "projector_distance_squared", "mean_rank", "min_rank"]
    records = frame.groupby(["patient_id", "sha256_id", "lag_frames", "lag_ms"], as_index=False)[cols].mean()
    patients = records.groupby(["patient_id", "lag_frames", "lag_ms"], as_index=False)[cols].mean()
    patients.to_csv(out/"patients.csv", index=False)
    patients.groupby(["lag_frames", "lag_ms"], as_index=False)[cols].mean().to_csv(out/"summary.csv", index=False)
    write_json(out/"config.json", dict(vars(args), checkpoint_sha256=fingerprint(args.checkpoint),
                                      interpretation="Descriptive subspace drift; not a compression result or physiological source identity"))
    ds.close()
    print(f"Saved {out/'summary.csv'}")


if __name__ == "__main__": main()
