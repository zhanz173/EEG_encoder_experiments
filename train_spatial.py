"""Train one spatial-factorization or single-stream model on one GPU."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from spatial_data import SpatialDataset, loader, write_json, fingerprint
from spatial_model import make_model, distortion
from spatial_metrics import Metrics


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def runtime(device, precision, threads=4):
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on selected GPU; choose fp16 or fp32")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_checkpoint(path, device):
    # Only load trusted locally produced checkpoints (optimizer/RNG state included).
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def evaluate(model, dl, scale, device, precision, loss, output=None, stride=16, quick=False):
    model.eval()
    metric = None if quick else Metrics(dl.dataset.config["sfreq"], stride,
                                        Path(output)/"windows.csv" if output else None)
    rows = []
    for batch in dl:
        x = batch["x"].to(device, non_blocking=True)/scale
        mask = batch["mask"].to(device, non_blocking=True)
        with amp_context(device, precision):
            o = model(x)
        d = distortion(x, o["y"], mask, loss)
        vals = torch.stack((d, o["rate"], o["bits_spatial"]/x[0].numel(), o["bits_temporal"]/x[0].numel()), 1).cpu().numpy()
        rows.extend(dict(patient_id=p, sha256_id=r, distortion=float(v[0]), rate=float(v[1]),
                         rate_spatial=float(v[2]), rate_temporal=float(v[3]))
                    for p, r, v in zip(batch["patient_id"], batch["sha256_id"], vals))
        if metric:
            metric.add(batch, x, o["y"], o["bits_spatial"], o["bits_temporal"])
    frame = pd.DataFrame(rows)
    means = frame.groupby(["patient_id", "sha256_id"])[["distortion", "rate", "rate_spatial", "rate_temporal"]].mean().groupby("patient_id").mean().mean()
    result = {k: float(v) for k, v in means.items()}
    if metric:
        result.update(metric.finish(output))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--architecture", choices=["factorized", "baseline"], default="factorized")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--temporal-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--spatial-stride", type=int, default=16)
    p.add_argument("--lambda-rate", type=float, default=.05)
    p.add_argument("--loss", choices=["nmse", "huber"], default="nmse")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64, help="Per GPU/job")
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--train-windows", type=int, default=8)
    p.add_argument("--monitor-windows", type=int, default=4, help="Evenly spaced windows per recording; 0=all")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=.0005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true", help="Optional torch.compile; benchmark on target machine")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate-test", action="store_true", help="Opt in after validation selection")
    p.add_argument("--max-steps", type=int, default=0, help="Smoke/debug only; 0=all")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulate, args.train_windows, args.threads, args.spatial_stride) < 1 or args.lambda_rate <= 0:
        p.error("Positive epochs/batch/accumulation/windows/threads/lambda required")
    if min(args.workers, args.monitor_windows, args.patience, args.max_steps) < 0 or not math.isfinite(args.lambda_rate):
        p.error("Invalid nonnegative setting or nonfinite lambda")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        p.error("Output exists; choose a new directory or --resume")
    if args.resume and not (out/"last.pt").exists():
        p.error("--resume requires last.pt")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runtime(device, args.precision, args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = json.loads((Path(args.prepared)/"data.json").read_text())
    for name, expected in data["files_sha256"].items():
        if fingerprint(Path(args.prepared)/name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    data_hash = fingerprint(Path(args.prepared)/"data.json")
    model_config = dict(architecture=args.architecture, rank=args.rank, n_channels=data["n_channels"],
                        spatial_dim=args.spatial_dim, temporal_dim=args.temporal_dim,
                        latent_dim=args.latent_dim, spatial_stride=args.spatial_stride)
    if data["window_samples"] % args.spatial_stride:
        p.error("Spatial stride must divide crop length")
    model = make_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    best, progress, stale, start = math.inf, math.inf, 0, 0
    if args.resume:
        ck = torch.load(out/"last.pt", map_location="cpu", weights_only=False)
        if ck["model_config"] != model_config or ck["data_hash"] != data_hash:
            raise ValueError("Resume model/data mismatch")
        for key in ("lambda_rate", "loss", "seed", "batch_size", "accumulate", "train_windows", "monitor_windows", "lr", "precision"):
            if ck["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume setting changed: {key}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        best, progress, stale, start = ck["best"], ck["progress"], ck["stale"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"], device)
    run_model = torch.compile(model) if args.compile else model
    config = dict(vars(args), model_config=model_config, data_hash=data_hash, scale=data["scale"],
                  parameters=sum(v.numel() for v in model.parameters()), torch_version=torch.__version__,
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  rate_units="estimated bits/channel-sample; headers excluded")
    write_json(out/"config.json", config)
    train_ds = SpatialDataset(args.prepared, args.shards_dir, "train", True, args.train_windows)
    val_ds = SpatialDataset(args.prepared, args.shards_dir, "val", max_windows=args.monitor_windows)
    train_dl = loader(train_ds, args.batch_size, args.workers, True, args.seed)
    val_dl = loader(val_ds, args.batch_size, args.workers, False, args.seed)
    for epoch in range(start+1, args.epochs+1):
        # Reproducible epoch-indexed crops, including persistent-worker resume.
        train_dl.sampler.set_epoch(epoch)
        train_dl.generator.manual_seed(args.seed+epoch*100003)
        torch.manual_seed(args.seed+epoch*100003)
        run_model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count, begun = 0., 0, time.monotonic()
        steps = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
        for step, batch in enumerate(train_dl):
            x = batch["x"].to(device, non_blocking=True)/data["scale"]
            mask = batch["mask"].to(device, non_blocking=True)
            with amp_context(device, args.precision):
                result = run_model(x)
            loss = (distortion(x, result["y"], mask, args.loss) + args.lambda_rate*result["rate"]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            group_size = min(args.accumulate, steps-(step//args.accumulate)*args.accumulate)
            group_start = (step//args.accumulate)*args.accumulate
            total_samples = min(len(train_ds), steps*args.batch_size)
            group_samples = min(group_size*args.batch_size, total_samples-group_start*args.batch_size)
            scaler.scale(loss*len(x)/group_samples).backward()
            if (step+1) % args.accumulate == 0 or step+1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += loss.detach().item()*len(x); count += len(x)
            if step+1 >= steps:
                break
        val = evaluate(run_model, val_dl, data["scale"], device, args.precision, args.loss, quick=True)
        score = val["distortion"] + args.lambda_rate*val["rate"]
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation objective")
        improved = score < best
        best = min(best, score)
        if score < progress-args.min_delta:
            progress, stale = score, 0
        else:
            stale += 1
        state = dict(model=model.state_dict(), model_config=model_config, args=vars(args),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                     best=best, progress=progress, stale=stale, scale=data["scale"], data_hash=data_hash,
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            tmp = out/(name+".tmp")
            torch.save(state, tmp); tmp.replace(out/name)
        row = dict(epoch=epoch, train_objective=total/count, val_objective=score, **val,
                   seconds=time.monotonic()-begun, improved=improved)
        with (out/"history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        if args.patience and stale >= args.patience:
            break
    model, ck = load_checkpoint(out/"best.pt", device)
    result = dict(best_epoch=ck["epoch"], data_hash=data_hash)
    for split in (["val", "test"] if args.evaluate_test else ["val"]):
        ds = SpatialDataset(args.prepared, args.shards_dir, split)
        result[split] = evaluate(model, loader(ds, args.batch_size, args.workers), data["scale"],
                                 device, args.precision, args.loss, out/f"evaluation_{split}", args.spatial_stride)
        ds.close()
    write_json(out/"results.json", result)
    print(json.dumps({"best_epoch": ck["epoch"], "results": str(out/"results.json"),
                      "validation_nmse": result["val"]["wave_nmse"],
                      "validation_rate": result["val"]["rate_total"]}), flush=True)
    train_ds.close(); val_ds.close()


if __name__ == "__main__":
    main()
