"""Check the actual CUDA kernels and measure one model on the target workstation."""
import argparse
import json
import time
import torch
from spatial_model import make_model, distortion
from train_spatial import amp_context, runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--architecture", choices=["factorized", "baseline"], default="factorized")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    if min(args.batch_size, args.steps) < 1:
        p.error("Positive batch size and steps required")
    device = torch.device(args.device)
    runtime(device, args.precision)
    model = make_model(dict(architecture=args.architecture)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    x = torch.randn(args.batch_size, 20, 2048, device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps+3):
        if step == 3:
            if device.type == "cuda": torch.cuda.synchronize(device)
            began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.precision):
            result = model(x)
        loss = (distortion(x, result["y"], mask)+.05*result["rate"]).mean()
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite benchmark loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer); scaler.update()
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-began
    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                          device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                          precision=args.precision, batch_size=args.batch_size,
                          windows_per_second=args.batch_size*args.steps/elapsed,
                          peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else None,
                          note="Synthetic in-memory training throughput; excludes HDF5 loading and evaluation"), indent=2))


if __name__ == "__main__": main()
