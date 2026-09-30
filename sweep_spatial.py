"""Two-GPU experiment queue: one isolated training/post-hoc job per GPU.

Works on native Windows and Linux. No NCCL, DataParallel, or cross-GPU model copy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from spatial_data import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", nargs="+", default=["0", "1"])
    p.add_argument("--lambdas", nargs="+", type=float, default=[.02, .1])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--ranks", nargs="+", type=int, default=[8])
    p.add_argument("--architectures", nargs="+", choices=["baseline", "factorized"], default=["baseline", "factorized"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--posthoc", action="store_true", help="Fit train dictionaries, then evaluate validation only")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        p.error("GPU IDs must be distinct")
    for name in ("ranks", "lambdas", "seeds", "architectures"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f"Duplicate {name} would overwrite jobs")
    if min(args.lambdas) <= 0:
        p.error("Positive lambdas required")
    root = Path(__file__).resolve().parent
    out = Path(args.output_dir).resolve()
    jobs = []
    for arch in args.architectures:
        for rank, lam, seed in itertools.product(args.ranks if arch == "factorized" else [8], args.lambdas, args.seeds):
            name = f"{arch}_k{rank}_lambda{lam:g}_seed{seed}"
            dest = out/name
            cmd = [sys.executable, str(root/"train_spatial.py"), "--prepared", str(Path(args.prepared).resolve()),
                   "--shards-dir", str(Path(args.shards_dir).resolve()), "--output-dir", str(dest),
                   "--architecture", arch, "--rank", str(rank), "--lambda-rate", str(lam), "--seed", str(seed),
                   "--device", "cuda:0", "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                   "--threads", str(args.threads), "--epochs", str(args.epochs), "--precision", args.precision]
            jobs.append(dict(name=name, command=cmd))
    plan = dict(jobs=jobs, gpu_slots=args.gpus, posthoc=args.posthoc)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if out.exists() and any(out.iterdir()) and not args.resume:
        p.error("Output exists; use --resume or a new directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"sweep.json").exists():
        old = json.loads((out/"sweep.json").read_text())
        if old["jobs"] != jobs or old["posthoc"] != args.posthoc:
            p.error("Resume configuration differs from saved sweep; use a new output directory")
    write_json(out/"sweep.json", plan)
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    results = []
    def worker(gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1")
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            dest = out/job["name"]
            log = out/(job["name"]+".log")
            status = dict(name=job["name"], gpu=gpu, status="running")
            try:
                commands = []
                if not (dest/"results.json").exists():
                    cmd = list(job["command"])
                    if args.resume and (dest/"last.pt").exists():
                        cmd.append("--resume")
                    commands.append(cmd)
                if args.posthoc:
                    base = [sys.executable, str(root/"posthoc_spatial.py")]
                    common = ["--prepared", str(Path(args.prepared).resolve()), "--shards-dir", str(Path(args.shards_dir).resolve()),
                              "--device", "cuda:0", "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--threads", str(args.threads)]
                    if not (dest/"dictionary/dictionary.pt").exists():
                        commands.append(base+["fit", "--checkpoint", str(dest/"best.pt"), "--output-dir", str(dest/"dictionary"),
                                              "--precision", args.precision]+common)
                    if not (dest/"posthoc_val/comparison.csv").exists():
                        commands.append(base+["evaluate", "--bundle", str(dest/"dictionary/dictionary.pt"),
                                              "--output-dir", str(dest/"posthoc_val"), "--split", "val"]+common)
                with log.open("a", encoding="utf-8") as f:
                    for command in commands:
                        print(f"GPU {gpu}: {job['name']} [{Path(command[1]).name}]", flush=True)
                        f.write(json.dumps(command)+"\n"); f.flush()
                        subprocess.run(command, env=env, cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True)
                status["status"] = "complete"
            except Exception as exc:
                status.update(status="failed", error=str(exc))
                print(f"FAILED {job['name']}; see {log}", flush=True)
            with lock:
                results.append(status)
                write_json(out/"queue_status.json", results)
            pending.task_done()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in as_completed(futures):
            future.result()
    if any(r["status"] == "failed" for r in results):
        raise SystemExit("Some jobs failed; successful jobs were preserved. Inspect queue_status.json and logs.")


if __name__ == "__main__":
    main()
