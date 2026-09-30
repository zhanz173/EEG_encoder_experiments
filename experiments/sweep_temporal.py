"""GPU job queue for joint 32-slow + 32-fast lambda sweeps.

One independent training process per GPU. Resume skips complete jobs and
restarts unfinished jobs from scratch (the trainer has no optimizer resume).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
MANAGED = {"--manifest", "--shards-dir", "--output-dir", "--architecture",
           "--channel-mode", "--slow-dim", "--fast-dim", "--latent-dim",
           "--lambda-rate", "--seed", "--device"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", default="runs/temporal_32_32_sweep")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[.01, .03, .06, .1, .2, .3, .5, .75, 1.])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--architectures", nargs="+", choices=["slow-fast", "baseline"],
                        default=["slow-fast"], help="Optionally queue a fresh single-stream control")
    parser.add_argument("--slow-dim", type=int, default=32)
    parser.add_argument("--fast-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64, help="Optional baseline width")
    parser.add_argument("--threads", type=int, default=4, help="CPU math threads per worker")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing or launching")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Training overrides, e.g. --epochs 30 --batch-size 16; place last")
    args = parser.parse_args(argv)
    for name in ("gpus", "lambdas", "seeds", "architectures"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name} are not allowed")
    if any(not math.isfinite(v) or v <= 0 for v in args.lambdas):
        parser.error("Lambdas must be finite and positive")
    if min(args.slow_dim, args.fast_dim, args.latent_dim, args.threads) < 1:
        parser.error("Dimensions and threads must be positive")
    if any(not gpu or "," in gpu or gpu.startswith("-") for gpu in args.gpus):
        parser.error("Each GPU slot must identify exactly one CUDA device")
    if any(token.split("=", 1)[0] in MANAGED for token in args.extra):
        parser.error("--extra cannot override queue-managed data, architecture, dimensions, lambda, seed, or device")
    return args


def build_plan(args):
    root = Path(args.output_dir).resolve()
    jobs = []
    for seed in args.seeds:
        for index, lam in enumerate(args.lambdas):
            for architecture in args.architectures:
                name = f"{architecture}_rate_{index:02d}_seed_{seed}"
                destination = root / name
                command = [sys.executable, "-m", "experiments.train",
                           "--manifest", str(Path(args.manifest).resolve()),
                           "--shards-dir", str(Path(args.shards_dir).resolve()),
                           "--output-dir", str(destination), "--architecture", architecture,
                           "--channel-mode", "joint", "--slow-dim", str(args.slow_dim),
                           "--fast-dim", str(args.fast_dim), "--latent-dim", str(args.latent_dim),
                           "--lambda-rate", repr(lam), "--seed", str(seed), "--device", "cuda",
                           "--epochs", "30", "--early-stop-patience", "5",
                           "--early-stop-min-delta", "0.001", "--batch-size", "16",
                           *args.extra]
                jobs.append(dict(name=name, architecture=architecture, lambda_rate=lam,
                                 seed=seed, run_dir=str(destination), command=command))
    return dict(version=1, threads=args.threads, jobs=jobs)


def gpu_environment(gpu, threads):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(threads),
                MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")


def check_gpus(gpus, threads):
    # Test the same isolated CUDA namespace each worker will use. No parent CUDA context.
    probe = ("import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1, "
             "'Requested GPU unavailable'; x=torch.zeros(1,device='cuda'); "
             "print(torch.cuda.get_device_name(0))")
    for gpu in gpus:
        result = subprocess.run([sys.executable, "-c", probe], env=gpu_environment(gpu, threads),
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"GPU {gpu} preflight failed: {result.stderr.strip()}")
        print(f"GPU {gpu}: {result.stdout.strip()}", flush=True)


def result_row(job):
    root = Path(job["run_dir"])
    if not (root / "best.pt").is_file():
        raise ValueError("Missing best.pt")
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for key in ("architecture", "lambda_rate", "seed"):
        if config[key] != job[key]:
            raise ValueError(f"Result configuration mismatch: {key}")
    for key, expected in job.get("training_config", {}).items():
        if config.get(key) != expected:
            raise ValueError(f"Result training configuration mismatch: {key}")
    for split in ("val", "test"):
        for key in ("huber", "nmse", "estimated_bits_per_channel_sample"):
            if not math.isfinite(result[split][key]):
                raise ValueError(f"Invalid {split} {key}")
    return {key: job[key] for key in ("name", "architecture", "lambda_rate", "seed", "run_dir")} | {
        key: config.get(key) for key in ("channel_mode", "latent_dim", "slow_dim", "fast_dim",
                                        "slow_stride", "fast_stride", "temporal_width", "slow_loss_weight",
                                        "distortion_loss", "huber_delta")} | {
        key: result.get(key) for key in ("best_epoch", "epochs_ran", "stopped_early", "best_val_objective")} | {
        f"{split}_{key}": value for split in ("val", "test") for key, value in result[split].items()}


def run_queue(plan, root, gpus, resume=False, poll_seconds=1):
    """Execute a saved plan; caller holds the exclusive queue lock."""
    jobs = plan["jobs"]
    statuses = {job["name"]: dict(name=job["name"], status="pending") for job in jobs}
    rows, pending, active = {}, [], {}

    def save():
        write_json(root / "queue_status.json", list(statuses.values()))
        ordered = [rows[job["name"]] for job in jobs if job["name"] in rows]
        temporary = root / "summary.csv.tmp"
        keys = list(dict.fromkeys(key for row in ordered for key in row))
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys or ["name", "status"])
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(root / "summary.csv")

    for job in jobs:
        try:
            if not resume:
                raise ValueError("Fresh job")
            rows[job["name"]] = result_row(job)
            statuses[job["name"]].update(status="complete", skipped=True)
        except (OSError, ValueError, KeyError, TypeError):
            pending.append(job)
    save()
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                status = statuses[job["name"]]
                status.update(status="running", gpu=gpu, started_at=time.time())
                log = (root / (job["name"] + ".log")).open("a", encoding="utf-8")
                try:
                    log.write(json.dumps(job["command"]) + "\n")
                    log.flush()
                    process = subprocess.Popen(job["command"], cwd=REPO,
                                               env=gpu_environment(gpu, plan["threads"]),
                                               stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (process, job, log)
                    print(f"GPU {gpu}: started {job['name']} (lambda={job['lambda_rate']})", flush=True)
                except OSError as exc:
                    log.close()
                    status.update(status="failed", error=str(exc), finished_at=time.time())
                save()
            for gpu, (process, job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                status = statuses[job["name"]]
                status.update(returncode=code, finished_at=time.time())
                try:
                    if code != 0:
                        raise ValueError(f"Training exited with code {code}")
                    rows[job["name"]] = result_row(job)
                    status["status"] = "complete"
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    status.update(status="failed", error=str(exc))
                print(f"GPU {gpu}: {status['status']} {job['name']}", flush=True)
                save()
            if active:
                time.sleep(poll_seconds)
    finally:
        # Stop children on Ctrl+C or scheduler errors; never leave GPUs running invisibly.
        for process, job, log in active.values():
            if process.poll() is None:
                process.terminate()
        for process, job, log in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            statuses[job["name"]].update(status="interrupted", finished_at=time.time())
        save()
    return not any(status["status"] != "complete" for status in statuses.values())


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    # Validate training flags before launching any expensive work; no abbreviations
    # that could override protected options via --extra are accepted by the trainer.
    from experiments.train import parse_args as parse_training_args
    for job in plan["jobs"]:
        job["training_config"] = vars(parse_training_args(job["command"][3:]))
    if args.dry_run:
        print(json.dumps(dict(plan, gpu_slots=args.gpus), indent=2))
        return
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise SystemExit("Output is not empty; use a new directory or --resume")
    if args.resume:
        if not (root / "sweep.json").is_file():
            raise SystemExit("Resume requires an existing sweep.json")
        if json.loads((root / "sweep.json").read_text(encoding="utf-8")) != plan:
            raise SystemExit("Saved sweep differs from requested configuration; use a new directory")
    check_gpus(args.gpus, args.threads)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "queue.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit("queue.lock exists: another scheduler may be running. Remove only after verifying it has stopped.")
    try:
        with handle:
            handle.write(str(os.getpid()))
        write_json(root / "sweep.json", plan)
        success = run_queue(plan, root, args.gpus, resume=args.resume)
    finally:
        lock.unlink()
    if not success:
        raise SystemExit("Some jobs failed; see queue_status.json and per-job logs. Rerun with --resume.")
    print(f"Saved {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
