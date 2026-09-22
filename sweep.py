"""Train several rate penalties with identical data splits and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", default="runs/sweep")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.001, 0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Remaining arguments passed to train.py; place this last")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, rate_weight in enumerate(args.lambdas):
        run_dir = root / f"rate_{index:02d}"
        command = [sys.executable, str(Path(__file__).with_name("train.py")),
                   "--manifest", args.manifest, "--shards-dir", args.shards_dir,
                   "--output-dir", str(run_dir), "--lambda-rate", str(rate_weight)]
        if args.extra:
            command += args.extra
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        rows.append({"lambda_rate": rate_weight, "run_dir": str(run_dir),
                     "distortion_loss": config["distortion_loss"],
                     "huber_delta": config["huber_delta"],
                     "best_epoch": result["best_epoch"],
                     "epochs_ran": result.get("epochs_ran"),
                     "stopped_early": result.get("stopped_early"),
                     "best_val_objective": result.get("best_val_objective"),
                     **{f"test_{key}": value for key, value in result["test"].items()}})
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
