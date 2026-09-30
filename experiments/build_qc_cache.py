"""Pre-compute the preprocessing QC cache for a whole EEG dataset.

Runs `preprocessing_pipeline` over every recording in a manifest and writes the
tables that training reads: which 8-s windows are usable, the robust scale of
each channel, and the artifact events behind those decisions. This is the slow
pass, done once; training then only reads small Parquet tables.

Work is chunked and resumable. Each chunk writes its own part files under
`parts/`, so an interrupted run continues where it stopped, and `--workers`
spreads chunks over processes. The final step concatenates the parts into the
four tables plus a summary.

    python -m experiments.build_qc_cache \
        --manifest H:/EEG/FHA/Resting/preprocessed/manifests/recordings.parquet \
        --shards-dir H:/EEG/FHA/Resting/preprocessed/shards \
        --output-dir H:/EEG/FHA/Resting/preprocessed/qc \
        --sfreq 256 --workers 4

Re-running skips finished chunks; `--restart` discards the cache and starts over.
`--finalize-only` rebuilds the tables from existing parts without reprocessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from utils.preprocessing_pipeline import (
    ACCEPT, TABLE_STEMS, QCConfig, load_bad_channels, load_manifest, process_recording,
    summarize_scan, write_qc_tables,
)


def chunk_path(parts_dir: Path, stem: str, chunk_id: int) -> Path:
    return parts_dir / stem / f"{chunk_id:05d}.parquet"


def chunk_is_done(parts_dir: Path, chunk_id: int) -> bool:
    """A chunk counts as done only when every table it owns was written."""
    return all(chunk_path(parts_dir, stem, chunk_id).exists() for stem in TABLE_STEMS)


def process_chunk(chunk_id: int, records: pd.DataFrame, shards_dir: str, output_dir: str,
                  config: QCConfig, bad_by_recording: dict) -> dict:
    """QC one chunk of recordings and write its part files. Runs in a worker."""
    parts_dir = Path(output_dir) / "parts"
    collected: dict[str, list[pd.DataFrame]] = {stem: [] for stem in TABLE_STEMS}
    handles: dict = {}
    failures = []
    started = time.time()
    try:
        for _, row in records.iterrows():
            recording_id = str(row["sha256_id"])
            try:
                tables, _ = process_recording(row, Path(shards_dir), config,
                                              bad_by_recording, handles)
            except Exception as error:  # one unreadable recording must not sink the run
                failures.append({"sha256_id": recording_id, "error": repr(error)})
                continue
            for stem, frame in tables.items():
                collected[stem].append(frame)
    finally:
        for handle in handles.values():
            handle.close()

    for stem, frames in collected.items():
        target = chunk_path(parts_dir, stem, chunk_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recordings with no events contribute an empty frame; dropping those
        # keeps concat from widening dtypes to object.
        frames = [frame for frame in frames if not frame.empty] or frames[:1]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            {"sha256_id": pd.Series(dtype="object")}
        )
        temporary = target.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)  # a part appears only once it is complete

    return {"chunk_id": chunk_id, "n_recordings": len(records), "failures": failures,
            "seconds": time.time() - started}


def finalize(output_dir: Path, config: QCConfig, manifest_path: str, shards_dir: str) -> dict:
    """Concatenate the part files into the four tables and the summary."""
    parts_dir = output_dir / "parts"
    collected: dict[str, list[pd.DataFrame]] = {}
    for stem in TABLE_STEMS:
        files = sorted((parts_dir / stem).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parts found for {stem} in {parts_dir / stem}")
        frames = [pd.read_parquet(path) for path in files]
        collected[stem] = [frame for frame in frames if not frame.empty] or [frames[0]]

    written = write_qc_tables(collected, output_dir)
    windows, events = written["window_qc"], written["events"]
    recordings = written["recording_qc"]
    report = summarize_scan(windows, events, recordings, config,
                            manifest=manifest_path, shards_dir=shards_dir)

    # The list training will actually draw from, so the cost is visible up front.
    usable = recordings[recordings["n_accepted_windows"] > 0]
    report["n_usable_recordings"] = int(len(usable))
    report["total_accepted_window_hours"] = round(
        float((windows["status"] == ACCEPT).sum() * config.window_sec / 3600), 2
    )
    failures_path = output_dir / "failures.json"
    if failures_path.exists():
        report["n_failed_recordings"] = len(json.loads(failures_path.read_text(encoding="utf-8")))
    (output_dir / "preprocessing_summary.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bad-channels", default=None,
                        help="bad_channels.parquet; defaults to the manifest's sibling file")
    parser.add_argument("--sfreq", type=float, default=None, help="keep only this sampling rate")
    parser.add_argument("--limit", type=int, default=0, help="0 processes every recording")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="recordings per resumable part file")
    parser.add_argument("--workers", type=int, default=1,
                        help="worker processes; 1 runs in this process")
    parser.add_argument("--restart", action="store_true", help="discard an existing cache")
    parser.add_argument("--finalize-only", action="store_true",
                        help="rebuild the tables from existing parts")

    defaults = QCConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)

    args = parser.parse_args()
    if args.chunk_size < 1 or args.workers < 1:
        parser.error("chunk-size and workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = QCConfig(**{key: value for key, value in vars(args).items()
                         if key in set(asdict(QCConfig()))})
    output_dir = Path(args.output_dir)
    if args.restart and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"

    if args.finalize_only:
        print(json.dumps(finalize(output_dir, config, args.manifest, args.shards_dir), indent=2))
        return

    bad_channels = args.bad_channels
    if bad_channels is None:
        candidate = Path(args.manifest).with_name("bad_channels.parquet")
        bad_channels = candidate if candidate.exists() else None
    bad_by_recording = load_bad_channels(bad_channels)
    records = load_manifest(args.manifest, args.sfreq, args.limit)

    # The config that produced a cache is part of it: mixing thresholds silently
    # would make the tables meaningless.
    config_path = output_dir / "qc_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != asdict(config) and any(parts_dir.glob("*/*.parquet")):
            raise SystemExit(
                f"{config_path} was written with different thresholds. Re-run with --restart "
                f"to rebuild the cache, or point --output-dir somewhere new."
            )
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    chunks = [(index // args.chunk_size, records.iloc[index: index + args.chunk_size])
              for index in range(0, len(records), args.chunk_size)]
    pending = [(chunk_id, frame) for chunk_id, frame in chunks
               if not chunk_is_done(parts_dir, chunk_id)]
    print(f"{len(records)} recordings, {len(chunks)} chunks, {len(pending)} to process "
          f"({len(chunks) - len(pending)} already cached), workers={args.workers}")

    failures: list[dict] = []
    done = 0
    started = time.time()

    def report(result: dict) -> None:
        nonlocal done
        done += 1
        failures.extend(result["failures"])
        elapsed = time.time() - started
        remaining = (len(pending) - done) * elapsed / max(done, 1)
        note = f" ({len(result['failures'])} failed)" if result["failures"] else ""
        print(f"[{done}/{len(pending)}] chunk {result['chunk_id']:05d} "
              f"{result['n_recordings']} recordings in {result['seconds']:.1f}s{note}"
              f" | eta {remaining / 60:.1f} min", flush=True)

    if args.workers == 1:
        for chunk_id, frame in pending:
            report(process_chunk(chunk_id, frame, args.shards_dir, str(output_dir),
                                 config, bad_by_recording))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_chunk, chunk_id, frame, args.shards_dir,
                                   str(output_dir), config, bad_by_recording)
                       for chunk_id, frame in pending]
            for future in as_completed(futures):
                report(future.result())

    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"WARNING: {len(failures)} recordings failed; see {output_dir / 'failures.json'}")

    summary = finalize(output_dir, config, args.manifest, args.shards_dir)
    print(json.dumps(summary, indent=2))
    print(f"\nQC cache ready in {output_dir} "
          f"({(time.time() - started) / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
