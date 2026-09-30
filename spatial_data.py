"""Frozen patient splits and grid windows, preserving electrode amplitude ratios."""
from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class SpatialDataset(Dataset):
    def __init__(self, prepared, shards_dir, split, training=False, windows_per_recording=8,
                 max_windows=0, untouched_only=False):
        self.prepared = Path(prepared)
        self.config = json.loads((self.prepared / "data.json").read_text())
        for name, expected in self.config.get("files_sha256", {}).items():
            if fingerprint(self.prepared/name) != expected:
                raise ValueError(f"Prepared data was modified: {name}")
        self.records = pd.read_csv(self.prepared / "records.csv", dtype={"sha256_id": str, "patient_id": str})
        self.records = self.records[self.records.split.eq(split)].reset_index(drop=True)
        windows = pd.read_csv(self.prepared / "windows.csv", dtype={"sha256_id": str})
        windows = windows[windows.sha256_id.isin(self.records.sha256_id)]
        if untouched_only:
            windows = windows[windows.masked_samples.eq(0)]
        if max_windows:
            selected = []
            for _, group in windows.groupby("sha256_id", sort=False):
                group = group.sort_values("start_sample")
                selected.append(group.iloc[np.linspace(0, len(group)-1, min(max_windows, len(group)), dtype=int)])
            windows = pd.concat(selected, ignore_index=True)
        starts = windows.groupby("sha256_id").start_sample.apply(list).to_dict()
        self.records = self.records[self.records.sha256_id.isin(starts)].reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"No usable {split} recordings")
        self.rows = self.records.to_dict("records")
        self.starts = [starts[r["sha256_id"]] for r in self.rows]
        self.index = [(i, s) for i, ss in enumerate(self.starts) for s in ss]
        self.events = {}
        events_path = self.prepared / "masked_events.csv"
        if events_path.exists():
            events = pd.read_csv(events_path, dtype={"sha256_id": str})
            self.events = {key: g[["channel_index", "start_sample", "stop_sample"]].to_numpy(int)
                           for key, g in events.groupby("sha256_id")}
        self.training, self.windows_per_recording = training, windows_per_recording
        self.crop_seed = 0
        self.shards_dir = Path(shards_dir)
        self._files = OrderedDict()

    def __len__(self):
        return len(self.rows)*self.windows_per_recording if self.training else len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        return state

    def __getitem__(self, idx):
        epoch = None
        if isinstance(idx, tuple):
            idx, epoch = idx
        if self.training:
            row_idx = idx // self.windows_per_recording
            starts = self.starts[row_idx]
            if epoch is None:
                pick = int(torch.randint(len(starts), (1,)).item())
            else:
                # Epoch travels with each index into persistent workers.
                rng = np.random.default_rng(np.random.SeedSequence([self.crop_seed, epoch, idx]))
                pick = int(rng.integers(len(starts)))
            start = starts[pick]
        else:
            row_idx, start = self.index[idx]
        row = self.rows[row_idx]
        shard = row["shard_name"]
        if shard not in self._files:
            # Bound handles and chunk caches per worker: important for two jobs.
            if len(self._files) >= 8:
                self._files.popitem(last=False)[1].close()
            self._files[shard] = h5py.File(self.shards_dir / shard, "r", rdcc_nbytes=1024*1024)
        self._files.move_to_end(shard)
        length = self.config["window_samples"]
        x = np.array(self._files[shard]["signals"][int(row["index_in_shard"]), :, start:start+length], dtype=np.float32)
        if x.shape != (self.config["n_channels"], length) or not np.isfinite(x).all():
            raise ValueError(f"Invalid signal in recording {row['sha256_id']} at {start}")
        mask = np.zeros_like(x, dtype=bool)
        pad = self.config.get("mask_pad_samples", 0)
        for channel, lo, hi in self.events.get(row["sha256_id"], []):
            a, b = max(lo-pad, start)-start, min(hi+pad, start+length)-start
            if a < b:
                mask[channel, a:b] = True
        # Interpolate in original signal units; never per-channel normalize.
        for c in np.flatnonzero(mask.any(axis=1)):
            good = np.flatnonzero(~mask[c])
            if not len(good):
                raise ValueError("A whole channel is masked; reject this window in QC")
            bad = np.flatnonzero(mask[c])
            x[c, bad] = np.interp(bad, good, x[c, good])
        return dict(x=torch.from_numpy(x), mask=torch.from_numpy(mask),
                    sha256_id=row["sha256_id"], patient_id=row["patient_id"], start=start)

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch*100003)
        for i in torch.randperm(len(self.dataset), generator=generator).tolist():
            yield (i, self.epoch)

    def __len__(self):
        return len(self.dataset)


def loader(ds, batch_size=64, workers=4, shuffle=False, seed=42):
    ds.crop_seed = seed
    kwargs = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed,
                  generator=torch.Generator().manual_seed(seed))
    if shuffle:
        kwargs["sampler"] = EpochSampler(ds, seed)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(ds, **kwargs)
