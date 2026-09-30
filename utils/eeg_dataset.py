from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ContinuousShardedEEGDataset(Dataset):
    """
    Dataset over continuous EEG recordings stored in HDF5 shards.

    Each item can return:
    - a random or fixed crop from a full EEG
    - the full EEG if window_sec is None
    - optional annotations and bad channels by sha256_id

    Manifest columns expected:
    - sha256_id
    - shard_name
    - index_in_shard
    - n_samples
    - sfreq
    """

    def __init__(
        self,
        recordings_path: Union[str, Path],
        shards_dir: Union[str, Path],
        annotations_path: Optional[Union[str, Path]] = None,
        bad_channels_path: Optional[Union[str, Path]] = None,
        query: Optional[str] = None,
        window_sec: Optional[float] = None,
        stride_sec: Optional[float] = None,
        random_crop: bool = True,
        pad_if_short: bool = True,
        return_annotations: bool = False,
        return_bad_channels: bool = False,
    ) -> None:
        self.recordings_path = Path(recordings_path)
        self.shards_dir = Path(shards_dir)

        self.df = self._read_table(self.recordings_path)
        self.df = self.df[self.df["status"] == "ok"].copy()

        if query is not None:
            self.df = self.df.query(query).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.annotations_df = self._read_table(Path(annotations_path)) if annotations_path is not None else None
        self.bad_channels_df = self._read_table(Path(bad_channels_path)) if bad_channels_path is not None else None

        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.random_crop = random_crop
        self.pad_if_short = pad_if_short
        self.return_annotations = return_annotations
        self.return_bad_channels = return_bad_channels

        self._files: Dict[str, h5py.File] = {}

        self.sha256_to_index: Dict[str, int] = {
            row["sha256_id"]: i for i, row in self.df.iterrows()
        }

        # Optional window indexing for deterministic sliding windows
        self.window_index = None
        if self.window_sec is not None and self.random_crop is False and self.stride_sec is not None:
            self.window_index = self._build_window_index()

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {path}")

    def _get_h5(self, shard_name: str) -> h5py.File:
        if shard_name not in self._files:
            self._files[shard_name] = h5py.File(self.shards_dir / shard_name, "r")
        return self._files[shard_name]

    def _build_window_index(self) -> List[Dict]:
        out = []
        for row_idx, row in self.df.iterrows():
            sfreq = float(row["sfreq"])
            n_samples = int(row["n_samples"])
            win = int(round(self.window_sec * sfreq))
            stride = int(round(self.stride_sec * sfreq))
            if win <= 0 or stride <= 0:
                raise ValueError("window_sec and stride_sec must produce positive sample counts")
            if n_samples < win:
                if self.pad_if_short:
                    out.append({"row_idx": row_idx, "start": 0, "stop": n_samples})
                continue
            for start in range(0, n_samples - win + 1, stride):
                out.append({"row_idx": row_idx, "start": start, "stop": start + win})
        return out

    def __len__(self) -> int:
        if self.window_index is not None:
            return len(self.window_index)
        return len(self.df)

    def get_index_by_sha256(self, sha256_id: str) -> Optional[int]:
        return self.sha256_to_index.get(sha256_id)

    def get_annotations_by_sha256(self, sha256_id: str) -> pd.DataFrame:
        if self.annotations_df is None:
            raise ValueError("annotations_path was not provided")
        return self.annotations_df[self.annotations_df["sha256_id"] == sha256_id].reset_index(drop=True)

    def get_bad_channels_by_sha256(self, sha256_id: str) -> List[str]:
        if self.bad_channels_df is None:
            raise ValueError("bad_channels_path was not provided")
        sub = self.bad_channels_df[self.bad_channels_df["sha256_id"] == sha256_id]
        if len(sub) == 0:
            return []
        return sub["channel_name"].tolist()

    def get_full_recording_by_sha256(self, sha256_id: str) -> Dict:
        idx = self.get_index_by_sha256(sha256_id)
        if idx is None:
            raise KeyError(f"SHA256 ID not found: {sha256_id}")
        return self._get_recording_item(idx, crop_override=None)

    def _load_full_signal(self, row: pd.Series) -> np.ndarray:
        shard_name = row["shard_name"]
        index_in_shard = int(row["index_in_shard"])
        n_samples = int(row["n_samples"])

        h5f = self._get_h5(shard_name)
        x = h5f["signals"][index_in_shard, :, :n_samples]  # (C, T)
        return np.asarray(x, dtype=np.float32)

    def _crop_signal(self, x: np.ndarray, sfreq: float, crop_override=None):
        total_len = x.shape[1]

        if crop_override is not None:
            start, stop = crop_override
            return x[:, start:stop], start, stop

        if self.window_sec is None:
            return x, 0, total_len

        win_len = int(round(self.window_sec * sfreq))
        if win_len <= 0:
            raise ValueError("window_sec must produce positive sample count")

        if total_len >= win_len:
            if self.random_crop:
                start = np.random.randint(0, total_len - win_len + 1)
                stop = start + win_len
                return x[:, start:stop], start, stop
            else:
                start = 0
                stop = win_len
                return x[:, start:stop], start, stop

        # short recording
        if not self.pad_if_short:
            return x, 0, total_len

        padded = np.zeros((x.shape[0], win_len), dtype=np.float32)
        padded[:, :total_len] = x
        return padded, 0, total_len

    def _make_item(self, row: pd.Series, x_crop: np.ndarray, start: int, stop: int) -> Dict:
        sfreq = float(row["sfreq"])
        item = {
            "x": torch.tensor(x_crop, dtype=torch.float32),
            "sha256_id": row["sha256_id"],
            "site": row["site"],
            "sfreq": sfreq,
            "crop_start_sample": int(start),
            "crop_stop_sample": int(stop),
            "crop_start_sec": float(start / sfreq),
            "crop_stop_sec": float(stop / sfreq),
            "n_samples_total": int(row["n_samples"]),
        }

        if self.return_annotations:
            if self.annotations_df is None:
                raise ValueError("return_annotations=True but annotations_path was not provided")
            anns = self.annotations_df[self.annotations_df["sha256_id"] == row["sha256_id"]]
            item["annotations"] = anns.to_dict(orient="records")

        if self.return_bad_channels:
            if self.bad_channels_df is None:
                raise ValueError("return_bad_channels=True but bad_channels_path was not provided")
            bads = self.bad_channels_df[self.bad_channels_df["sha256_id"] == row["sha256_id"]]
            item["bad_channels"] = bads["channel_name"].tolist()

        return item

    def _get_recording_item(self, idx: int, crop_override=None) -> Dict:
        row = self.df.iloc[idx]
        x = self._load_full_signal(row)
        x_crop, start, stop = self._crop_signal(x, float(row["sfreq"]), crop_override=crop_override)
        return self._make_item(row, x_crop, start, stop)

    def __getitem__(self, idx: int) -> Dict:
        if self.window_index is not None:
            spec = self.window_index[idx]
            row = self.df.iloc[spec["row_idx"]]
            x = self._load_full_signal(row)
            x_crop, start, stop = self._crop_signal(
                x, float(row["sfreq"]), crop_override=(spec["start"], spec["stop"])
            )
            return self._make_item(row, x_crop, start, stop)

        return self._get_recording_item(idx)

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EEGWindowDataset(Dataset):
    """Read short windows directly from HDF5 without loading a full recording.

    ``records`` is a prefiltered, pre-split manifest. Training draws a new random
    position each access; evaluation uses a fixed set of non-overlapping windows.
    HDF5 handles are opened lazily in each worker process.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        shards_dir: Union[str, Path],
        window_sec: float = 8.0,
        windows_per_recording: int = 4,
        training: bool = True,
        n_channels: int = 20,
        sfreq: float = 256.0,
        eval_start_sec: float = 0.0,
    ) -> None:
        if windows_per_recording < 1:
            raise ValueError("windows_per_recording must be positive")
        if eval_start_sec < 0:
            raise ValueError("eval_start_sec must be nonnegative")
        required = {"sha256_id", "shard_name", "index_in_shard", "n_samples", "sfreq"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.window_samples = int(round(window_sec * sfreq))
        if self.window_samples < 1:
            raise ValueError("window_sec must produce at least one sample")
        self.records = records.reset_index(drop=True).copy()
        if self.records.empty:
            raise ValueError("No recordings selected")
        if not np.all(np.isclose(self.records["sfreq"].to_numpy(dtype=float), sfreq)):
            raise ValueError("All selected recordings must have the requested sampling rate")
        if (self.records["n_samples"] < self.window_samples).any():
            raise ValueError("All selected recordings must contain a full window")
        if "n_channels" in self.records and (self.records["n_channels"] != n_channels).any():
            raise ValueError("All selected recordings must have the requested channel count")
        self.shards_dir = Path(shards_dir)
        self.windows_per_recording = windows_per_recording
        self.training = training
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.eval_start_sec = eval_start_sec
        self._files: Dict[str, h5py.File] = {}
        self._pid = os.getpid()

        self.eval_index = []
        if not training:
            first_window = int(np.ceil(eval_start_sec * sfreq / self.window_samples))
            for row_idx, row in self.records.iterrows():
                n_whole = int(row["n_samples"]) // self.window_samples
                if n_whole <= first_window:
                    raise ValueError(
                        f"Recording {row['sha256_id']} has no full window after eval_start_sec"
                    )
                count = min(windows_per_recording, n_whole - first_window)
                positions = np.linspace(first_window, n_whole - 1, count, dtype=int)
                self.eval_index.extend(
                    (row_idx, int(pos) * self.window_samples) for pos in positions
                )

    def __len__(self) -> int:
        if self.training:
            return len(self.records) * self.windows_per_recording
        return len(self.eval_index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_pid"] = None
        return state

    def _get_h5(self, shard_name: str) -> h5py.File:
        if self._pid != os.getpid():
            self._files = {}
            self._pid = os.getpid()
        if shard_name not in self._files:
            self._files[shard_name] = h5py.File(self.shards_dir / shard_name, "r")
        return self._files[shard_name]

    def __getitem__(self, idx: int) -> Dict:
        if self.training:
            row_idx = idx // self.windows_per_recording
            row = self.records.iloc[row_idx]
            last_start = int(row["n_samples"]) - self.window_samples
            start = int(torch.randint(last_start + 1, (1,)).item())
        else:
            row_idx, start = self.eval_index[idx]
            row = self.records.iloc[row_idx]
        shard_name = str(row["shard_name"])
        x = self._get_h5(shard_name)["signals"][
            int(row["index_in_shard"]), :, start : start + self.window_samples
        ]
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.n_channels, self.window_samples):
            raise ValueError(f"Unexpected EEG shape {x.shape} in {shard_name}")
        return {
            "x": torch.from_numpy(x),
            "sha256_id": str(row["sha256_id"]),
            "crop_start_sample": start,
        }

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CleanEEGWindowDataset(EEGWindowDataset):
    """EEGWindowDataset restricted to the windows a QC cache accepted.

    Windows come from the fixed grid the QC scored (0-8 s, 8-16 s, ...), never
    from an arbitrary offset, because acceptance was decided per grid window.
    Each item is robust-normalized with its own recording's per-channel median
    and MAD, and carries a `mask` marking samples that were interpolated over a
    masked artifact, so the reconstruction loss can ignore them:

        loss = ((x - y).square() * ~mask).sum() / (~mask).sum()

    Build the cache first with `build_qc_cache.py`.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        shards_dir: Union[str, Path],
        qc_dir: Union[str, Path],
        window_sec: float = 8.0,
        windows_per_recording: int = 4,
        training: bool = True,
        n_channels: int = 20,
        sfreq: float = 256.0,
        eval_start_sec: float = 0.0,
        drop_suspect_recordings: bool = False,
    ) -> None:
        from utils.preprocessing_pipeline import QCConfig, WindowPreprocessor

        qc_dir = Path(qc_dir)
        config_path = qc_dir / "qc_config.json"
        config = QCConfig(**json.loads(config_path.read_text(encoding="utf-8"))) \
            if config_path.exists() else QCConfig()
        if abs(config.window_sec - window_sec) > 1e-9:
            raise ValueError(
                f"QC cache was built with window_sec={config.window_sec}, not {window_sec}; "
                f"rebuild the cache to change the window length"
            )
        self.qc = WindowPreprocessor(qc_dir, config)

        records = records[records["sha256_id"].astype(str).map(
            lambda key: len(self.qc.accepted_starts(key)) > 0
        )].reset_index(drop=True)
        if records.empty:
            raise ValueError("No selected recording has an accepted window in the QC cache")
        if drop_suspect_recordings:
            recording_qc = pd.read_parquet(qc_dir / "recording_qc.parquet")
            suspect = set(recording_qc.loc[recording_qc["suspect"], "sha256_id"].astype(str))
            records = records[~records["sha256_id"].astype(str).isin(suspect)].reset_index(drop=True)
            if records.empty:
                raise ValueError("Every selected recording is marked suspect in the QC cache")

        # Built as training=True so the parent skips its own grid-wide evaluation
        # index; this class selects evaluation windows from accepted ones below.
        super().__init__(records, shards_dir, window_sec, windows_per_recording, True,
                         n_channels, sfreq, eval_start_sec)
        self.training = training

        # Accepted starts per recording, in this dataset's row order.
        first_start = int(np.ceil(eval_start_sec * sfreq / self.window_samples)) * self.window_samples
        self.starts: List[np.ndarray] = []
        for _, row in self.records.iterrows():
            starts = self.qc.accepted_starts(str(row["sha256_id"]))
            starts = starts[starts + self.window_samples <= int(row["n_samples"])]
            if not training:
                starts = starts[starts >= first_start]
            self.starts.append(starts)

        keep = [index for index, starts in enumerate(self.starts) if len(starts)]
        if len(keep) < len(self.records):
            self.records = self.records.iloc[keep].reset_index(drop=True)
            self.starts = [self.starts[index] for index in keep]
            if self.records.empty:
                raise ValueError("No recording retains an accepted window after eval_start_sec")

        # Evaluation covers each recording evenly with a fixed, reproducible set.
        self.eval_index = []
        if not training:
            for row_idx, starts in enumerate(self.starts):
                count = min(windows_per_recording, len(starts))
                chosen = np.linspace(0, len(starts) - 1, count).astype(int)
                self.eval_index.extend((row_idx, int(starts[position])) for position in chosen)

    def __getitem__(self, idx: int) -> Dict:
        if self.training:
            row_idx = idx // self.windows_per_recording
            starts = self.starts[row_idx]
            start = int(starts[int(torch.randint(len(starts), (1,)).item())])
        else:
            row_idx, start = self.eval_index[idx]
        row = self.records.iloc[row_idx]
        shard_name = str(row["shard_name"])
        raw = self._get_h5(shard_name)["signals"][
            int(row["index_in_shard"]), :, start : start + self.window_samples
        ]
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.n_channels, self.window_samples):
            raise ValueError(f"Unexpected EEG shape {raw.shape} in {shard_name}")
        recording_id = str(row["sha256_id"])
        x, mask = self.qc.apply(raw, recording_id, start, self.sfreq)
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "sha256_id": recording_id,
            "crop_start_sample": start,
        }


def test():
    # Example usage of the ContinuousShardedEEGDataset
    dataset = ContinuousShardedEEGDataset(
    recordings_path=r"H:\EEG\FHA\Resting\preprocessed\manifests\recordings.parquet",
    shards_dir=r"H:\EEG\FHA\Resting\preprocessed\shards",
    annotations_path=r"H:\EEG\FHA\Resting\preprocessed\manifests\annotations.parquet",
    bad_channels_path=r"H:\EEG\FHA\Resting\preprocessed\manifests\bad_channels.parquet",
    return_annotations=True,
    return_bad_channels=False,
)

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample shape: {sample['x'].shape}")

    # check EEG neumrical range 
    p5 = np.percentile(sample['x'].numpy(), 5)
    p95 = np.percentile(sample['x'].numpy(), 95)
    print(f"5th percentile: {p5}, 95th percentile: {p95}")
    dataset.close()

if __name__ == "__main__":
    # keys = ['x', 'sha256_id', 'site', 'sfreq', 'crop_start_sample', 'crop_stop_sample', 'crop_start_sec', 'crop_stop_sec', 'n_samples_total', 'annotations']
    # EGG shape: (C, T), C = channels, T = time
    test()
