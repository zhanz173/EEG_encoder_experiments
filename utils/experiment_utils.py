"""Data selection and evaluation shared by the local experiment scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.eeg_dataset import EEGWindowDataset
from models.model import waveform_huber


def read_manifest(path: str | Path, sfreq: float, n_channels: int, window_sec: float) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        records = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        records = pd.read_csv(path)
    else:
        raise ValueError("Manifest must be CSV or Parquet")
    required = {"sha256_id", "shard_name", "index_in_shard", "n_samples", "sfreq"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "status" in records:
        records = records[records["status"] == "ok"]
    if "n_channels" in records:
        records = records[records["n_channels"] == n_channels]
    records = records[np.isclose(records["sfreq"], sfreq)]
    records = records[records["n_samples"] >= round(window_sec * sfreq)]
    if records.empty:
        raise ValueError("No recordings match the requested channel count, sampling rate, and window")
    return records.reset_index(drop=True)


def split_records(
    records: pd.DataFrame,
    group_col: str,
    seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
) -> dict[str, pd.DataFrame]:
    if group_col not in records:
        raise ValueError(f"Group column {group_col!r} not found in manifest")
    if records[group_col].isna().any():
        raise ValueError(f"Group column {group_col!r} has missing values")
    groups = records[group_col].astype(str).unique()
    if len(groups) < 3:
        raise ValueError("At least three split groups are required")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_val = max(1, int(0.1 * len(groups)))
    n_test = max(1, int(0.1 * len(groups)))
    group_sets = {
        "val": set(groups[:n_val]),
        "test": set(groups[n_val : n_val + n_test]),
        "train": set(groups[n_val + n_test :]),
    }
    limits = {"train": max_train, "val": max_val, "test": max_test}
    result = {}
    labels = records[group_col].astype(str)
    for split, group_set in group_sets.items():
        subset = records[labels.isin(group_set)]
        limit = limits[split]
        if limit > 0 and len(subset) > limit:
            subset = subset.sample(n=limit, random_state=seed)
        result[split] = subset.reset_index(drop=True)
        if result[split].empty:
            raise ValueError(f"Empty {split} split")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: EEGWindowDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def estimate_input_scale(dataset: EEGWindowDataset, n_windows: int = 32) -> float:
    """One fixed robust scale from training windows, preserving between-window amplitude."""
    indices = np.linspace(0, len(dataset) - 1, min(n_windows, len(dataset)), dtype=int)
    samples = [dataset[int(i)]["x"].numpy() for i in indices]
    scale = float(np.percentile(np.abs(np.concatenate([x.ravel() for x in samples])), 75))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not estimate positive input scale from training data")
    return scale


def _spectral_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_fft = 256
    hop = 128
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    def power(signal: torch.Tensor) -> torch.Tensor:
        frames = signal.unfold(-1, n_fft, hop) * window
        return torch.fft.rfft(frames, dim=-1).abs().square().mean(dim=-2)
    px, py = power(x), power(y)
    return (torch.log10(px.clamp_min(1e-8)) - torch.log10(py.clamp_min(1e-8))).abs().mean()


def _spatial_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def correlation(signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return normalized @ normalized.transpose(-1, -2)
    cx, cy = correlation(x), correlation(y)
    return ((cx - cy).square().sum(dim=(-1, -2)).sqrt()
            / cx.square().sum(dim=(-1, -2)).sqrt().clamp_min(1e-8)).mean()


@torch.no_grad()
def evaluate(model, loader: DataLoader, scale: float, device: torch.device,
             max_batches: int = 0, huber_delta: float = 1.0) -> dict:
    model.eval()
    totals = {key: 0.0 for key in (
        "squared_error", "signal_energy", "huber", "estimated_bits", "channel_samples",
        "log_psd_mae", "spatial_corr_error", "line_length_rel_error", "peak_rel_error",
    )}
    n_windows = 0
    component_totals = dict(slow_huber=0.0, slow_error=0.0, slow_bits=0.0, fast_bits=0.0)
    has_components = hasattr(model, "forward_components")
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True) / scale
        if has_components:
            parts = model.forward_components(x)
            y, rate = parts["reconstruction"], parts["rate"]
            component_totals["slow_huber"] += waveform_huber(x, parts["slow"], delta=huber_delta).item() * x.shape[0]
            component_totals["slow_error"] += (x - parts["slow"]).square().sum().item()
            component_totals["slow_bits"] += parts["slow_rate"].item() * x.numel()
            component_totals["fast_bits"] += parts["fast_rate"].item() * x.numel()
        else:
            y, rate = model(x)
        count = x.shape[0]
        totals["squared_error"] += (x - y).square().sum().item()
        totals["signal_energy"] += x.square().sum().item()
        totals["huber"] += waveform_huber(x, y, delta=huber_delta).item() * count
        totals["estimated_bits"] += rate.item() * x.numel()
        totals["channel_samples"] += x.numel()
        totals["log_psd_mae"] += _spectral_error(x, y).item() * count
        totals["spatial_corr_error"] += _spatial_error(x, y).item() * count
        lx = x.diff(dim=-1).abs().mean(dim=(-1, -2))
        ly = y.diff(dim=-1).abs().mean(dim=(-1, -2))
        totals["line_length_rel_error"] += ((lx - ly).abs() / lx.clamp_min(1e-8)).sum().item()
        px = x.abs().amax(dim=(-1, -2))
        py = y.abs().amax(dim=(-1, -2))
        totals["peak_rel_error"] += ((px - py).abs() / px.clamp_min(1e-8)).sum().item()
        n_windows += count
    if n_windows == 0:
        raise ValueError("No evaluation windows")
    nmse = totals["squared_error"] / max(totals["signal_energy"], 1e-8)
    estimated_rate = totals["estimated_bits"] / totals["channel_samples"]
    result = {
        "n_windows": n_windows,
        "nmse": nmse,
        "huber": totals["huber"] / n_windows,
        "snr_db": -10 * np.log10(max(nmse, 1e-12)),
        "estimated_bits_per_channel_sample": estimated_rate,
        "estimated_bits_per_second": estimated_rate * loader.dataset.n_channels * loader.dataset.sfreq,
        "log_psd_mae": totals["log_psd_mae"] / n_windows,
        "spatial_corr_error": totals["spatial_corr_error"] / n_windows,
        "line_length_rel_error": totals["line_length_rel_error"] / n_windows,
        "peak_rel_error": totals["peak_rel_error"] / n_windows,
    }
    if has_components:
        slow_nmse = component_totals["slow_error"] / max(totals["signal_energy"], 1e-8)
        result.update(slow_huber=component_totals["slow_huber"] / n_windows,
                      slow_nmse=slow_nmse, fast_nmse_gain=slow_nmse - nmse,
                      slow_estimated_bits_per_channel_sample=component_totals["slow_bits"] / totals["channel_samples"],
                      fast_estimated_bits_per_channel_sample=component_totals["fast_bits"] / totals["channel_samples"])
    return result
