"""Phase-sensitive EEG band reconstruction metrics for continuous autoencoders."""

from __future__ import annotations

from collections import defaultdict

import torch


# Half-open intervals in Hz. Low and high beta are separate to test the
# proposed alpha/low-beta to delta transition.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "beta": (13.0, 30.0),
}


def _window_components(x: torch.Tensor, y: torch.Tensor, sfreq: float):
    """Return per-window signal, error, and output energies by frequency band.

    A Hann taper limits crop-edge leakage. FFT masks define ideal band passes;
    this is an analysis metric, not the training objective.
    """
    size = x.shape[-1]
    taper = torch.hann_window(size, device=x.device, dtype=x.dtype)
    x_fft = torch.fft.rfft(x * taper, dim=-1)
    y_fft = torch.fft.rfft(y * taper, dim=-1)
    freqs = torch.fft.rfftfreq(size, d=1.0 / sfreq).to(x.device)
    result = {}
    for name, (low, high) in BANDS.items():
        mask = ((freqs >= low) & (freqs < high)).to(x_fft.dtype)
        x_band = torch.fft.irfft(x_fft * mask, n=size, dim=-1)
        y_band = torch.fft.irfft(y_fft * mask, n=size, dim=-1)
        result[name] = (
            x_band.square().sum(dim=(-1, -2)),
            (x_band - y_band).square().sum(dim=(-1, -2)),
            y_band.square().sum(dim=(-1, -2)),
        )
    result["wave"] = (
        x.square().sum(dim=(-1, -2)),
        (x - y).square().sum(dim=(-1, -2)),
        y.square().sum(dim=(-1, -2)),
    )
    return result


def _metrics(sums: dict[str, float], n_windows: int) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"n_windows": n_windows}
    for band in ("wave", *BANDS):
        energy = max(sums[f"{band}_energy"], 1e-8)
        metrics[f"{band}_nmse"] = sums[f"{band}_error"] / energy
        metrics[f"{band}_power_ratio"] = sums[f"{band}_output_energy"] / energy
    return metrics


@torch.no_grad()
def evaluate_wave_nmse(model, loader, scale: float, device: torch.device) -> float:
    """Cheap validation readout for checkpoint selection each epoch."""
    model.eval()
    error = 0.0
    energy = 0.0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True) / scale
        y = model(x)
        error += (x - y).square().sum().item()
        energy += x.square().sum().item()
    if energy <= 0:
        raise ValueError("No usable validation signal")
    return error / energy


@torch.no_grad()
def evaluate_continuous_ae(model, loader, scale: float, device: torch.device):
    model.eval()
    total = defaultdict(float)
    by_recording = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    n_windows = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True) / scale
        y = model(x)
        components = _window_components(x, y, loader.dataset.sfreq)
        ids = batch["sha256_id"]
        band_names = tuple(components)
        values = torch.stack(
            [torch.stack(components[band], dim=-1) for band in band_names], dim=1
        ).cpu().numpy()
        for recording_id, window_values in zip(ids, values):
            counts[recording_id] += 1
            for band, (energy, error, output_energy) in zip(band_names, window_values):
                for suffix, value in (("energy", energy), ("error", error),
                                      ("output_energy", output_energy)):
                    key = f"{band}_{suffix}"
                    value = float(value)
                    total[key] += value
                    by_recording[recording_id][key] += value
        n_windows += len(ids)
    if n_windows == 0:
        raise ValueError("No evaluation windows")
    recording_rows = [
        {"sha256_id": recording_id, **_metrics(sums, counts[recording_id])}
        for recording_id, sums in by_recording.items()
    ]
    return _metrics(total, n_windows), recording_rows
