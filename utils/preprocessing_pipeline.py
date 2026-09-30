"""Artifact-aware preprocessing for the EEG rate-distortion experiments.

The pipeline separates *measurement* artifacts from *signal* that merely looks
extreme, so that the auto-encoder is trained on clean windows without having
its amplitude statistics distorted by clipping.

Stages
------
1. Robust per-channel statistics per recording (median / MAD, two-pass with a
   trimmed second pass) computed on the raw signal, excluding the setup lead-in.
2. Detection on the robust z-score of the *raw* signal, before normalization:
     - isolated, near-instantaneous, enormous transient      -> mask samples
     - simultaneous broadband step across many channels      -> reject window
     - spatially structured sharp transient (tens-hundreds ms) -> preserve, flag
     - sustained high amplitude                              -> preserve, flag
     - flat / dead segments                                  -> reject window
3. Window-level accept / reject on the fixed 8-s grid (0-8, 8-16, ...).
4. Robust normalization, with masked samples linearly interpolated and reported
   through a per-sample mask so the reconstruction loss can ignore them.

Nothing is ever clipped: amplitude that survives detection reaches the model at
its true robust scale.

Command line
------------
    python -m utils.preprocessing_pipeline scan --manifest ... --shards-dir ... --output-dir ...
    python -m utils.preprocessing_pipeline selftest
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826

# Window statuses
ACCEPT = "accept"
REJECT = "reject"

# Event kinds
IMPULSE = "impulse"                    # isolated, near-instantaneous -> masked
CHANNEL_TRANSIENT = "channel_transient"  # isolated, short            -> masked
CHANNEL_ARTIFACT = "channel_artifact"  # isolated, too long to mask   -> reject
BROADBAND_STEP = "broadband_step"      # many channels at once        -> reject
STRUCTURED_SHARP = "structured_sharp"  # spatially structured, sharp  -> preserved
SUSTAINED = "sustained"                # long high amplitude          -> preserved

EVENT_KINDS = (IMPULSE, CHANNEL_TRANSIENT, CHANNEL_ARTIFACT, BROADBAND_STEP,
               STRUCTURED_SHARP, SUSTAINED)


@dataclass
class QCConfig:
    """Thresholds for detection and window acceptance.

    Amplitude thresholds are in robust z units: (x - median) / (1.4826 * MAD),
    per channel, per recording.
    """

    window_sec: float = 8.0
    lead_in_sec: float = 16.0            # excluded from statistics, still judged
    drop_lead_in: bool = False           # reject lead-in windows outright

    # Statistics
    stat_trim_z: float = 8.0             # second-pass trim for median/MAD

    # Amplitude thresholds
    z_extreme: float = 12.0              # "enormous" transient
    z_involved: float = 6.0              # channel counts as participating
    dz_jump: float = 8.0                 # per-sample step, in robust units
    z_implausible: float = 50.0          # beyond physiology: reject, never clip

    # Temporal classification
    impulse_max_ms: float = 40.0
    channel_transient_max_ms: float = 200.0
    structured_max_ms: float = 500.0
    sustained_min_ms: float = 500.0
    burst_link_ms: float = 125.0         # crossings closer than twice this are one burst
    mask_pad_ms: float = 20.0
    jump_dilate_ms: float = 250.0

    # Spatial classification
    isolated_max_channels: int = 2       # at most this many -> electrode-local
    broadband_channel_frac: float = 0.5  # at least this share -> movement/reference
    broadband_min_extreme_channels: int = 2

    # Window acceptance
    suspect_channel_cover: float = 0.5   # share of a window under isolated sustained artifact
    max_window_mask_frac: float = 0.02
    max_channel_mask_frac: float = 0.10
    flat_scale_ratio: float = 0.05       # window MAD below this share of recording MAD
    flat_min_channel_frac: float = 0.2
    max_bad_channel_frac: float = 0.34

    # Reporting
    min_accept_frac: float = 0.8         # recordings below this are flagged suspect
    scale_mode: str = "per_channel"      # or "per_recording"

    def __post_init__(self) -> None:
        if self.scale_mode not in ("per_channel", "per_recording"):
            raise ValueError("scale_mode must be 'per_channel' or 'per_recording'")
        if not 0 < self.broadband_channel_frac <= 1:
            raise ValueError("broadband_channel_frac must be in (0, 1]")
        if self.z_involved <= 0 or self.z_extreme <= self.z_involved:
            raise ValueError("Require 0 < z_involved < z_extreme")
        if self.z_implausible <= self.z_extreme:
            raise ValueError("Require z_implausible > z_extreme")
        if self.impulse_max_ms > self.channel_transient_max_ms:
            raise ValueError("impulse_max_ms must not exceed channel_transient_max_ms")


@dataclass
class RecordingStats:
    """Per-channel robust location and scale, in raw signal units."""

    median: np.ndarray        # (C,)
    scale: np.ndarray         # (C,) = 1.4826 * MAD, never zero
    diff_scale: np.ndarray    # (C,) robust scale of the first difference
    degenerate: np.ndarray    # (C,) bool, MAD was zero / non-finite


@dataclass
class RecordingQC:
    stats: RecordingStats
    events: pd.DataFrame
    windows: pd.DataFrame
    summary: dict


# ---------------------------------------------------------------------------
# small array helpers
# ---------------------------------------------------------------------------


def _runs(flags: np.ndarray) -> np.ndarray:
    """Half-open [start, stop) index pairs of the True runs in a 1-D mask."""
    if flags.ndim != 1:
        raise ValueError("_runs expects a 1-D boolean array")
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2)


def _dilate(flags: np.ndarray, radius: int) -> np.ndarray:
    """True wherever any True lies within `radius` samples."""
    if radius <= 0:
        return flags.astype(bool)
    counts = np.cumsum(np.concatenate(([0], flags.astype(np.int64))))
    n = flags.size
    start = np.clip(np.arange(n) - radius, 0, n)
    stop = np.clip(np.arange(n) + radius + 1, 0, n)
    return (counts[stop] - counts[start]) > 0


def _mad(values: np.ndarray, axis: int = -1) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis) * MAD_TO_SIGMA


def _samples(milliseconds: float, sfreq: float) -> int:
    return int(round(milliseconds * sfreq / 1000.0))


# ---------------------------------------------------------------------------
# stage 1: robust statistics
# ---------------------------------------------------------------------------


def robust_stats(x: np.ndarray, sfreq: float, config: QCConfig) -> RecordingStats:
    """Two-pass per-channel median / MAD, ignoring the setup lead-in.

    The second pass drops samples beyond `stat_trim_z` of the first pass, so a
    long artifact burst cannot inflate the scale that everything else is
    measured against.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (C, T) signal, got shape {x.shape}")
    lead_in = int(round(config.lead_in_sec * sfreq))
    body = x[:, lead_in:] if x.shape[1] - lead_in >= int(round(4 * sfreq)) else x
    body = np.where(np.isfinite(body), body, np.nan)

    median = np.nanmedian(body, axis=1)
    mad = np.nanmedian(np.abs(body - median[:, None]), axis=1) * MAD_TO_SIGMA

    scale = np.where(np.isfinite(mad) & (mad > 0), mad, np.nan)
    z = (body - median[:, None]) / np.where(np.isnan(scale), 1.0, scale)[:, None]
    keep = np.abs(z) <= config.stat_trim_z
    for channel in range(x.shape[0]):
        kept = body[channel][keep[channel]]
        if kept.size < max(16, int(0.1 * body.shape[1])):
            continue
        channel_median = np.median(kept)
        channel_mad = np.median(np.abs(kept - channel_median)) * MAD_TO_SIGMA
        median[channel] = channel_median
        if np.isfinite(channel_mad) and channel_mad > 0:
            scale[channel] = channel_mad

    degenerate = ~np.isfinite(scale) | (scale <= 0)
    if degenerate.all():
        scale = np.ones_like(scale)
    else:
        scale = np.where(degenerate, np.median(scale[~degenerate]), scale)

    if config.scale_mode == "per_recording":
        shared = float(np.median(scale[~degenerate])) if not degenerate.all() else 1.0
        scale = np.full_like(scale, shared)

    difference = np.diff(np.where(np.isfinite(x), x, np.nan), axis=1)
    diff_scale = np.nanmedian(np.abs(difference - np.nanmedian(difference, axis=1, keepdims=True)),
                              axis=1) * MAD_TO_SIGMA
    bad_diff = ~np.isfinite(diff_scale) | (diff_scale <= 0)
    diff_scale = np.where(bad_diff, scale, diff_scale)

    return RecordingStats(median=median.astype(np.float64), scale=scale.astype(np.float64),
                          diff_scale=diff_scale.astype(np.float64), degenerate=degenerate)


def robust_z(x: np.ndarray, stats: RecordingStats) -> np.ndarray:
    return (x - stats.median[:, None]) / stats.scale[:, None]


# ---------------------------------------------------------------------------
# stage 2: detection
# ---------------------------------------------------------------------------


def _broadband_mask(z: np.ndarray, dz: np.ndarray, good: np.ndarray,
                    sfreq: float, config: QCConfig) -> np.ndarray:
    """Samples belonging to a simultaneous broadband step across many channels.

    Requires both a spatially shared *step* (large first difference on many
    channels at the same instant) and a spatially shared *excursion*. Sustained
    generalized high amplitude fails the step test and is left alone.
    """
    n_good = int(good.sum())
    if n_good == 0:
        return np.zeros(z.shape[1], dtype=bool)
    involved = (np.abs(z[good]) > config.z_involved).sum(axis=0) / n_good
    extreme = (np.abs(z[good]) > config.z_extreme).sum(axis=0)
    jump = (np.abs(dz[good]) > config.dz_jump).sum(axis=0) / n_good

    step = _dilate(jump >= config.broadband_channel_frac,
                   _samples(config.jump_dilate_ms, sfreq))
    excursion = (involved >= config.broadband_channel_frac) & (
        extreme >= config.broadband_min_extreme_channels
    )
    hit = step & excursion
    if not hit.any():
        return hit
    # Extend to the full excursion the step belongs to, so a partially covered
    # movement burst is rejected in one piece.
    out = np.zeros_like(hit)
    for start, stop in _runs(excursion):
        if hit[start:stop].any():
            out[start:stop] = True
    return out


def detect_events(x: np.ndarray, sfreq: float, stats: RecordingStats,
                  bad_channels: np.ndarray, config: QCConfig
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Classify extreme excursions, one event per channel and burst.

    A burst is a group of threshold crossings linked across gaps shorter than
    twice `burst_link_ms`, so an oscillatory 4-s discharge is one sustained
    event rather than a few hundred spikes. Duration is measured on the burst
    envelope; the shape criteria then separate an electrode pop from EEG.

    Returns the event table, the mask to interpolate (C, T), the broadband
    artifact mask (T,), and the coverage mask (C, T) of channel artifacts too
    long to interpolate, which marks channels as locally unusable.
    """
    n_channels, n_samples = x.shape
    good = ~bad_channels
    z = robust_z(x, stats)
    z = np.where(np.isfinite(z), z, 0.0)
    difference = np.diff(x, axis=1, prepend=x[:, :1])
    dz = np.where(np.isfinite(difference), difference, 0.0) / stats.diff_scale[:, None]

    broadband = _broadband_mask(z, dz, good, sfreq, config)
    n_good = max(int(good.sum()), 1)
    n_involved = ((np.abs(z) > config.z_involved) & good[:, None]).sum(axis=0)

    mask = np.zeros((n_channels, n_samples), dtype=bool)
    suspect = np.zeros((n_channels, n_samples), dtype=bool)
    pad = _samples(config.mask_pad_ms, sfreq)
    link = max(_samples(config.burst_link_ms, sfreq), 1)
    impulse_max = max(_samples(config.impulse_max_ms, sfreq), 1)
    transient_max = max(_samples(config.channel_transient_max_ms, sfreq), 1)
    structured_max = max(_samples(config.structured_max_ms, sfreq), 1)
    sustained_min = max(_samples(config.sustained_min_ms, sfreq), 1)

    rows = []
    for channel in range(n_channels):
        exceed = np.abs(z[channel]) > config.z_extreme
        if not exceed.any():
            continue
        extreme_runs = _runs(exceed)
        linked = _runs(_dilate(np.abs(z[channel]) > config.z_involved, link))
        burst_of_run = np.searchsorted(linked[:, 0], extreme_runs[:, 0], side="right") - 1

        for burst_index in np.unique(burst_of_run):
            members = extreme_runs[burst_of_run == burst_index]
            burst_start, burst_stop = linked[burst_index]
            # Undo the linking dilation to recover the true envelope extent.
            envelope = max(int(burst_stop - burst_start) - 2 * link, 1)
            longest = int(np.max(members[:, 1] - members[:, 0]))
            start, stop = int(members[0, 0]), int(members[-1, 1])
            peers = int(max(np.max(n_involved[lo:hi]) for lo, hi in members))
            isolated = peers <= config.isolated_max_channels
            overlaps_broadband = bool(broadband[start:stop].any())

            if overlaps_broadband:
                # The window goes on the broadband verdict; nothing to mask.
                kind, masked = BROADBAND_STEP, False
            elif isolated:
                # One or two channels is an electrode, not a generator: an
                # excursion here is artifact however long it lasts. Short ones
                # are cheap to interpolate, long ones cost the window.
                if longest <= impulse_max:
                    kind, masked = IMPULSE, True
                elif longest <= transient_max:
                    kind, masked = CHANNEL_TRANSIENT, True
                else:
                    kind, masked = CHANNEL_ARTIFACT, False
            elif envelope >= sustained_min:
                kind, masked = SUSTAINED, False
            elif longest <= structured_max:
                kind, masked = STRUCTURED_SHARP, False
            else:
                kind, masked = SUSTAINED, False

            peak = float(np.max(np.abs(z[channel, start:stop])))
            if masked:
                # Mask the whole burst span, which is what the runtime rebuilds
                # from the event table, so the budget below counts what training
                # will actually interpolate.
                mask[channel, max(start - pad, 0):min(stop + pad, n_samples)] = True
            if kind == CHANNEL_ARTIFACT:
                suspect[channel, max(burst_start, 0):min(burst_stop, n_samples)] = True
            rows.append({
                "channel_index": channel,
                "start_sample": start,
                "stop_sample": stop,
                "duration_ms": longest * 1000.0 / sfreq,
                "envelope_ms": envelope * 1000.0 / sfreq,
                "n_crossings": int(len(members)),
                "kind": kind,
                "masked": masked,
                "isolated": isolated,
                # Amplitude no physiology reaches; left in the signal it would
                # dominate the reconstruction loss, so the window goes.
                "implausible": bool(peak > config.z_implausible and not masked),
                "peak_abs_z": peak,
                "peak_abs_dz": float(np.max(np.abs(dz[channel, start:stop]))),
                "n_channels_involved": peers,
                "frac_channels_involved": peers / n_good,
                "channel_is_flagged_bad": bool(bad_channels[channel]),
            })

    columns = ["channel_index", "start_sample", "stop_sample", "duration_ms", "envelope_ms",
               "n_crossings", "kind", "masked", "isolated", "implausible", "peak_abs_z",
               "peak_abs_dz", "n_channels_involved", "frac_channels_involved",
               "channel_is_flagged_bad"]
    events = pd.DataFrame(rows, columns=columns)
    return events, mask, broadband, suspect


# ---------------------------------------------------------------------------
# stage 3: window decisions
# ---------------------------------------------------------------------------


def _window_view(values: np.ndarray, window: int) -> np.ndarray:
    """(C, T) -> (C, n_windows, window) on the fixed non-overlapping grid."""
    n_windows = values.shape[1] // window
    return values[:, : n_windows * window].reshape(values.shape[0], n_windows, window)


def assess_windows(x: np.ndarray, sfreq: float, stats: RecordingStats, mask: np.ndarray,
                   broadband: np.ndarray, suspect: np.ndarray, events: pd.DataFrame,
                   bad_channels: np.ndarray, config: QCConfig) -> pd.DataFrame:
    n_channels, n_samples = x.shape
    window = int(round(config.window_sec * sfreq))
    if window <= 0:
        raise ValueError("window_sec must produce a positive sample count")
    n_windows = n_samples // window
    if n_windows == 0:
        return pd.DataFrame()

    good = ~bad_channels
    n_good = max(int(good.sum()), 1)
    z = robust_z(x, stats)

    windowed = _window_view(x, window)
    window_mad = _mad(np.where(np.isfinite(windowed), windowed, np.nan), axis=2)
    flat = window_mad < config.flat_scale_ratio * stats.scale[:, None]
    flat &= good[:, None]
    n_flat = flat.sum(axis=0)

    nonfinite = (~np.isfinite(windowed)).sum(axis=(0, 2))
    mask_frac_channel = _window_view(mask, window).mean(axis=2)     # (C, n_windows)
    mask_frac = mask_frac_channel.mean(axis=0)
    max_channel_mask_frac = mask_frac_channel.max(axis=0)
    has_broadband = broadband[: n_windows * window].reshape(n_windows, window).any(axis=1)
    suspect_channels = ((_window_view(suspect, window).mean(axis=2) > config.suspect_channel_cover)
                        & good[:, None])
    n_suspect = suspect_channels.sum(axis=0)
    unusable_frac = (n_suspect + int(bad_channels.sum())) / max(n_channels, 1)
    # Non-finite samples read as zero here; the nonfinite rule rejects them anyway.
    peak_abs_z = _window_view(np.where(np.isfinite(z), np.abs(z), 0.0), window).max(axis=(0, 2))

    counts = {kind: np.zeros(n_windows, dtype=int) for kind in EVENT_KINDS}
    implausible = np.zeros(n_windows, dtype=bool)
    channel_artifact = np.zeros(n_windows, dtype=bool)
    if not events.empty:
        for kind, group in events.groupby("kind"):
            index = np.clip(group["start_sample"].to_numpy() // window, 0, n_windows - 1)
            np.add.at(counts[kind], index, 1)
        # An event spans every window it touches, not just the one it starts in.
        for flags, selected in ((implausible, events[events["implausible"]]),
                                (channel_artifact, events[events["kind"] == CHANNEL_ARTIFACT])):
            for _, event in selected.iterrows():
                lo = min(int(event["start_sample"]) // window, n_windows - 1)
                hi = min(int(event["stop_sample"]) // window, n_windows - 1)
                flags[lo: hi + 1] = True

    lead_in_windows = int(np.ceil(config.lead_in_sec / config.window_sec))

    rows = []
    for index in range(n_windows):
        in_lead_in = index < lead_in_windows
        reason = ""
        if nonfinite[index]:
            reason = "nonfinite"
        elif n_flat[index] >= max(1, int(round(config.flat_min_channel_frac * n_good))):
            reason = "flat"
        elif has_broadband[index]:
            reason = "broadband_step"
        elif implausible[index]:
            reason = "implausible_amplitude"
        elif channel_artifact[index]:
            reason = "channel_artifact"
        elif mask_frac[index] > config.max_window_mask_frac:
            reason = "mask_budget"
        elif max_channel_mask_frac[index] > config.max_channel_mask_frac:
            reason = "channel_mask_budget"
        elif unusable_frac[index] > config.max_bad_channel_frac:
            reason = "bad_channels"
        elif in_lead_in and config.drop_lead_in:
            reason = "lead_in"
        rows.append({
            "window_index": index,
            "start_sample": index * window,
            "start_sec": index * window / sfreq,
            "stop_sec": (index + 1) * window / sfreq,
            "status": REJECT if reason else ACCEPT,
            "reason": reason,
            "in_lead_in": in_lead_in,
            "masked_frac": float(mask_frac[index]),
            "max_channel_masked_frac": float(max_channel_mask_frac[index]),
            "n_flat_channels": int(n_flat[index]),
            "n_suspect_channels": int(n_suspect[index]),
            "has_implausible_amplitude": bool(implausible[index]),
            "n_nonfinite_samples": int(nonfinite[index]),
            "peak_abs_z": float(peak_abs_z[index]),
            **{f"n_{kind}": int(counts[kind][index]) for kind in EVENT_KINDS},
        })
    return pd.DataFrame(rows)


def analyze_recording(x: np.ndarray, sfreq: float, config: QCConfig | None = None,
                      bad_channel_indices: list[int] | None = None) -> RecordingQC:
    """Run statistics, detection and window scoring on one (C, T) recording."""
    config = config or QCConfig()
    x = np.asarray(x, dtype=np.float64)
    bad_channels = np.zeros(x.shape[0], dtype=bool)
    if bad_channel_indices:
        bad_channels[np.asarray(bad_channel_indices, dtype=int)] = True

    stats = robust_stats(x, sfreq, config)
    events, mask, broadband, suspect = detect_events(x, sfreq, stats, bad_channels, config)
    windows = assess_windows(x, sfreq, stats, mask, broadband, suspect, events, bad_channels,
                             config)

    accepted = int((windows["status"] == ACCEPT).sum()) if not windows.empty else 0
    total = int(len(windows))
    summary = {
        "n_channels": int(x.shape[0]),
        "n_samples": int(x.shape[1]),
        "sfreq": float(sfreq),
        "n_windows": total,
        "n_accepted_windows": accepted,
        "accept_frac": accepted / total if total else 0.0,
        "n_flagged_bad_channels": int(bad_channels.sum()),
        "n_degenerate_channels": int(stats.degenerate.sum()),
        "masked_sample_frac": float(mask.mean()),
        "n_events": int(len(events)),
        "suspect": bool(total == 0 or accepted / total < config.min_accept_frac),
    }
    for kind in EVENT_KINDS:
        summary[f"n_{kind}"] = int((events["kind"] == kind).sum()) if not events.empty else 0
    return RecordingQC(stats=stats, events=events, windows=windows, summary=summary)


# ---------------------------------------------------------------------------
# stage 4: normalization
# ---------------------------------------------------------------------------


def fill_masked(x: np.ndarray, mask: np.ndarray, fill_value: np.ndarray | None = None) -> np.ndarray:
    """Linearly interpolate masked samples per channel; edges take a constant."""
    out = np.array(x, dtype=np.float64, copy=True)
    for channel in range(out.shape[0]):
        bad = mask[channel]
        if not bad.any():
            continue
        good = ~bad
        if not good.any():
            out[channel] = 0.0 if fill_value is None else fill_value[channel]
            continue
        index = np.flatnonzero(bad)
        out[channel, bad] = np.interp(index, np.flatnonzero(good), out[channel, good])
    return out


def normalize(x: np.ndarray, stats: RecordingStats, mask: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Robust-normalize and interpolate masked samples. Returns (z, mask)."""
    z = robust_z(np.asarray(x, dtype=np.float64), stats)
    z = np.where(np.isfinite(z), z, np.nan)
    if mask is None:
        mask = np.zeros(z.shape, dtype=bool)
    mask = mask | ~np.isfinite(z)
    z = fill_masked(np.where(np.isfinite(z), z, 0.0), mask)
    return z.astype(np.float32), mask


# ---------------------------------------------------------------------------
# artifacts on disk + runtime access
# ---------------------------------------------------------------------------


def qc_tables(recording_id: str, qc: RecordingQC, channel_names: list[str] | None = None,
              site: str = "") -> dict[str, pd.DataFrame]:
    """The four per-recording tables, keyed by output file stem."""
    n_channels = qc.summary["n_channels"]
    names = channel_names or [str(i) for i in range(n_channels)]
    windows = qc.windows.copy()
    windows.insert(0, "sha256_id", recording_id)
    events = qc.events.copy()
    events.insert(0, "sha256_id", recording_id)
    stats = pd.DataFrame({
        "sha256_id": recording_id,
        "channel_index": np.arange(n_channels),
        "channel_name": names,
        "median": qc.stats.median,
        "scale": qc.stats.scale,
        "diff_scale": qc.stats.diff_scale,
        "degenerate": qc.stats.degenerate,
    })
    recordings = pd.DataFrame([{"sha256_id": recording_id, "site": site, **qc.summary}])
    return {"window_qc": windows, "events": events, "channel_stats": stats,
            "recording_qc": recordings}


def write_qc_tables(tables: dict[str, list[pd.DataFrame]], output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for stem, frames in tables.items():
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(output_dir / f"{stem}.parquet", index=False)
        written[stem] = frame
    return written


def _load_signal(shards_dir: Path, shard_name: str, index_in_shard: int, n_samples: int,
                 handles: dict) -> np.ndarray:
    import h5py

    if shard_name not in handles:
        handles[shard_name] = h5py.File(shards_dir / shard_name, "r")
    return np.asarray(handles[shard_name]["signals"][int(index_in_shard), :, :int(n_samples)],
                      dtype=np.float64)


TABLE_STEMS = ("window_qc", "events", "channel_stats", "recording_qc")


def load_manifest(manifest_path: str | Path, sfreq: float | None = None,
                  limit: int = 0) -> pd.DataFrame:
    """Recordings eligible for QC, in manifest order."""
    manifest_path = Path(manifest_path)
    records = (pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet"
               else pd.read_csv(manifest_path))
    if "status" in records:
        records = records[records["status"] == "ok"]
    if sfreq is not None:
        records = records[np.isclose(records["sfreq"], sfreq)]
    records = records.reset_index(drop=True)
    if limit > 0:
        records = records.head(limit)
    if records.empty:
        raise ValueError("No recordings selected from the manifest")
    return records


def load_bad_channels(bad_channels_path: str | Path | None) -> dict[str, set[str]]:
    if bad_channels_path is None or not Path(bad_channels_path).exists():
        return {}
    bad_table = pd.read_parquet(bad_channels_path)
    return {str(key): set(group["channel_name"].astype(str))
            for key, group in bad_table.groupby("sha256_id")}


def process_recording(row: pd.Series, shards_dir: Path, config: QCConfig,
                      bad_by_recording: dict[str, set[str]], handles: dict
                      ) -> tuple[dict[str, pd.DataFrame], dict]:
    """QC one manifest row. `handles` caches open HDF5 shards across calls."""
    recording_id = str(row["sha256_id"])
    x = _load_signal(Path(shards_dir), str(row["shard_name"]), row["index_in_shard"],
                     row["n_samples"], handles)
    names = row.get("channel_names")
    if isinstance(names, str):
        names = json.loads(names)
    names = [str(name) for name in names] if names is not None else [
        str(i) for i in range(x.shape[0])
    ]
    flagged = bad_by_recording.get(recording_id, set())
    bad_indices = [i for i, name in enumerate(names) if name in flagged]

    qc = analyze_recording(x, float(row["sfreq"]), config, bad_indices)
    tables = qc_tables(recording_id, qc, names, str(row["site"]) if "site" in row else "")
    tables["channel_stats"]["flagged_bad"] = [
        channel in bad_indices for channel in range(x.shape[0])
    ]
    return tables, qc.summary


def summarize_scan(windows: pd.DataFrame, events: pd.DataFrame, recordings: pd.DataFrame,
                   config: QCConfig, **extra) -> dict:
    """The report written next to the QC tables."""
    return {
        **extra,
        "config": asdict(config),
        "n_recordings": int(len(recordings)),
        "n_windows": int(len(windows)),
        "n_accepted_windows": int((windows["status"] == ACCEPT).sum()),
        "accept_frac": float((windows["status"] == ACCEPT).mean()),
        "reject_reasons": windows.loc[windows["status"] == REJECT, "reason"]
                                 .value_counts().to_dict(),
        "event_kinds": events["kind"].value_counts().to_dict() if "kind" in events else {},
        "n_suspect_recordings": int(recordings["suspect"].sum()),
        "n_recordings_without_windows": int((recordings["n_accepted_windows"] == 0).sum()),
        "median_recording_accept_frac": float(recordings["accept_frac"].median()),
    }


def scan_manifest(manifest_path: str | Path, shards_dir: str | Path, output_dir: str | Path,
                  config: QCConfig, bad_channels_path: str | Path | None = None,
                  limit: int = 0, sfreq: float | None = None) -> dict:
    """Run the pipeline over a manifest in one process and write the QC tables.

    For the full dataset use `build_qc_cache.py`, which adds chunked resume and
    worker processes over the same per-recording code path.
    """
    manifest_path, shards_dir, output_dir = Path(manifest_path), Path(shards_dir), Path(output_dir)
    records = load_manifest(manifest_path, sfreq, limit)
    bad_by_recording = load_bad_channels(bad_channels_path)

    handles: dict = {}
    collected: dict[str, list[pd.DataFrame]] = {stem: [] for stem in TABLE_STEMS}
    try:
        for _, row in records.iterrows():
            tables, summary = process_recording(row, shards_dir, config, bad_by_recording, handles)
            for stem, frame in tables.items():
                collected[stem].append(frame)
            print(f"{str(row['sha256_id'])[:12]} accept={summary['accept_frac']:.3f} "
                  f"windows={summary['n_windows']} events={summary['n_events']}"
                  f"{' SUSPECT' if summary['suspect'] else ''}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    written = write_qc_tables(collected, output_dir)
    windows, events, recordings = (written["window_qc"], written["events"],
                                   written["recording_qc"])

    report = summarize_scan(windows, events, recordings, config,
                            manifest=str(manifest_path), shards_dir=str(shards_dir))
    (output_dir / "preprocessing_summary.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    return report


class WindowPreprocessor:
    """Runtime access to the QC tables: which windows to use and how to scale them."""

    def __init__(self, qc_dir: str | Path, config: QCConfig | None = None) -> None:
        qc_dir = Path(qc_dir)
        self.config = config or QCConfig()
        windows = pd.read_parquet(qc_dir / "window_qc.parquet")
        self.accepted = {
            str(key): group["start_sample"].to_numpy(dtype=np.int64)
            for key, group in windows[windows["status"] == ACCEPT].groupby("sha256_id")
        }
        stats = pd.read_parquet(qc_dir / "channel_stats.parquet").sort_values(
            ["sha256_id", "channel_index"]
        )
        self.stats: dict[str, RecordingStats] = {
            str(key): RecordingStats(
                median=group["median"].to_numpy(dtype=np.float64),
                scale=group["scale"].to_numpy(dtype=np.float64),
                diff_scale=group["diff_scale"].to_numpy(dtype=np.float64),
                degenerate=group["degenerate"].to_numpy(dtype=bool),
            )
            for key, group in stats.groupby("sha256_id")
        }
        events = pd.read_parquet(qc_dir / "events.parquet")
        self.masked_events: dict[str, np.ndarray] = {}
        if "masked" in events:
            masked = events[events["masked"]]
            for key, group in masked.groupby("sha256_id"):
                self.masked_events[str(key)] = group[
                    ["channel_index", "start_sample", "stop_sample"]
                ].to_numpy(dtype=np.int64)

    def accepted_starts(self, recording_id: str) -> np.ndarray:
        return self.accepted.get(recording_id, np.zeros(0, dtype=np.int64))

    def apply(self, x: np.ndarray, recording_id: str, start_sample: int, sfreq: float
              ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize one raw window and return (normalized, masked-sample mask)."""
        stats = self.stats[recording_id]
        mask = np.zeros(x.shape, dtype=bool)
        pad = _samples(self.config.mask_pad_ms, sfreq)
        stop_sample = start_sample + x.shape[1]
        for channel, start, stop in self.masked_events.get(recording_id, np.zeros((0, 3), int)):
            lo, hi = max(start - pad, start_sample), min(stop + pad, stop_sample)
            if lo < hi:
                mask[channel, lo - start_sample: hi - start_sample] = True
        return normalize(x, stats, mask)


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------


def _synthetic_recording(sfreq: float = 256.0, duration_sec: float = 300.0,
                         n_channels: int = 20, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Pink-ish, spatially mixed noise with one instance of each artifact class."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sfreq)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    shaping = 1.0 / np.sqrt(np.maximum(freqs, 0.5))
    sources = np.fft.irfft(np.fft.rfft(rng.standard_normal((n_channels, n)), axis=1) * shaping,
                           n=n, axis=1)
    mixing = np.eye(n_channels) + 0.4 * rng.standard_normal((n_channels, n_channels)) / n_channels
    x = mixing @ sources
    x *= 20.0 / np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1,
                          keepdims=True) / MAD_TO_SIGMA  # ~20 uV robust sigma
    sigma = _mad(x, axis=1)[:, None]

    truth = {}
    x[:, : int(8 * sfreq)] = 0.0                                  # flat setup lead-in
    truth["flat_window"] = 0

    at = int(40 * sfreq)                                          # isolated impulse, 3 samples
    x[3, at: at + 3] += 45 * sigma[3]
    truth["impulse_window"] = at // int(8 * sfreq)

    at = int(80 * sfreq)                                          # broadband movement step
    step = np.concatenate([np.linspace(0, 1, 4), np.ones(int(0.4 * sfreq)),
                           np.linspace(1, 0, int(0.2 * sfreq))])
    x[:, at: at + step.size] += 25 * sigma * step * (1 + 0.2 * rng.standard_normal((n_channels, 1)))
    truth["broadband_window"] = at // int(8 * sfreq)

    at = int(120 * sfreq)                                         # structured sharp transient
    length = int(0.1 * sfreq)
    shape = np.sin(np.pi * np.arange(length) / length)
    x[4:10, at: at + length] += 14 * sigma[4:10] * shape
    truth["structured_window"] = at // int(8 * sfreq)

    at = int(160 * sfreq)                                         # sustained high amplitude
    length = int(4 * sfreq)
    burst = 13 * np.sin(2 * np.pi * 3 * np.arange(length) / sfreq)
    x[2:14, at: at + length] += sigma[2:14] * burst
    truth["sustained_window"] = at // int(8 * sfreq)
    truth["sustained_stop_window"] = (at + length) // int(8 * sfreq)

    at = int(240 * sfreq)                                         # single-channel long transient
    length = int(0.6 * sfreq)
    x[11, at: at + length] += 20 * sigma[11, 0] * np.sin(np.pi * np.arange(length) / length)
    truth["channel_artifact_window"] = at // int(8 * sfreq)

    at = int(200 * sfreq)                                         # saturating electrode
    length = int(3 * sfreq)
    drift = 90 * np.sin(2 * np.pi * 0.5 * np.arange(length) / sfreq)
    x[7, at: at + length] += sigma[7, 0] * drift
    truth["saturated_window"] = at // int(8 * sfreq)
    return x, truth


def selftest() -> None:
    sfreq = 256.0
    x, truth = _synthetic_recording(sfreq=sfreq)
    config = QCConfig()
    qc = analyze_recording(x, sfreq, config)
    windows = qc.windows.set_index("window_index")
    events = qc.events
    failures = []

    def check(condition: bool, message: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    flat = windows.loc[truth["flat_window"]]
    check(flat["status"] == REJECT and flat["reason"] == "flat",
          f"flat lead-in window rejected as flat (got {flat['reason']!r})")

    index = truth["impulse_window"]
    impulse_events = events[(events["kind"] == IMPULSE) &
                            (events["start_sample"] // int(8 * sfreq) == index)]
    check(len(impulse_events) == 1, f"single-sample transient classified as impulse "
                                    f"(got {len(impulse_events)})")
    check(windows.loc[index, "status"] == ACCEPT,
          f"impulse window kept, artifact masked (got {windows.loc[index, 'reason']!r})")
    check(0 < windows.loc[index, "masked_frac"] <= config.max_window_mask_frac,
          f"impulse masked a small sample fraction ({windows.loc[index, 'masked_frac']:.5f})")

    index = truth["broadband_window"]
    check(windows.loc[index, "status"] == REJECT and
          windows.loc[index, "reason"] == "broadband_step",
          f"broadband step window rejected (got {windows.loc[index, 'reason']!r})")

    index = truth["structured_window"]
    check(windows.loc[index, "n_structured_sharp"] > 0,
          f"spatially structured transient classified as structured_sharp "
          f"({windows.loc[index, 'n_structured_sharp']} events)")
    check(windows.loc[index, "status"] == ACCEPT and windows.loc[index, "masked_frac"] == 0,
          "structured transient preserved, nothing masked")

    lo, hi = truth["sustained_window"], truth["sustained_stop_window"]
    sustained = windows.loc[lo:hi]
    check((sustained["status"] == ACCEPT).all(), "sustained high-amplitude windows kept")
    check((sustained["masked_frac"] == 0).all(), "sustained high amplitude never masked")
    check(sustained["peak_abs_z"].max() > 10,
          f"sustained amplitude reaches the model unclipped "
          f"(peak |z| = {sustained['peak_abs_z'].max():.1f})")

    index = truth["channel_artifact_window"]
    check(windows.loc[index, "status"] == REJECT and
          windows.loc[index, "reason"] == "channel_artifact",
          f"single-channel 600 ms transient rejects the window rather than passing as "
          f"structure (got {windows.loc[index, 'reason']!r})")

    index = truth["saturated_window"]
    check(windows.loc[index, "status"] == REJECT and
          windows.loc[index, "reason"] == "implausible_amplitude",
          f"saturating single channel rejects the window, unclipped "
          f"(got {windows.loc[index, 'reason']!r})")

    stats = qc.stats
    z, _ = normalize(x, stats)
    body = z[:, int(16 * sfreq):]
    check(np.allclose(np.median(body, axis=1), 0, atol=0.15),
          "normalized channels are centred on zero")
    check(np.allclose(_mad(body, axis=1), 1.0, atol=0.2),
          f"normalized channels have unit robust scale (MAD range "
          f"{_mad(body, axis=1).min():.2f}-{_mad(body, axis=1).max():.2f})")

    accept_frac = qc.summary["accept_frac"]
    check(accept_frac >= config.min_accept_frac,
          f"accept fraction on a mostly clean recording is {accept_frac:.3f} (>= 0.8)")

    # Round-trip through the QC tables the way training will read them.
    window_samples = int(config.window_sec * sfreq)
    with tempfile.TemporaryDirectory() as directory:
        qc_dir = Path(directory)
        write_qc_tables({stem: [frame] for stem, frame in qc_tables("synthetic", qc).items()},
                        qc_dir)
        runtime = WindowPreprocessor(qc_dir, config)
        starts = set(runtime.accepted_starts("synthetic").tolist())
        check(truth["impulse_window"] * window_samples in starts and
              truth["broadband_window"] * window_samples not in starts,
              "runtime window index keeps the impulse window and drops the broadband one")
        start = truth["impulse_window"] * window_samples
        window = x[:, start: start + window_samples]
        normalized, window_mask = runtime.apply(window, "synthetic", start, sfreq)
        raw_peak = float(np.abs(robust_z(window, stats)).max())
        check(window_mask.any() and float(np.abs(normalized).max()) < raw_peak / 4,
              f"runtime masking removes the impulse (|z| {raw_peak:.0f} -> "
              f"{float(np.abs(normalized).max()):.1f} over {int(window_mask.sum())} samples)")
        offline_mask = detect_events(x, sfreq, stats, np.zeros(x.shape[0], dtype=bool), config)[1]
        check(np.array_equal(window_mask, offline_mask[:, start: start + window_samples]),
              "runtime mask reproduces the mask the QC budget was computed from")

    print("\nreject reasons:",
          qc.windows.loc[qc.windows["status"] == REJECT, "reason"].value_counts().to_dict())
    print("event kinds:", events["kind"].value_counts().to_dict())
    if failures:
        raise SystemExit(f"{len(failures)} self-test check(s) failed")
    print("\nAll self-test checks passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = QCConfig()
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="run QC over a recordings manifest")
    scan.add_argument("--manifest", required=True)
    scan.add_argument("--shards-dir", required=True)
    scan.add_argument("--output-dir", required=True)
    scan.add_argument("--bad-channels", default=None,
                      help="bad_channels.parquet; defaults to the manifest's sibling file")
    scan.add_argument("--limit", type=int, default=0, help="0 scans every recording")
    scan.add_argument("--sfreq", type=float, default=None, help="keep only this sampling rate")
    _add_config_arguments(scan)

    subparsers.add_parser("selftest", help="validate the detectors on synthetic data")
    args = parser.parse_args()

    if args.command == "selftest":
        selftest()
        return

    config_fields = set(asdict(QCConfig()))
    config = QCConfig(**{key: value for key, value in vars(args).items() if key in config_fields})
    bad_channels = args.bad_channels
    if bad_channels is None:
        candidate = Path(args.manifest).with_name("bad_channels.parquet")
        bad_channels = candidate if candidate.exists() else None
    report = scan_manifest(args.manifest, args.shards_dir, args.output_dir, config,
                           bad_channels, args.limit, args.sfreq)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
