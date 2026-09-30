# EEG preprocessing and QC pipeline

The recordings in `H:/EEG/FHA/Resting/preprocessed` were cleaned in an earlier project and
carry roughly flagged bad channels, but they still contain high-amplitude segments: electrode
pops, movement and reference steps, saturating channels, and flat stretches during setup. Those
segments dominate a squared-error objective and make a rate–distortion curve reflect artifact
energy rather than EEG.

This pipeline decides, **per 8-second window**, what the model is allowed to see — and it never
clips amplitude. Clipping would silently rewrite exactly the physiological and pathological
activity the encoder should learn to represent. Windows are either kept as they are, kept with a
few interpolated samples, or dropped.

Two pieces:

| File | Role |
|---|---|
| `utils/preprocessing_pipeline.py` | The algorithm: statistics, detection, window scoring, normalization |
| `experiments/build_qc_cache.py` | The one-off batch pass that writes the QC cache for a whole dataset |
| `utils/eeg_dataset.py` → `CleanEEGWindowDataset` | Training-time dataloader that reads the cache |

---

## 1. Processing steps

### Stage 1 — Robust per-channel statistics

For each recording, each channel gets a median and a MAD-based scale
(`scale = 1.4826 × MAD`), computed in two passes: the second pass drops samples beyond
`|z| > 8` of the first, so one artifact burst cannot inflate the scale that everything else is
then measured against. The first `lead_in_sec = 16` seconds are excluded from estimation
(setup artifact), though those windows are still judged normally.

Channels whose MAD is zero or non-finite are marked `degenerate` and borrow the median scale of
the other channels, so a dead channel cannot produce infinite z-scores.

Everything downstream is expressed in **robust z units**: `z = (x − median) / scale`, per
channel, per recording.

### Stage 2 — Detection, before normalization

Excursions past `z_extreme = 12` are grouped into **bursts**: crossings separated by less than
`2 × burst_link_ms = 250 ms` belong to one burst. Without this, a 4-second oscillatory discharge
is reported as several hundred separate spikes and misclassified. Duration is measured two ways —
the longest single crossing, and the burst envelope — and the shape of the burst decides what
it is:

| Pattern | Kind | Action |
|---|---|---|
| ≤ 2 channels, ≤ 40 ms | `impulse` | **Mask samples**, keep window |
| ≤ 2 channels, ≤ 200 ms | `channel_transient` | **Mask samples**, keep window |
| ≤ 2 channels, longer | `channel_artifact` | **Reject window** — too long to interpolate honestly |
| Shared step across ≥ 50% of channels | `broadband_step` | **Reject window** |
| ≥ 3 channels, tens–hundreds of ms | `structured_sharp` | **Preserve** — plausibly real EEG |
| ≥ 3 channels, envelope ≥ 500 ms | `sustained` | **Preserve, never clipped** |

Two details carry most of the weight:

- **A broadband step requires both a step and an excursion.** The step test is a large *first
  difference* on at least half the channels at the same instant; the excursion test is
  `|z| > 6` on at least half the channels with at least two above `z_extreme`. Movement and
  reference artifacts are phase-aligned jumps and fail neither test. Generalized high-amplitude
  rhythmic activity has no simultaneous step and survives.
- **One or two channels is an electrode, not a generator.** An excursion confined to ≤ 2 channels
  is treated as artifact however long it lasts. Short ones are interpolated; long ones cost the
  window. This is what keeps a single saturating channel from being mistaken for spatial
  structure.

**One hard amplitude bound.** Any preserved (unmasked) event peaking above
`z_implausible = 50` rejects its windows. No physiology sits at 50× its own background scale;
without this bound, a saturating channel at |z| = 320 was being kept as "sustained". The bound
rejects — it does not clip.

### Stage 3 — Window decisions

Windows are the fixed non-overlapping grid the model trains on: 0–8 s, 8–16 s, … A window is
rejected for the first reason that applies:

| `reason` | Meaning |
|---|---|
| `nonfinite` | NaN/Inf samples present |
| `flat` | ≥ 20% of good channels have window MAD below 5% of their recording MAD (dead/setup) |
| `broadband_step` | Movement or reference artifact |
| `implausible_amplitude` | A preserved event above `z_implausible` |
| `channel_artifact` | An isolated excursion too long to interpolate |
| `mask_budget` | More than 2% of the window's samples would be interpolated |
| `channel_mask_budget` | More than 10% of a single channel would be interpolated |
| `bad_channels` | Flagged plus locally unusable channels exceed 34% of the montage |
| `lead_in` | Only when `--drop-lead-in` is passed |

An empty `reason` means `status == "accept"`.

### Stage 4 — Normalization

Accepted windows are normalized per channel with that recording's median and scale, masked
samples are linearly interpolated, and the mask is handed to the caller so the loss can exclude
those samples. Every window reaches the model at unit robust scale with its true relative
amplitude intact.

---

## 2. Building the QC cache

```powershell
& "C:\Users\Zhenyu's PC\torch\Scripts\python.exe" -m experiments.build_qc_cache `
    --manifest H:/EEG/FHA/Resting/preprocessed/manifests/recordings.parquet `
    --shards-dir H:/EEG/FHA/Resting/preprocessed/shards `
    --output-dir H:/EEG/FHA/Resting/preprocessed/qc `
    --sfreq 256 --workers 4
```

`bad_channels.parquet` is picked up automatically from next to the manifest; override with
`--bad-channels`.

**Cost.** Measured on 32 cold random recordings: 0.29 s per recording with one worker,
0.16 s with four (the job is I/O bound on the shard reads). The full manifest is 40,716
recordings × 360 s, so expect roughly **1–3.5 hours** depending on workers and disk.

**Resumable.** Work is chunked (`--chunk-size`, default 16) and each chunk writes its own part
files under `parts/` only once complete. Re-running the same command skips finished chunks:

```
24 recordings, 6 chunks, 2 to process (4 already cached), workers=2
```

- `--restart` discards the cache and starts over.
- `--finalize-only` rebuilds the four tables from existing parts without reprocessing.
- A recording that fails to read is recorded in `failures.json` and does not sink the run.
- The thresholds used are written to `qc_config.json`. Re-running against the same directory
  with *different* thresholds is refused, so a cache can never be half one policy and half
  another.

### What the cache contains

| File | Rows | Use |
|---|---|---|
| `window_qc.parquet` | one per 8-s window | `status`, `reason`, `peak_abs_z`, per-kind event counts |
| `events.parquet` | one per channel-burst | `kind`, `masked`, `duration_ms`, `envelope_ms`, `peak_abs_z`, `n_channels_involved` |
| `channel_stats.parquet` | one per channel per recording | `median`, `scale`, `diff_scale`, `degenerate`, `flagged_bad` |
| `recording_qc.parquet` | one per recording | `accept_frac`, `suspect`, event totals |
| `preprocessing_summary.json` | — | Config, accept rate, reject-reason histogram, usable hours |
| `qc_config.json` | — | The thresholds this cache was built with |

`window_qc` is also the QC report: sort by `peak_abs_z`, group by `reason`, or join `events` to
see exactly which channel and which burst drove any decision.

---

## 3. Using the dataloader

`CleanEEGWindowDataset` subclasses the existing `EEGWindowDataset` and takes one extra argument,
the cache directory:

```python
from utils.eeg_dataset import CleanEEGWindowDataset
from utils.experiment_utils import make_loader, read_manifest, split_records

records = read_manifest(manifest, sfreq=256.0, n_channels=20, window_sec=8.0)
splits = split_records(records, "sha256_id", seed, 256, 64, 64)

train_ds = CleanEEGWindowDataset(
    splits["train"], shards_dir, qc_dir="H:/EEG/FHA/Resting/preprocessed/qc",
    window_sec=8.0, windows_per_recording=4, training=True,
)
val_ds = CleanEEGWindowDataset(
    splits["val"], shards_dir, qc_dir="H:/EEG/FHA/Resting/preprocessed/qc",
    training=False, windows_per_recording=4, eval_start_sec=16.0,
)
loader = make_loader(train_ds, batch_size=16, workers=0, shuffle=True)
```

Each item is:

```python
{"x": (C, T) float32,   # robust-normalized, unit scale, never clipped
 "mask": (C, T) bool,   # True where a sample was interpolated over an artifact
 "sha256_id": str,
 "crop_start_sample": int}
```

Behavioral differences from `EEGWindowDataset`:

- **Windows are grid-aligned**, not drawn from arbitrary offsets — acceptance was decided per
  grid window, so a window straddling two grid cells has no verdict. Training samples uniformly
  from that recording's accepted windows; evaluation takes a reproducible `linspace` over them.
- **Recordings with no accepted window are dropped** from the split automatically. Pass
  `drop_suspect_recordings=True` to also drop recordings below `min_accept_frac` (0.8).
- **Normalization is built in.** Do *not* also divide by `estimate_input_scale`; that would apply
  a second, global scale on top of the per-recording one.

### Loss with the mask

Interpolated samples are not data; exclude them:

```python
keep = ~batch["mask"]
distortion = ((x - y).square() * keep).sum() / keep.sum().clamp(min=1)
```

In practice masking touches ~0.0006% of samples, so ignoring the mask changes little — but a
window that survived on a 2% mask budget is exactly where it matters.

### Wiring into `experiments/train.py`

Three changes: construct `CleanEEGWindowDataset` instead of `EEGWindowDataset` with a
`--qc-dir` argument; drop the `scale = estimate_input_scale(...)` call and the `/ scale`
divisions in the train loop and in `evaluate`; and use the masked distortion above. Record the
`qc_dir` in `config.json` so a checkpoint says which QC policy produced it.

---

## 4. What this does to the data

Measured over 60 random recordings (2,700 windows) with default thresholds:

- **96.4% of windows accepted**; median recording 98.9%; only 2 of 60 recordings below the 80%
  floor.
- Rejections: `broadband_step` 73, `implausible_amplitude` 21, `channel_artifact` 4.
- Masking touches 0.0006% of samples — surgical, as intended.
- 15% of rejections fall in the first 16 seconds, consistent with setup artifact.

Checked against the earlier `runs/sweep_30_epochs/amplitude_audit`: of the 16 windows that audit
flagged, **all 11 with |z| > 20 are now rejected**, and the 5 kept peak at |z| ≤ 13 — ordinary
EEG that a raw-amplitude threshold mislabeled because those recordings simply have a larger
overall scale.

`python -m utils.preprocessing_pipeline selftest` builds a synthetic 1/f recording with one of each
artifact injected and asserts 18 properties, including that sustained 13σ activity reaches the
model unclipped and that the mask the dataloader rebuilds is byte-identical to the one the QC
budget was computed from.

---

## 5. Tuning and known trade-offs

| Knob | Default | Raise it to… |
|---|---|---|
| `z_implausible` | 50 | keep more extreme multi-channel activity (fewer rejects) |
| `z_extreme` | 12 | detect fewer events overall |
| `broadband_channel_frac` | 0.5 | require broader involvement before rejecting a window |
| `max_bad_channel_frac` | 0.34 | keep recordings with many flagged channels |
| `channel_transient_max_ms` | 200 | interpolate longer isolated artifacts instead of dropping the window |

Two things worth deciding deliberately:

1. **~2% of accepted windows still peak at |z| 20–48.** These are multi-channel sustained bursts
   — preserved on purpose, since that is the category that may be pathological. They will
   dominate the energy term of an NMSE-based rate–distortion curve. `window_qc.peak_abs_z` lets
   you rerun the sweep excluding them as a sensitivity check rather than guessing.
2. **Per-recording normalization changes what NMSE means.** Every recording now contributes at
   unit robust variance, so the curve no longer weights high-amplitude recordings more heavily.
   That is usually what you want for a rate–distortion study, but it is not comparable
   point-for-point with results from the old single global scale.
