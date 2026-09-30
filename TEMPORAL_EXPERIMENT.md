# Slow + fast temporal residual autoencoder

The next temporal experiment is implemented in `models/temporal_model.py` and
uses the existing training and evaluation entry points. The spatial experiment
remains separate.

## Architecture

For an input `x` of shape `[B, C, T]`:

```text
z_s = quantize(E_s(x))         slow latent, stride 64
x_s = D_s(z_s)                 coarse waveform prediction
r   = x - x_s                 residual formed from the decoded slow code
z_f = quantize(E_f(r))         fast latent, stride 16
x_f = D_f(z_f)                 residual prediction
x_hat = x_s + x_f
```

Both branches now default to **joint-channel** processing (`--channel-mode joint`),
matching the baseline's use of all 20 electrodes as the input channels of Conv1d
with `groups=1`. Each branch maps `[B, C, T]` to a joint latent `[B, D, L]` and
reconstructs all C channels. With 256 Hz, 8-second crops, and dimensions 4/4,
the slow and fast latents are `[B, 4, 32]` and `[B, 4, 128]`.

`--channel-mode independent` retains the original shared-weight, per-electrode
model: internally `[B*C, 1, T]`, with latents `[B, C, D, L]`. Dimensions are
per electrode in that mode, but total joint features in joint mode. Thus the same
4/4 dimensions yield 20 times fewer latent scalars in joint mode for 20 channels;
they are a small smoke/pilot setting, not a capacity-matched choice. Compare
achieved total rates and tune widths/dimensions using validation data.

The baseline uses widths `(32, 48, 64, 96, 128)`, four stride-2 stages (16x total),
and four bottleneck residual blocks with dilations `(1, 2, 4, 8)`. Its default
64-feature latent is `[B, 64, T/16]`. Its mirrored decoder outputs 20 channels.
The continuous baseline shares this architecture and bypasses quantization.

Both branches use a 7-tap input convolution, stride-2 convolutional stages with
residual blocks, a dilated residual bottleneck, and mirrored transposed-convolution
decoders. Hidden width defaults to 32. Strides must be powers of two, with
`slow_stride > fast_stride >= 2`; crop length must be divisible by `slow_stride`.
The implementation is offline/noncausal: symmetric padding and GroupNorm use
context within the crop. The decoder reconstructs from the two latents alone.

Training adds uniform noise in `[-0.5, 0.5]` to each code; evaluation rounds to
integers. The residual always uses the decoded noisy/rounded slow latent. Both
branches train jointly, including gradients through the residual into the slow
branch. Separate learned factorized logistic priors estimate both streams' costs:

```text
R_s = sum(bits(z_s)) / (B*C*T)
R_f = sum(bits(z_f)) / (B*C*T)
L = Huber(x, x_hat) + alpha * Huber(x, x_s) + lambda * (R_s + R_f)
```

Huber is the existing per-window RMS-normalized waveform loss; the input retains
the fixed training-derived amplitude scale. `alpha` defaults to 0.25 and is set
with `--slow-loss-weight`. The auxiliary loss encourages a useful slow prediction
and reduces the incentive to put all reconstruction in the fast stream. It does
not guarantee either stream is used or separated spectrally. Both slow and fast
decoders can synthesize high-frequency waveforms: “slow” refers to latent update
rate. No spectral target or low-pass constraint is imposed.

## Run

From the repository root with the existing Python environment and data paths:

```powershell
python -m experiments.train --architecture slow-fast --channel-mode joint `
  --manifest 'H:\EEG\FHA\Resting\preprocessed\manifests\recordings.parquet' `
  --shards-dir 'H:\EEG\FHA\Resting\preprocessed\shards' `
  --output-dir runs/temporal_slow_fast_pilot `
  --window-sec 8 --slow-stride 64 --fast-stride 16 `
  --slow-dim 4 --fast-dim 4 --temporal-width 32 `
  --slow-loss-weight 0.25 --lambda-rate 0.01 `
  --epochs 30 --early-stop-patience 5 --early-stop-min-delta 0.001 `
  --batch-size 8 --eval-start-sec 8

python -m experiments.evaluate `
  --checkpoint runs/temporal_slow_fast_pilot/best.pt --split test

python -m unittest discover -s tests -v
```

Use a fresh output directory. Architecture settings are saved in `model_config`;
the evaluator also loads legacy baseline checkpoints. Older slow-fast checkpoints
without a channel-mode field load as independent models; new checkpoints save
the mode explicitly. Validation checkpoint
selection uses the same full objective as training, including the slow auxiliary
term. Leave `--architecture` unset to run the existing baseline. The existing
loader, selected recording IDs, seed, input scale, and metrics are reused.
The generic loader does not automatically apply the spatial experiment's QC
cache; use consistently preprocessed inputs for comparisons.

## Inspect and compare

### Dual-GPU lambda queue (32 slow + 32 fast)

`experiments.sweep_temporal` schedules independent jobs, one per GPU. Each worker
sees only its assigned device through `CUDA_VISIBLE_DEVICES` and trains on its
local CUDA device. As soon as a worker finishes, it takes the next lambda job.
This is parallel hyperparameter training, not multi-GPU training of one model.

```powershell
python -m experiments.sweep_temporal `
  --manifest 'H:\EEG\FHA\Resting\preprocessed\manifests\recordings.parquet' `
  --shards-dir 'H:\EEG\FHA\Resting\preprocessed\shards' `
  --output-dir runs/temporal_32_32_sweep --gpus 0 1 `
  --slow-dim 32 --fast-dim 32 --seeds 42 `
  --lambdas 0.01 0.03 0.06 0.1 0.2 0.3 0.5 0.75 1.0 `
  --extra --epochs 30 --batch-size 16 --num-workers 0 `
  --max-train-recordings 2560 --max-val-recordings 256 --max-test-recordings 256 `
  --eval-start-sec 8 --early-stop-patience 5 --early-stop-min-delta 0.001
```

The example creates nine jobs: GPU 0 and GPU 1 take the first two, then each free
GPU takes the next. Default slow/fast strides are 64/16 and the auxiliary slow
loss weight is 0.25. Lambda multiplies the **sum** of stream rates. The lambda
grid is configurable and is a starting grid, not a validated optimal range.

Put `--dry-run` before `--extra` to inspect the full plan without creating files
or running training. Put `--resume` there to rerun the same saved sweep: completed
jobs are verified and skipped, failed/interrupted jobs restart from epoch one.
Changing the saved training plan requires a new directory. GPU slots may change
on resume (e.g. `--gpus 0`). A queue lock prevents simultaneous schedulers in one
output directory; after a hard crash, remove a stale lock only after checking
that the old scheduler and its workers have stopped. Ctrl+C stops active workers.
Other jobs continue after an individual training failure; the queue exits nonzero
if any fail. Per-job logs append across attempts.

Outputs are `sweep.json` (commands and effective training settings),
`queue_status.json` (pending/running/complete/failed/interrupted), `summary.csv`
(incremental validation/test metrics including both rates), and a log plus run
directory per job. Each run contains the normal checkpoint/config/history/results.
The GPU preflight allocates a tiny tensor on each requested device before launching.
Use `--gpus 0` on a single-GPU machine.

To retrain the baseline alongside the new architecture, add
`--architectures slow-fast baseline --latent-dim 64` before `--extra`. That creates
18 jobs for the nine-point grid and alternates architectures at each lambda.
The same seed produces the same recording selection and training-derived input
scale across architectures. Different seeds also change the split in the current
trainer; these are not repeated initializations on a fixed split.

For historical baseline comparison, reproduce its seed, manifest, recording
limits, evaluation start/crops, input preprocessing, and distortion settings.
The example uses settings from the earlier README sweep; it does not import or
verify a historical checkpoint. Older NMSE-trained baselines differ from this
Huber-trained pipeline. Compare at achieved **total rate**, not equal lambda.
With strides 64/16, 32+32 features emit fewer scalars than 64 features at stride 16
(5120 versus 8192 per 8-second crop); the dimensions alone are not matched capacity.

Evaluation adds `slow_huber`, `slow_nmse`, `fast_nmse_gain`, and separate
`slow_estimated_bits_per_channel_sample` and
`fast_estimated_bits_per_channel_sample`. Total rate includes both streams.
`fast_nmse_gain = slow_nmse - nmse`, so positive values mean the residual branch
helps; it can be negative. NMSE follows the existing evaluator's pooled energy
aggregation, while Huber averages window losses. These are not patient-level
confidence intervals.

`model.forward_components(x)` exposes both waveforms, the input residual, both
latents, and both rates. `model.decode(z_s, z_f)` returns the final, slow, and fast
waveforms without the input. Use its slow waveform for the slow-only ablation;
zeroing fast codes can still produce a nonzero decoder output. Calling
`model(x, quantize=False)` bypasses quantization and returns no rate, for continuous
capacity diagnostics; the CLI above trains the quantized model.

Start with this pilot and inspect slow-only versus full reconstructions, stream
rates, and residual gain. Compare `alpha=0` against `alpha=0.25` to see whether the
auxiliary loss helps. Then vary slow stride (32, 64, 128) at fixed fast stride 16
and sweep positive rate penalties with identical splits and evaluation windows.
Compare achieved **total** rates, not equal lambda or equal latent width. The
joint mode removes the channel-mixing mismatch with the existing baseline.
The branch widths, depth, normalization/decoder details, and auxiliary objective
still differ. Report parameter counts and use an alpha=0 ablation; matching
channel handling alone does not isolate the benefit of the hierarchy.

Rates are estimates, without entropy-coded bitstreams or headers. Zero lambda
does not train the priors. Recording IDs are the default split groups; patient
generalization requires patient IDs. Functional tests and a synthetic training
smoke run verify implementation, not an EEG reconstruction or factorization result.
