# Spatial factorization experiments: workstation run guide

This implementation trains the proposed bilinear model X_hat_j = A_j S_j and a single-stream control, then replaces spatial matrices with train-fitted dictionary representatives. The factorized post-hoc experiment keeps temporal quantized codes and decoded signals unchanged. No main clinical-data experiments have been run by the implementation task.

## Hardware and environment

The default work-PC configuration is **one independent experiment per RTX 5070 Ti**, with two jobs in parallel. Each job fits on one card; VRAM is not pooled. This is a throughput-oriented sweep and works on native Windows as well as Linux. There is no DDP/NCCL requirement.

Install a stable CUDA-enabled PyTorch build supporting your GPUs using the official selector: https://pytorch.org/get-started/locally/ . Do not copy the local development environment's CUDA wheel blindly. Then:

```powershell
python -m pip install -r requirements-spatial.txt
python benchmark_spatial.py --device cuda:0 --batch-size 64
python benchmark_spatial.py --device cuda:1 --batch-size 64
```

The benchmark exercises actual forward/backward kernels, BF16, and the optimizer. It also reports peak allocated memory and synthetic throughput. Defaults are **BF16, batch 64 per job, four loader workers per job**, pinned transfers, prefetching, persistent workers, TF32, cuDNN benchmarking, and fused AdamW on CUDA. Entropy likelihood, distortion, and bilinear synthesis use float32. FP16 with gradient scaling and FP32 are supported. BF16 is deliberately the initial work-PC setting, not a measured optimum.

Tune batch 32/64/128 on the work machine using this benchmark; use real training logs to account for disk I/O. Two jobs reading the same HDD can be I/O-bound: prefer local SSD shards or lower workers to 2 per job. Each worker limits HDF5 file handles to eight and limits per-file chunk cache. Optional `train_spatial.py --compile` is available, but is off by default for portable Windows execution. CPU smoke tests use small batches and do not change workstation defaults.

## 1. Prepare data once

Use the same prepared directory for all architectures, seeds, dictionaries, and held-out comparisons. It contains patient-disjoint split membership, exact evaluation windows, masked-event intervals, a training-only global scale, and input fingerprints.

```powershell
$manifest = 'H:\EEG\FHA\Resting\preprocessed\manifests\recordings.parquet'
$metadata = 'H:\EEG\FHA\Resting\preprocessed\EEG_Metadata.csv'
$shards = 'H:\EEG\FHA\Resting\preprocessed\shards'
$qc = 'H:\EEG\FHA\Resting\preprocessed\qc'

python prepare_spatial.py --manifest $manifest --metadata $metadata `
  --shards-dir $shards --qc-dir $qc --output-dir runs/spatial_data `
  --train-recordings 2560 --val-recordings 256 --test-recordings 256
```

Create the QC cache with the existing `build_qc_cache.py` if needed. The preparer checks cache window length and uses accepted windows and masked events. It **does not apply per-channel normalization**: interpolation occurs in source units, then every channel shares one training-derived scale. This preserves spatial amplitude ratios. Both training and evaluation exclude the first 16 seconds. Training samples accepted grid windows; final evaluation uses all accepted windows.

Missing patient mappings are excluded, conflicting mappings are rejected, and selected patient groups never overlap. The default metadata key is `Hashed_PatientURN`; override column names explicitly when needed. A separate raw-data run requires `--allow-no-qc` and omission of `--qc-dir`. Raw and QC runs must use different output/prepared directories.

For a smaller clinical pilot use 512/128/128 recordings. Preparation fails if QC leaves any split empty. Prepared directories are immutable: use a new directory to change preprocessing or split settings.

## 2. Run the four-model pilot across both GPUs

```powershell
python sweep_spatial.py --prepared runs/spatial_data --shards-dir $shards `
  --output-dir runs/spatial_pilot --gpus 0 1 --lambdas 0.02 0.1 --seeds 0 `
  --ranks 8 --epochs 60 --batch-size 64 --workers 4 --posthoc
```

This schedules a baseline and rank-8 factorized model at each lambda. The two lambda values are **starting points to calibrate on validation**, not guaranteed matched-rate points. Use `--dry-run` to print commands without launching. Each GPU slot runs training, train-only dictionary fitting, and validation post-hoc evaluation before taking another job. Logs are next to job directories; `queue_status.json` records failures without discarding successful jobs.

Add `--resume` to the exact same sweep command after interruption. Training saves atomic `last.pt` (optimizer/scaler included) and `best.pt`. Completed stages are skipped. Interrupted post-hoc stages are not partially resumed: preserve their incomplete directory under another name and rerun that stage into a fresh directory. Do not silently mix stage outputs. Resume checks training settings and prepared data identity. To change the sweep or extend epoch counts through the scheduler, use a new sweep directory; individual `train_spatial.py --resume --epochs ...` supports extending a run.

Model defaults:

| Setting | Default |
|---|---:|
| EEG crop | 20 x 2048 samples |
| Spatial rank K | 8 |
| Spatial latent dimensions | 16 |
| Spatial update stride | 16 samples / 62.5 ms |
| Temporal latent dimensions | 32 |
| Temporal encoder stride | 16 |
| Baseline latent dimensions | 64 |
| Primary loss | per-window NMSE + lambda times total estimated rate |
| Training windows per recording/epoch | 8 |
| Checkpoint monitoring | four evenly spaced validation windows per recording |
| Final evaluation | all accepted validation windows |

The spatial branch is a patch MLP with no overlap across spatial frames. The temporal branch uses the existing residual convolutional architecture. Spatial columns are unit-normalized; final synthesis is the matrix product only. No unquantized skip, smoothness loss, cluster loss, or post-product EEG decoder is present. Parameter counts are saved; these models are not exactly parameter-matched.

For an individual run:

```powershell
python train_spatial.py --prepared runs/spatial_data --shards-dir $shards `
  --output-dir runs/spatial_one --architecture factorized --rank 8 `
  --lambda-rate 0.05 --device cuda:0 --batch-size 64 --workers 4
```

Additional flags include `--spatial-stride`, `--spatial-dim`, `--temporal-dim`, `--latent-dim`, `--loss huber`, `--accumulate`, `--monitor-windows 0` (all), and `--evaluate-test`. Test evaluation is **off by default**. Training and split seeds are separate. Repeating a run on different hardware is not guaranteed bit-identical with cuDNN autotuning.

## 3. Dictionary experiments independently

```powershell
python posthoc_spatial.py fit --checkpoint runs/spatial_one/best.pt `
  --prepared runs/spatial_data --shards-dir $shards --output-dir runs/spatial_one/dictionary `
  --sizes 1 4 8 16 32 64 --holds 1 2 4 8 16 32 --device cuda:0

python posthoc_spatial.py evaluate --bundle runs/spatial_one/dictionary/dictionary.pt `
  --prepared runs/spatial_data --shards-dir $shards --output-dir runs/spatial_one/posthoc_val `
  --split val --device cuda:0
```

Fitting uses up to 65,536 spatial examples, approximately balanced across patients/recordings, sampled from four evenly spaced training windows per recording. MiniBatchKMeans initializes representatives, which are replaced by actual decoded training matrices. The effective dictionary size is saved if centers collapse. Dictionaries are checkpoint-specific; checkpoint hashes prevent applying one to another model.

Two assignment rules are distinct:

- `m16_nearest`: nearest normalized spatial matrix, original temporal codes unchanged.
- `m16_hold4`: one representative per four spatial frames (250 ms), selected by block reconstruction error against the original input with fixed temporal coefficients. This rule is used consistently at all hold lengths, including `hold1`.

The encoder may see the original input for assignment; the receiver needs only the dictionary, index, and temporal stream. No hidden rotation, alignment matrix, or refitted temporal coefficients are supplied. Each 8-second crop is an independently coded sequence; transitions reset and boundary runs are marked censored. No smoothing is applied to labels.

Rate models include fixed-length indices and a smoothed first-order Markov model fitted on training assignments only. Every sequence start is charged. Both are **estimated** rates; no arithmetic-coded files or headers are implemented. Dictionary storage is reported separately and amortized across evaluated samples. The dictionary is a global decoder parameter, not free recording-specific information.

Baseline checkpoints use the same post-hoc interface for generic latent-vector dictionary replacement. Since their latent vector width differs from `20*K`, equal M does not imply equal dictionary storage: compare the reported dictionary-bit budgets or rerun appropriate sizes before attributing gains specifically to spatial factorization.

## 4. Inspect validation, then lock test choices

```powershell
python summarize_spatial.py --root runs/spatial_pilot --output-dir runs/spatial_pilot_summary
```

Outputs include total rate-distortion, hold-interval, and dictionary-size band-error figures; CSVs retain each model/split separately. `validation_candidates.json` lists the smallest validation nearest-matrix dictionary within proposed tolerances (+0.01 waveform NMSE, +0.03 per-band NMSE). This is an engineering candidate, not clinical validation. Hold intervals are inspected separately. Test data never select candidates.

After choosing cases on validation:

```powershell
python posthoc_spatial.py evaluate --bundle runs/spatial_one/dictionary/dictionary.pt `
  --prepared runs/spatial_data --shards-dir $shards --output-dir runs/spatial_one/posthoc_test `
  --split test --cases m16_nearest m16_hold4 --device cuda:0
```

When copying runs to another machine, override `--checkpoint` if the saved absolute checkpoint path is no longer valid. Copy the prepared directory unchanged and pass the new `--shards-dir`; raw path strings in data.json are provenance, not runtime dependencies.

Every evaluated case saves recording/patient metrics, paired patient bootstrap intervals relative to the unreplaced checkpoint, assignments, censored run lengths, occupancy, transition counts, and shuffled-label transition control statistics. Add `--window-metrics` to stream detailed window metrics to disk; this can produce large files across 42 cases. Metrics accumulate by recording to keep memory bounded. Training's final evaluation also saves window metrics. Band metrics use filled windows and additionally report untouched-window sensitivity metrics. Covariance/correlation/peak/line-length metrics describe the filled signals; waveform error excludes interpolated samples. Rates count every transmitted sample/code regardless of masks. Uncertainty does not include training seed variation; repeat selected fits with `--seeds 0 24 48` and inspect separately.

## Checks and boundaries

```powershell
python -m unittest discover -s tests -p "test_spatial*.py" -v
python smoke_spatial.py
python smoke_spatial.py --workers 1 --qc --verify-resume
```

Smoke tests generate small synthetic HDF5 recordings, prepare patient splits, train both architectures on CPU, fit dictionaries, evaluate replacements, and build figures. They validate execution and invariants, not reconstruction quality or convergence.

The implemented primary experiment clusters ordered spatial matrices in the learned codec coordinates. It does not solve factor-rotation ambiguity or recover clinical microstates. The optional subspace diagnostic is separate from the compression result. Actual entropy coding, temporal re-encoding after basis alignment, clinically labeled transient detection, and fully parameter-matched architecture sweeps are follow-ups.

```powershell
python diagnose_spatial.py --checkpoint runs/spatial_one/best.pt `
  --prepared runs/spatial_data --shards-dir $shards --output-dir runs/spatial_one/subspaces `
  --split val --device cuda:0
```

This compares ordered-matrix drift with rotation-invariant projector drift at different lags, using SVD and an explicit rank tolerance. It is descriptive and does not assign a compressed bitstream. Each post-hoc case also saves `worst_waveform.npz` containing the original input, original reconstruction, replacement reconstruction, mask, and sample rate in original signal units for failure inspection.

References: the proposed design is in `notes/spatial_factorization_experiment.md`; mixed precision follows https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html .
