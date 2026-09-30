# Experiment proposal: spatial factors, recurrence, and coding cost

Status: proposed protocol for discussion; no models trained or existing results changed.

## 1. Question and separate hypotheses

Can rapidly updated spatial factors in an EEG autoencoder be replaced by a small shared dictionary without materially degrading reconstruction, and does their assignment sequence require few bits?

Test three separate claims:

1. Spatial redundancy: a small number K of simultaneously active spatial patterns is sufficient for useful reconstruction.
2. Recurrence: many spatial-factor matrices can be replaced by a small dictionary of recurring configurations.
3. Persistence: spatial configurations can be updated less frequently without materially degrading reconstruction.

Recurrence does not imply persistence. A two-state sequence can switch at every frame. A small spatial fraction of the original model's rate is not sufficient evidence either: information can migrate into the temporal stream. The decisive comparison replaces spatial information while keeping the temporal codes fixed, and measures total rate and distortion.

This is a representation/compression experiment. Clusters are not presumed to be clinical microstates or unique physiological sources.

## 2. Data and comparison conditions

- Use the local 20-channel, 256 Hz Fraser Health EEG. Preserve electrode order and document the available reference and units. Do not download data.
- Build a patient-disjoint split from the verified unique ScanID -> Hashed_PatientURN matches in the local metadata. Earlier inspection found 35,946 of 40,716 manifest rows matched uniquely, with no conflicting mappings. Revalidate this join when implementing. Keep unmatched recordings outside the primary experiment until their grouping is resolved.
- First pilot: approximately 512 training, 128 validation, and 128 test recordings, selected after patient grouping. Expansion target: approximately 2,560/256/256, subject to available patients and compute. Freeze patient and recording lists independently of training seeds.
- Train with random 8-second windows. Evaluate all eligible non-overlapping 8-second windows after the first 16 seconds, rather than only four windows per recording. Record exact accepted starts and rejection reasons.
- Freeze one QC policy for all models. Use recording-specific robust statistics for artifact detection if needed, but preserve relative electrode amplitudes in model inputs: use one training-derived global scale in the primary experiment. The current CleanEEGWindowDataset normalizes per channel and is not a drop-in primary loader for this protocol. Add an explicit separation between QC and normalization when implementing.
- Preserve original units for final reconstruction metrics. Exclude interpolated samples from waveform loss and report an untouched-window sensitivity analysis for spectral metrics; pointwise masks alone do not remove interpolation effects from spectra.
- Keep montage, loss, window selection, and aggregation identical across the new baseline and factorized models. The old report is motivation, not a directly comparable control.

## 3. Architecture: nonlinear encoders, constrained bilinear reconstruction

Input X has shape [B, 20, 2048]. Start with spatial updates every 16 samples (62.5 ms): J = 128 spatial frames per crop. This is the initial computational resolution, not a claim about physiological timescales. A finer 8-sample update is a targeted follow-up, not a required first sweep.

Two branches encode X:

- Spatial encoder E_s -> z_s [B, d_s, J], initially d_s = 16.
- Temporal encoder E_t -> z_t [B, d_t, 128], initially d_t = 32, using the existing temporal convolutional family with total stride 16.

Quantize both streams and decode:

- A_j = D_s(Q(z_s,j)), shape [20, K] for each spatial frame.
- S = D_t(Q(z_t)), shape [K, 2048].
- For samples in frame j: X_hat_j = A_j S_j.

Start with K = 8; follow with K = 4 and 12 only if the pilot is usable. K counts concurrent spatial patterns; dictionary size M later counts whole spatial configurations. They are different quantities.

For the first spatial encoder, use a small shared MLP on each non-overlapping [20, 16] patch (flatten 320 -> 128 -> d_s). The spatial decoder is a shared MLP d_s -> 128 -> 20K. This permits rapid spatial changes without temporal smoothing or overlapping spatial-encoder windows. It is deliberately a simple diagnostic architecture. The temporal branch can have a longer receptive field, recorded explicitly.

Normalize each decoded column of A to unit L2 norm, with a numerical floor. Temporal coefficients carry amplitude. Monitor near-zero columns and singular values of A. Normalization removes a scale degree of freedom but does not ensure unique orientation or independence.

The final operation is the matrix product. Do not add an electrode-specific waveform decoder, an unquantized skip connection, or a residual waveform branch after it: those would provide alternative routes around the factorization.

Use piecewise-constant A initially. Report reconstruction error around frame boundaries. Treat interpolation or blending of A as a later architectural variant, because it imposes additional smoothness.

## 4. Training and rate

Train end-to-end with:

L = D_waveform(X, X_hat) + lambda * (R_s + R_t).

Use per-window waveform NMSE in the primary pilot to retain the original study's distortion objective. Add an explicit loss option to the current Huber-only training entry point. Huber can be a separate sensitivity experiment, not a simultaneous change in the primary comparison.

Use scalar rounding at evaluation and additive uniform quantization noise during training, as in the existing model. Start with independent factorized logistic priors for the two streams. Count rate as summed negative log2 discrete-bin probability divided by original channel-samples. Report estimated bits/channel-sample, total bits/second, and R_s and R_t separately.

No temporal smoothness penalty, cluster loss, minimum dwell time, or unequal branch rate penalties in the primary experiment. Otherwise persistence or low spatial rate could be imposed by the objective.

Choose two lambda values on validation that produce separated, overlapping achieved-rate ranges for the baseline and factorized model. Do not assume old lambda values reproduce old rates. If comparison ranges do not overlap, add a rate point rather than extrapolate.

Track train and validation objectives, branch rates, band fidelity, and conditioning of A. Checkpoint selection uses the declared validation waveform-plus-rate objective. Freeze split seed, model seed, and crop sampling policy separately. Repeat retained configurations with three independent training seeds before interpreting stability.

## 5. Post-hoc clustering: primary operational experiment

Freeze each trained model. Extract spatial matrices decoded from quantized codes and the corresponding decoded temporal coefficients.

Fit dictionaries using training recordings only, with balanced sampling per patient/recording so long recordings do not dominate. Try M = 1, 4, 8, 16, 32, 64. Bound the fitting sample and record its IDs and frame positions. Repeat clustering initialization to check sensitivity; select using validation only.

Primary distance: Frobenius distance between normalized, ordered A matrices in the coordinates learned by that model. For a practical bounded fitting procedure, run k-means on flattened matrices, then choose the nearest actual training A as each representative and reassign to the resulting dictionary. This is a k-means-initialized representative dictionary, not a claim of globally optimized k-medoids.

For every held-out frame, select its nearest representative A_c and reconstruct:

X_hat_cluster,j = A_c S_j.

Keep temporal quantized codes and S_j exactly unchanged. Do not retrain the autoencoder, refit temporal coefficients, or apply a per-frame alignment matrix in this primary test. The encoder can compute A_j to choose c, but the receiver needs only c, the shared dictionary, and the original temporal stream. The spatial latent itself is no longer transmitted.

Measure the change in distortion relative to that same checkpoint before replacement. This measures how replaceable its spatial stream really is.

## 6. Factor ambiguity: keep the diagnosis separate from the codec

A S = (A G)(G^-1 S) for invertible G. Ordered-matrix clustering therefore tests the learned codec coordinates; it is not invariant to equivalent factor bases. Compare models across seeds by distortion, bitrate, and subspaces, not raw cluster IDs.

As a secondary diagnostic, orthonormalize each full-rank A to Q and compare projectors P = Q Q^T using squared distance ||P_i-P_j||_F^2. Record deficient rank and numerical tolerance. This distance ignores rotations, permutations, and signs within a subspace.

If subspaces cluster well but ordered matrices do not, the spatial subspace may recur while the codec's coordinate basis drifts. That is evidence for a basis-consistency problem, not immediate evidence for a cheap deployable codec.

Any follow-up that rotates or re-estimates coefficients for each representative must produce newly quantized temporal codes and recount their rate. A rotation or least-squares refit derived from unavailable original factors cannot be granted to the receiver for free. Report that follow-up separately from frozen-temporal replacement.

## 7. Persistence and update-rate experiment

Analyze contiguous assignments without smoothing: occupancy, run-length distribution, transition matrix, and transition frequency. Reset at missing/rejected windows and recording boundaries. Treat crop-edge runs as censored for dwell-time summaries; do not join disjoint crops.

Next, hold one representative for 1, 2, 4, 8, 16, or 32 spatial frames, corresponding to 62.5 ms through 2 s. In each block choose the representative minimizing summed squared reconstruction error using the original input and fixed decoded S. The encoder may use X for this choice; the decoder uses only the transmitted representative index. Apply the same rule at every interval, including one frame, so this curve has a consistent assignment procedure. Keep nearest-A clustering results as a distinct experiment.

Keep all temporal codes fixed. Plot distortion against achieved index rate and update interval. This intervention is the stronger test of persistence: a long observed cluster dwell can arise from coarse clustering, whereas successful infrequent replacement demonstrates tolerance to fewer spatial updates.

Use time-shuffled assignment sequences, within contiguous segments, as a control for transition statistics only. Shuffling preserves occupancy but removes ordering. It is not a waveform reconstruction control and does not prove physiological states. An optional raw-topography comparison can assess whether similar persistence already exists in the input.

## 8. Count the post-hoc rate honestly

For a shared training dictionary:

R_post = (bits(temporal codes) + bits(spatial indices) + stream overhead) / (C T).

Report two index-rate estimates:

1. Fixed-length: ceil(log2 M) per assignment (zero for M = 1).
2. Sequence model: a smoothed initial distribution and first-order transition probabilities fitted on training assignments, frozen before validation/test. Sum -log2 probabilities on held-out assignments, including sequence starts. Fit separately for each dictionary and update/assignment rule.

Do not use held-out empirical entropy as an achievable coder rate. Negative log probabilities remain estimated rate until an actual entropy coder produces a bitstream. Report transitions/run-length coding only with a defined model and counted lengths.

Treat a training-global dictionary as shared decoder parameters, like model weights. Also disclose its storage and an amortized deployment cost: at float32, 32*M*20*K bits. At K = 8 and M = 16, this is 81,920 bits (10,240 bytes). Any recording-specific dictionary must be charged to that recording. Do not compare an actual coded post-hoc stream against estimated baseline rate without labeling the difference.

Illustration only: at 16 spatial assignments/s and M = 16, fixed-length indices cost 64 bits/s, or 0.0125 bits/channel-sample for 20 channels at 256 Hz, excluding the dictionary and temporal stream. This is a rate calculation, not a predicted reconstruction result.

For fairness, also apply post-hoc vector clustering to the single-stream baseline's latent frames using a comparable dictionary-storage budget. If generic vector quantization gives the same gain, improvement cannot be attributed specifically to spatial factorization. A temporally conditioned entropy model for both baseline and original streams is a later coding-model control if transition coding drives the apparent advantage.

## 9. Metrics and proposed decision rules

Primary aggregation: compute recording metrics, average recordings within patients, then average patients. Also report pooled waveform NMSE as a secondary measure. Bootstrap patients with paired model results; report individual training-seed results and variability separately rather than treating frames as independent observations.

Endpoints:

- Waveform NMSE and channel-wise reconstruction errors.
- Band NMSE and power ratio for delta, theta, alpha, low beta, and high beta; whole beta as a summary. Preserve the existing phase-sensitive band definitions.
- Spatial correlation error and spatial covariance error, because correlation alone discards amplitude differences.
- Peak and line-length error, with explicit waveform examples of failures.
- Tail distortion (e.g. patient-level 90th percentile), R_s/R_t, dictionary utilization, assignment rate, run lengths, and frame-boundary artifacts.

Proposed engineering tolerances, to freeze before test inspection: no more than +0.01 absolute mean-patient waveform NMSE and +0.03 absolute band NMSE in each prespecified band relative to the unreplaced factorized checkpoint. These are design choices, not clinical equivalence margins. Select the smallest validation dictionary satisfying them; report test differences and paired intervals even if the selected dictionary fails on test. Do not retune using test.

Evidence supporting recurrence: M <= 16 passes the tolerances and cuts the spatial stream's estimated rate by at least 50%, with the total rate also reported. Neither 16 nor 50% is a biological threshold; both are proposed practical targets.

Evidence supporting persistence: a longer hold interval, initially 250 ms or more, passes the same fidelity tolerances relative to the original checkpoint and reduces achieved spatial rate. If 62.5 ms works but longer holds fail, recurrence may be supported without slow change.

Evidence for a useful codec: the complete post-hoc system improves the total rate-distortion tradeoff relative to the new single-stream control, including the generic clustering control. Spatial replaceability can be scientifically interesting even if this final comparison fails.

Failure interpretations:

- Small dictionaries work, but labels switch rapidly: recurrence without persistence.
- Only subspace clustering works: basis ambiguity or drift; a codec redesign may be needed.
- Spatial rate falls but temporal rate is disproportionately large: information allocation may be inefficient; inspect total rate and baseline comparison.
- Low waveform error but beta/transient loss: broad waveform preservation is insufficient for the stated study.
- Factorized models reconstruct poorly before clustering: architecture/optimization failure; do not interpret it as disproving EEG spatial redundancy.

## 10. Execution order and outputs

1. Implement the minimal factorized model, rate accounting, patient join, and fixed evaluation windows. Verify shapes, finite gradients, quantized-only decoder inputs, exact rate sums, and column normalization.
2. Use synthetic rank-K data with known fixed, switching, and rapidly varying spatial subspaces to verify the reconstruction and clustering pipeline. Compare subspaces, not necessarily individual recovered factors. A rank-one per-sample construction is a degeneracy check rather than evidence of compression.
3. Train K = 8 and a new single-stream baseline at two validation-chosen rates, one seed each: four initial fits. Match parameter counts approximately or disclose the difference.
4. Run dictionary and hold-interval sweeps on those frozen models. These require no model retraining.
5. If promising, repeat the selected operating point with three total training seeds and expand K to 4 and 12. Add finer spatial updates only when 62.5 ms resolution appears limiting.
6. Freeze dictionary size, holding interval, metrics, and thresholds on validation; evaluate the untouched test set once for the chosen primary comparison. Label additional test curves as exploratory.

Deliver a total rate-distortion plot, dictionary-size/reconstruction plot, hold-interval/reconstruction plot, spatial/temporal rate breakdown, occupancy and transition summaries, and representative success/failure waveforms. Save model configs, split membership, training histories, dictionaries, assignment rules, coding probabilities, and patient-level metrics.

No slow-fast temporal encoder is needed for this stage. Its purpose is to determine whether explicit spatial factorization and post-hoc dictionary replacement are worth pursuing first.

## References and limits of attribution

- Li and Mandt, Disentangled Sequential Autoencoder (2018): https://arxiv.org/abs/1803.02991 — precedent for sequence-level and dynamic latent separation; not evidence that this proposed EEG factorization will work.
- Locatello et al., Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations (2019): https://proceedings.mlr.press/v97/locatello19a.html — motivation to avoid assuming latent labels alone establish disentanglement.
- Local motivating results: notes/report.md. The protocol above is a new proposal; all dimensions, pilot sizes, and decision tolerances are proposed defaults.
