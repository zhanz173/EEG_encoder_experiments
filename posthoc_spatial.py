"""Fit train-only representative dictionaries and evaluate frozen-code replacement.

fit: freeze model, fit dictionaries and coding distributions on TRAIN only.
evaluate: apply frozen bundle to validation (default) or explicitly selected test.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from spatial_data import SpatialDataset, loader, write_json, fingerprint
from spatial_model import SpatialFactorAE
from spatial_metrics import Metrics, sequence_stats, paired_bootstrap
from train_spatial import load_checkpoint, amp_context, runtime


def nearest(values, dictionary):
    # Both operands flatten in exactly the same order; no hidden basis alignment.
    v = values.flatten(-2) if values.ndim == 4 else values
    d = dictionary.flatten(1)
    return (v.square().sum(-1, keepdim=True) + d.square().sum(1)
            - 2*v @ d.T).argmin(-1)


def held_assignments(x, s, dictionary, stride, hold, mask=None):
    """Exact blockwise distortion minimizer using fixed S and only transmitted indices."""
    b, c, t = x.shape
    j, k = t//stride, s.shape[1]
    if j % hold:
        raise ValueError("Hold length must divide the number of spatial frames")
    xb = x.reshape(b, c, j, stride).permute(0, 2, 1, 3)
    sb = s.reshape(b, k, j, stride).permute(0, 2, 1, 3)
    gram = sb @ sb.transpose(-1, -2)
    cross = xb @ sb.transpose(-1, -2)
    dd = dictionary.transpose(1, 2) @ dictionary
    # Input energy is identical across candidates, so it cancels in argmin.
    costs = torch.einsum("mkl,bjkl->bjm", dd, gram) - 2*torch.einsum("mck,bjck->bjm", dictionary, cross)
    if mask is not None:
        points = mask.nonzero()
        # Subtract each masked sample's candidate-dependent contribution, in
        # bounded chunks. Its input-only square is absent from all costs already.
        for chunk in points.split(4096):
            if not len(chunk):
                continue
            n, channel, sample = chunk.unbind(1)
            coeff = s[n, :, sample]
            pred = torch.einsum("nmk,nk->nm", dictionary[:, channel, :].permute(1, 0, 2), coeff)
            correction = pred.square()-2*pred*x[n, channel, sample, None]
            flat_indices = n*j + sample//stride
            costs.reshape(-1, len(dictionary)).index_add_(0, flat_indices, -correction)
    return costs.reshape(b, j//hold, hold, len(dictionary)).sum(2).argmin(-1)


def assignments(x, o, dictionary, architecture, stride, rule, hold, mask=None):
    if architecture == "baseline":
        return nearest(o["zt"].transpose(1, 2), dictionary)
    if rule == "nearest":
        return nearest(o["a"], dictionary)
    return held_assignments(x, o["s"], dictionary, stride, hold, mask)


def reconstruct(model, o, labels, dictionary, architecture, hold):
    chosen = dictionary[labels.repeat_interleave(hold, 1)]
    if architecture == "baseline":
        return model.decoder(chosen.transpose(1, 2)).float()
    return SpatialFactorAE.combine(chosen, o["s"])


def counts(m):
    return dict(initial=np.zeros(m), occupancy=np.zeros(m), transition=np.zeros((m, m)))


def coding(count, smoothing):
    initial = count["initial"]+smoothing
    trans = count["transition"]+smoothing
    return dict(initial=initial/initial.sum(), transition=trans/trans.sum(1, keepdims=True))


def code_bits(labels, probabilities):
    initial = torch.as_tensor(probabilities["initial"], device=labels.device, dtype=torch.float64)
    trans = torch.as_tensor(probabilities["transition"], device=labels.device, dtype=torch.float64)
    return (-torch.log2(initial[labels[:, 0]]) - torch.log2(trans[labels[:, :-1], labels[:, 1:]]).sum(1)).float()


def cases(dictionaries, architecture, holds):
    for requested, dictionary in dictionaries.items():
        yield f"m{requested}_nearest", requested, "nearest", 1
        if architecture == "factorized":
            for h in holds:
                yield f"m{requested}_hold{h}", requested, "distortion", h


def representative_dictionary(samples, m, seed):
    flat = samples.reshape(len(samples), -1)
    if len(flat) < m:
        raise ValueError(f"Need at least {m} training examples")
    km = MiniBatchKMeans(n_clusters=m, random_state=seed, n_init=3, batch_size=min(4096, len(flat)),
                        max_iter=150, reassignment_ratio=0)
    km.fit(flat)
    center = km.cluster_centers_
    best, idx = np.full(m, np.inf), np.zeros(m, int)
    for start in range(0, len(flat), 2048):
        chunk = flat[start:start+2048]
        distance = ((chunk*chunk).sum(1)[:, None] + (center*center).sum(1)[None, :] - 2*chunk @ center.T)
        local = distance.argmin(0)
        score = distance[local, np.arange(m)]
        improve = score < best
        best[improve], idx[improve] = score[improve], local[improve]+start
    # Collapsed representations may have fewer distinct representatives.
    # Store and charge the effective dictionary size explicitly.
    reps = samples[np.unique(idx)]
    flat_unique, unique_idx = np.unique(reps.reshape(len(reps), -1), axis=0, return_index=True)
    return reps[np.sort(unique_idx)].astype(np.float32)


@torch.no_grad()
def fit(args):
    device = torch.device(args.device)
    runtime(device, args.precision, args.threads)
    model, ck = load_checkpoint(args.checkpoint, device)
    if fingerprint(Path(args.prepared)/"data.json") != ck["data_hash"]:
        raise ValueError("Prepared data differs from checkpoint")
    cfg, stride = ck["model_config"], ck["model_config"]["spatial_stride"]
    arch = cfg["architecture"]
    if arch == "baseline":
        stride = 16
    ds = SpatialDataset(args.prepared, args.shards_dir, "train", max_windows=args.fit_windows)
    dl = loader(ds, args.batch_size, args.workers)
    length = ds.config["window_samples"]
    if any((length//stride) % h for h in args.holds) or min(args.holds) < 1:
        raise ValueError("Every hold length must divide crop spatial-frame count")
    rng = np.random.default_rng(args.seed)
    records_per_patient = ds.records.groupby("patient_id").size().to_dict()
    per_patient = args.max_samples/max(1, len(records_per_patient))
    window_counts = {r["sha256_id"]: len(ss) for r, ss in zip(ds.rows, ds.starts)}
    examples = []
    provenance = []
    for batch in dl:
        x = batch["x"].to(device)/ck["scale"]
        with amp_context(device, args.precision):
            o = model(x)
        values = o["a"] if arch == "factorized" else o["zt"].transpose(1, 2)
        values = values.cpu().numpy()
        for i, value in enumerate(values):
            rec, patient = batch["sha256_id"][i], batch["patient_id"][i]
            quota = max(1, int(per_patient/records_per_patient[patient]/window_counts[rec]))
            pos = np.sort(rng.choice(len(value), min(quota, len(value)), replace=False))
            examples.extend(value[pos])
            provenance.extend(dict(sha256_id=rec, patient_id=patient, start=int(batch["start"][i]), frame=int(j)) for j in pos)
    if len(examples) > args.max_samples:
        chosen = rng.choice(len(examples), args.max_samples, replace=False)
        examples = [examples[i] for i in chosen]
        provenance = [provenance[i] for i in chosen]
    samples = np.stack(examples).astype(np.float32)
    dictionaries = {}
    for m in args.sizes:
        dictionaries[str(m)] = torch.from_numpy(representative_dictionary(samples, m, args.seed)).to(device)
        print(f"Dictionary requested={m}, effective={len(dictionaries[str(m)])}", flush=True)
    all_cases = list(cases(dictionaries, arch, args.holds))
    count = {key: counts(len(dictionaries[m])) for key, m, _, _ in all_cases}
    for batch in dl:
        x = batch["x"].to(device)/ck["scale"]
        with amp_context(device, args.precision):
            o = model(x)
        for key, m, rule, hold in all_cases:
            labels = assignments(x, o, dictionaries[m], arch, stride, rule, hold, batch["mask"].to(device))
            sequence_stats(labels.cpu().numpy(), batch, None, count[key], [])
    bundle = dict(checkpoint_sha256=fingerprint(args.checkpoint), checkpoint=str(Path(args.checkpoint).resolve()),
                  data_hash=ck["data_hash"], model_config=cfg, precision=args.precision,
                  dictionaries={k: v.cpu() for k, v in dictionaries.items()}, holds=args.holds,
                  probabilities={k: coding(v, args.smoothing) for k, v in count.items()},
                  training_counts=count, fit_args=vars(args), examples=len(samples),
                  rate_note="Estimated index bits using train-fitted Markov probabilities; crop starts reset")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out/"dictionary.pt")
    pd.DataFrame(provenance).to_csv(out/"fit_samples.csv", index=False)
    write_json(out/"fit_config.json", dict(vars(args), effective_sizes={k: len(v) for k, v in dictionaries.items()},
                                          checkpoint_sha256=bundle["checkpoint_sha256"], examples=len(samples)))
    ds.close()
    print(f"Saved {out/'dictionary.pt'}", flush=True)


@torch.no_grad()
def evaluate(args):
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    checkpoint = args.checkpoint or bundle["checkpoint"]
    if fingerprint(checkpoint) != bundle["checkpoint_sha256"]:
        raise ValueError("Dictionary belongs to a different checkpoint")
    if fingerprint(Path(args.prepared)/"data.json") != bundle["data_hash"]:
        raise ValueError("Prepared data differs from dictionary")
    precision = bundle["precision"]
    device = torch.device(args.device)
    runtime(device, precision, args.threads)
    model, ck = load_checkpoint(checkpoint, device)
    arch, stride = ck["model_config"]["architecture"], ck["model_config"]["spatial_stride"]
    if arch == "baseline":
        stride = 16
    dictionaries = {k: v.to(device) for k, v in bundle["dictionaries"].items()}
    all_cases = list(cases(dictionaries, arch, bundle["holds"]))
    if args.cases:
        allowed = set(args.cases)
        if not allowed <= {c[0] for c in all_cases}:
            raise ValueError("Unknown requested case")
        all_cases = [c for c in all_cases if c[0] in allowed]
    ds = SpatialDataset(args.prepared, args.shards_dir, args.split, untouched_only=args.untouched_only)
    dl = loader(ds, args.batch_size, args.workers)
    meters = {key: Metrics(ds.config["sfreq"], stride,
                           Path(args.output_dir)/key/"windows.csv" if args.window_metrics else None)
              for key, _, _, _ in all_cases}
    original = Metrics(ds.config["sfreq"], stride)
    count = {key: counts(len(dictionaries[m])) for key, m, _, _ in all_cases}
    shuffled = {key: counts(len(dictionaries[m])) for key, m, _, _ in all_cases}
    runs = {key: [] for key, _, _, _ in all_cases}
    assignment_rows = {key: [] for key, _, _, _ in all_cases}
    fixed_bits = {key: 0 for key, _, _, _ in all_cases}
    examples = {}
    rng = np.random.default_rng(42)
    for batch in dl:
        x = batch["x"].to(device)/ck["scale"]
        with amp_context(device, precision):
            o = model(x)
        original.add(batch, x, o["y"], o["bits_spatial"], o["bits_temporal"])
        for key, m, rule, hold in all_cases:
            d = dictionaries[m]
            labels = assignments(x, o, d, arch, stride, rule, hold, batch["mask"].to(device))
            # Baseline generic vector replacement changes its only latent stream;
            # factorized replacement preserves temporal codes byte-for-byte.
            if arch == "baseline":
                with amp_context(device, precision):
                    y = reconstruct(model, o, labels, d, arch, hold)
                bt = torch.zeros_like(o["bits_temporal"])
            else:
                y = reconstruct(model, o, labels, d, arch, hold)
                bt = o["bits_temporal"]
            bs = code_bits(labels, bundle["probabilities"][key])
            meters[key].add(batch, x, y, bs, bt)
            # Keep only one worst waveform window per case, bounded memory.
            keep = ~batch["mask"].to(device)
            errors = ((x-y).square()*keep).sum((1, 2))/(x.square()*keep).sum((1, 2)).clamp_min(1e-12)
            worst = int(errors.argmax())
            score = float(errors[worst])
            if key not in examples or score > examples[key]["nmse"]:
                examples[key] = dict(nmse=score, x=x[worst].cpu().numpy()*ck["scale"],
                                     original=o["y"][worst].cpu().numpy()*ck["scale"],
                                     replacement=y[worst].cpu().numpy()*ck["scale"],
                                     mask=batch["mask"][worst].numpy(), sfreq=ds.config["sfreq"],
                                     sha256_id=batch["sha256_id"][worst], start=int(batch["start"][worst]))
            fixed_bits[key] += labels.numel()*math.ceil(math.log2(len(d)))
            cpu = labels.cpu().numpy()
            sequence_stats(cpu, batch, None, count[key], runs[key])
            permuted = np.stack([rng.permutation(row) for row in cpu])
            sequence_stats(permuted, batch, None, shuffled[key], [])
            for i, row in enumerate(cpu):
                assignment_rows[key].append(dict(sha256_id=batch["sha256_id"][i], patient_id=batch["patient_id"][i],
                                                start=int(batch["start"][i]), labels=" ".join(map(str, row))))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = original.finish(out/"original")
    before = pd.read_csv(out/"original/patients.csv", dtype={"patient_id": str})
    summaries = []
    channel_samples = original.channel_samples
    for key, m, rule, hold in all_cases:
        target = out/key
        summary = meters[key].finish(target)
        after = pd.read_csv(target/"patients.csv", dtype={"patient_id": str})
        pd.DataFrame(paired_bootstrap(before, after, args.bootstrap)).to_csv(target/"paired_bootstrap.csv", index=False)
        d = dictionaries[m]
        summary.update(case=key, requested_m=int(m), effective_m=len(d), rule=rule, hold=hold,
                       update_ms=1000*stride*hold/ds.config["sfreq"], dictionary_bits=d.numel()*32,
                       fixed_index_rate_pooled=fixed_bits[key]/channel_samples,
                       dictionary_rate_amortized=d.numel()*32/channel_samples,
                       delta_wave_nmse=summary["wave_nmse"]-reference["wave_nmse"])
        for name in ("delta", "theta", "alpha", "low_beta", "high_beta", "beta"):
            summary[f"delta_{name}_nmse"] = summary[f"{name}_nmse"]-reference[f"{name}_nmse"]
        summary["passes_proposed_tolerance"] = (summary["delta_wave_nmse"] <= .01 and
                  all(summary[f"delta_{b}_nmse"] <= .03 for b in ("delta", "theta", "alpha", "low_beta", "high_beta")))
        if reference["rate_spatial"] > 0:
            summary["spatial_rate_reduction"] = 1-summary["rate_spatial"]/reference["rate_spatial"]
        run_df = pd.DataFrame(runs[key])
        run_df["duration_ms"] = run_df.frames*summary["update_ms"]
        run_df.to_csv(target/"runs.csv", index=False)
        pd.DataFrame(assignment_rows[key]).to_csv(target/"assignments.csv", index=False)
        np.savez_compressed(target/"worst_waveform.npz", **examples[key])
        transitions = count[key]["transition"]
        summary["switch_fraction"] = float((transitions.sum()-np.trace(transitions))/max(transitions.sum(), 1))
        trans_shuffled = shuffled[key]["transition"]
        summary["shuffled_switch_fraction"] = float((trans_shuffled.sum()-np.trace(trans_shuffled))/max(trans_shuffled.sum(), 1))
        pd.DataFrame(transitions).to_csv(target/"transitions.csv", index=False)
        pd.DataFrame({"occupancy": count[key]["occupancy"]}).to_csv(target/"occupancy.csv", index=False)
        write_json(target/"summary.json", summary)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out/"comparison.csv", index=False)
    write_json(out/"evaluation_config.json", dict(vars(args), checkpoint_sha256=bundle["checkpoint_sha256"],
                                                 dictionary_sha256=fingerprint(args.bundle), precision=precision,
                                                 model_config=ck["model_config"], training_seed=ck["args"]["seed"],
                                                 lambda_rate=ck["args"]["lambda_rate"], data_hash=ck["data_hash"]))
    ds.close()
    print(f"Saved {out/'comparison.csv'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--prepared", required=True)
        s.add_argument("--shards-dir", required=True)
        s.add_argument("--output-dir", required=True)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--checkpoint", required=name == "fit")
        if name == "fit":
            s.add_argument("--sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
            s.add_argument("--holds", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
            s.add_argument("--max-samples", type=int, default=65536)
            s.add_argument("--fit-windows", type=int, default=4)
            s.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
            s.add_argument("--smoothing", type=float, default=.5)
            s.add_argument("--seed", type=int, default=42)
        else:
            s.add_argument("--bundle", required=True)
            s.add_argument("--split", choices=["val", "test"], default="val")
            s.add_argument("--cases", nargs="+", help="Restrict test to validation-selected cases")
            s.add_argument("--untouched-only", action="store_true")
            s.add_argument("--window-metrics", action="store_true", help="Stream detailed window metrics to disk; large outputs")
            s.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error("Use a new output directory to preserve previous results")
    if args.batch_size < 1 or min(args.workers, args.threads) < 0:
        p.error("Invalid batch/worker settings")
    if args.command == "fit":
        if min(args.sizes) < 1 or args.max_samples < max(args.sizes) or args.smoothing <= 0 or args.fit_windows < 1:
            p.error("Positive sizes/smoothing/windows and sufficient samples required")
        fit(args)
    else:
        if args.bootstrap < 1:
            p.error("bootstrap must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
