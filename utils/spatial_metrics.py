"""Recording/patient aggregation and reproducible evaluation for spatial codecs."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.band_metrics import BANDS, _window_components
from utils.spatial_data import write_json


class Metrics:
    def __init__(self, sfreq, stride=16, window_output=None):
        self.sfreq, self.stride = sfreq, stride
        self.totals = {}
        self.n_windows = 0
        self.channel_samples = 0
        self.window_output = Path(window_output) if window_output else None
        self.window_written = False
        if self.window_output:
            self.window_output.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def add(self, batch, x, y, bits_s, bits_t):
        x, y = x.float(), y.float()
        mask = batch["mask"].to(x.device)
        clean = ~mask.flatten(1).any(1)
        keep = ~mask
        comps = _window_components(x, y, self.sfreq)
        comps["wave"] = ((x.square()*keep).sum((1, 2)), ((x-y).square()*keep).sum((1, 2)),
                         (y.square()*keep).sum((1, 2)))
        data = {}
        for band, (energy, error, power) in comps.items():
            for key, value in (("energy", energy), ("error", error), ("power", power)):
                data[f"{band}_{key}"] = value
                if band != "wave":
                    data[f"untouched_{band}_{key}"] = value * clean
        def covariance(v):
            v = v-v.mean(-1, keepdim=True)
            return v @ v.transpose(1, 2) / v.shape[-1]
        cx, cy = covariance(x), covariance(y)
        def corr(c):
            sd = c.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
            return c / (sd[:, :, None]*sd[:, None, :])
        data["covariance_error_sum"] = (cx-cy).flatten(1).norm(dim=1) / cx.flatten(1).norm(dim=1).clamp_min(1e-12)
        data["correlation_error_sum"] = (corr(cx)-corr(cy)).flatten(1).norm(dim=1) / corr(cx).flatten(1).norm(dim=1).clamp_min(1e-12)
        data["peak_error_sum"] = (x.abs().amax((1, 2))-y.abs().amax((1, 2))).abs() / x.abs().amax((1, 2)).clamp_min(1e-12)
        lx, ly = x.diff().abs().mean((1, 2)), y.diff().abs().mean((1, 2))
        data["line_length_error_sum"] = (lx-ly).abs()/lx.clamp_min(1e-12)
        boundary = torch.arange(x.shape[-1], device=x.device) % self.stride
        boundary = (boundary == 0) | (boundary == self.stride-1)
        bk = keep & boundary[None, None, :]
        data["boundary_error"] = ((x-y).square()*bk).sum((1, 2))
        data["boundary_energy"] = (x.square()*bk).sum((1, 2))
        data["bits_spatial"], data["bits_temporal"] = bits_s.float(), bits_t.float()
        data["channel_samples"] = torch.full_like(bits_s, x[0].numel())
        data["n_windows"] = torch.ones_like(bits_s)
        data["untouched_windows"] = clean.float()
        for c in range(x.shape[1]):
            data[f"ch{c}_energy"] = (x[:, c].square()*keep[:, c]).sum(1)
            data[f"ch{c}_error"] = ((x[:, c]-y[:, c]).square()*keep[:, c]).sum(1)
        names = list(data)
        values = torch.stack([data[k] for k in names], 1).cpu().numpy()
        window_rows = []
        for i, vals in enumerate(values):
            key = (batch["sha256_id"][i], batch["patient_id"][i])
            if key not in self.totals:
                self.totals[key] = np.zeros(len(names), np.float64)
            self.totals[key] += vals.astype(np.float64)
            if self.window_output:
                window_rows.append(dict(zip(names, map(float, vals)), sha256_id=key[0],
                                        patient_id=key[1], start=int(batch["start"][i])))
        self.names = names
        self.n_windows += len(x)
        self.channel_samples += x.numel()
        if self.window_output:
            pd.DataFrame(window_rows).to_csv(self.window_output, mode="a" if self.window_written else "w",
                                             index=False, header=not self.window_written)
            self.window_written = True

    def finish(self, output=None):
        if not self.totals:
            raise ValueError("No evaluation windows")
        records = pd.DataFrame([dict(zip(self.names, vals), sha256_id=key[0], patient_id=key[1])
                                for key, vals in self.totals.items()])
        metrics = []
        for col in list(records):
            if col.endswith("_error"):
                stem = col[:-6]
                denom = records[f"{stem}_energy"].replace(0, np.nan)
                name = f"{stem}_nmse"
                records[name] = records[col]/denom
                metrics.append(name)
                if f"{stem}_power" in records:
                    name = f"{stem}_power_ratio"
                    records[name] = records[f"{stem}_power"]/denom
                    metrics.append(name)
            elif col.endswith("_sum"):
                name = col[:-4]
                records[name] = records[col]/records.n_windows
                metrics.append(name)
        for branch in ("spatial", "temporal"):
            name = f"rate_{branch}"
            records[name] = records[f"bits_{branch}"]/records.channel_samples
            metrics.append(name)
        records["rate_total"] = records.rate_spatial + records.rate_temporal
        metrics.append("rate_total")
        patients = records.groupby("patient_id", as_index=False)[metrics].mean()
        summary = {k: float(patients[k].mean()) for k in metrics if patients[k].notna().any()}
        summary.update(patients=len(patients), recordings=len(records), windows=self.n_windows,
                       pooled_wave_nmse=float(records.wave_error.sum()/max(records.wave_energy.sum(), 1e-12)),
                       wave_nmse_p90=float(patients.wave_nmse.quantile(.9)))
        if output:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            records.to_csv(output/"recordings.csv", index=False)
            patients.to_csv(output/"patients.csv", index=False)
            write_json(output/"summary.json", summary)
        return summary


def sequence_stats(labels, batch, previous, counts, runs):
    """Streaming transitions; each crop is an explicit independently coded segment."""
    # No artificial transitions across crop boundaries, rejected gaps, or patients.
    for row in labels:
        row = np.asarray(row, dtype=int)
        counts["initial"][row[0]] += 1
        np.add.at(counts["occupancy"], row, 1)
        if len(row)>1:
            np.add.at(counts["transition"], (row[:-1], row[1:]), 1)
        changes = np.r_[0, np.flatnonzero(row[1:] != row[:-1])+1, len(row)]
        for i, (start, stop) in enumerate(zip(changes[:-1], changes[1:])):
            runs.append(dict(state=int(row[start]), frames=int(stop-start),
                             censored=bool(i == 0 or i == len(changes)-2)))


def paired_bootstrap(before, after, repeats=2000, seed=42):
    a, b = before.set_index("patient_id"), after.set_index("patient_id")
    if set(a.index) != set(b.index):
        raise ValueError("Paired comparisons require the same patients")
    b = b.loc[a.index]
    rng = np.random.default_rng(seed)
    rows = []
    for col in a.columns.intersection(b.columns):
        diff = (b[col]-a[col]).dropna().to_numpy()
        if not len(diff):
            continue
        # Bounded batches avoid a large repeats x patients array on full datasets.
        draws = np.concatenate([diff[rng.integers(0, len(diff), (min(256, repeats-start), len(diff)))].mean(1)
                                for start in range(0, repeats, 256)])
        lo, hi = np.quantile(draws, [.025, .975])
        rows.append(dict(metric=col, difference=float(diff.mean()), ci_low=float(lo), ci_high=float(hi), patients=len(diff)))
    return rows
