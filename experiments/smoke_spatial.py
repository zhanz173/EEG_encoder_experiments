"""Small CPU end-to-end check using generated EEG-like data, never clinical data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import h5py
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir")
    p.add_argument("--workers", type=int, default=0, help="Use 1 to test Windows spawn workers")
    p.add_argument("--qc", action="store_true", help="Exercise accepted-window QC and raw-unit interpolation")
    p.add_argument("--verify-resume", action="store_true", help="Compare resumed and uninterrupted factorized training")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir or root/"runs"/f"spatial_smoke_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if out.exists():
        raise ValueError("Smoke output must be a new directory")
    out.mkdir(parents=True)
    rng = np.random.default_rng(10)
    t = np.arange(768)/256
    signals, rows, meta = [], [], []
    for i in range(30):
        a = rng.normal(size=(20, 3)); a /= np.linalg.norm(a, axis=0)
        s = np.array([np.sin(2*np.pi*f*t+rng.uniform(-3, 3)) for f in (3, 10, 22)])
        x = a@s + .02*rng.normal(size=(20, len(t)))
        signals.append(x.astype(np.float32))
        key = f"scan{i:03d}"
        rows.append(dict(sha256_id=key, shard_name="synthetic.h5", index_in_shard=i,
                         n_samples=len(t), sfreq=256, n_channels=20, status="ok"))
        meta.append(dict(ScanID=key, Hashed_PatientURN=f"patient{i//2:03d}"))
    with h5py.File(out/"synthetic.h5", "w") as f:
        f.create_dataset("signals", data=np.stack(signals), chunks=(1, 20, 256))
    pd.DataFrame(rows).to_csv(out/"manifest.csv", index=False)
    pd.DataFrame(meta).to_csv(out/"metadata.csv", index=False)
    qc_options = ["--allow-no-qc"]
    if args.qc:
        qc = out/"qc"
        qc.mkdir()
        (qc/"qc_config.json").write_text(json.dumps(dict(window_sec=1., mask_pad_ms=0.)))
        pd.DataFrame([dict(sha256_id=r["sha256_id"], start_sample=start, status="accept")
                      for r in rows for start in (0, 256, 512)]).to_parquet(qc/"window_qc.parquet", index=False)
        pd.DataFrame([dict(sha256_id=r["sha256_id"], channel_index=0, start_sample=10,
                           stop_sample=12, masked=True) for r in rows]).to_parquet(qc/"events.parquet", index=False)
        qc_options = ["--qc-dir", qc]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    def run(script, *options):
        command = [sys.executable, "-m", "experiments." + Path(script).stem, *map(str, options)]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)
    run("prepare_spatial.py", "--manifest", out/"manifest.csv", "--metadata", out/"metadata.csv",
        "--shards-dir", out, "--output-dir", out/"prepared", *qc_options, "--window-sec", 1,
        "--eval-start-sec", 0, "--train-recordings", 8, "--val-recordings", 2, "--test-recordings", 2)
    prepared_records = pd.read_csv(out/"prepared/records.csv")
    assert prepared_records.groupby("patient_id").split.nunique().max() == 1
    if args.qc:
        from utils.spatial_data import SpatialDataset
        ds = SpatialDataset(out/"prepared", out, "train")
        item = ds[0]
        with h5py.File(out/"synthetic.h5", "r") as f:
            raw = f["signals"][int(ds.rows[0]["index_in_shard"]), :, :256]
        assert np.array_equal(item["x"].numpy()[1:], raw[1:])
        assert int(item["mask"].sum()) == 2
        assert np.allclose(item["x"][0, 10:12], np.interp([10, 11], [9, 12], raw[0, [9, 12]]))
        ds.close()
    for architecture in ("factorized", "baseline"):
        dest = out/architecture
        common = ["--prepared", out/"prepared", "--shards-dir", out, "--device", "cpu", "--threads", 2,
                  "--batch-size", 2, "--workers", args.workers]
        training = [*common, "--architecture", architecture, "--rank", 4, "--spatial-dim", 8,
                    "--temporal-dim", 8, "--latent-dim", 16, "--max-steps", 2, "--train-windows", 1,
                    "--precision", "fp32", "--evaluate-test"]
        run("train_spatial.py", *training, "--output-dir", dest, "--epochs", 2)
        if args.verify_resume and architecture == "factorized":
            import torch
            run("train_spatial.py", *training, "--output-dir", dest, "--epochs", 3, "--resume")
            run("train_spatial.py", *training, "--output-dir", out/"uninterrupted", "--epochs", 3)
            resumed = torch.load(dest/"last.pt", map_location="cpu", weights_only=False)
            control = torch.load(out/"uninterrupted/last.pt", map_location="cpu", weights_only=False)
            assert resumed["epoch"] == control["epoch"] == 3
            for key, value in resumed["model"].items():
                assert torch.equal(value, control["model"][key]), f"Resume differs: {key}"
        run("posthoc_spatial.py", "fit", *common, "--checkpoint", dest/"best.pt",
            "--output-dir", dest/"dictionary", "--sizes", 1, 2, "--holds", 1, 2,
            "--max-samples", 64, "--fit-windows", 1, "--precision", "fp32")
        run("posthoc_spatial.py", "evaluate", *common, "--bundle", dest/"dictionary/dictionary.pt",
            "--output-dir", dest/"posthoc_val", "--bootstrap", 20)
        comparison = pd.read_csv(dest/"posthoc_val/comparison.csv")
        assert np.isfinite(comparison[["rate_total", "wave_nmse", "delta_wave_nmse"]]).all().all()
        if architecture == "factorized":
            ref = json.loads((dest/"posthoc_val/original/summary.json").read_text())
            assert np.allclose(comparison.rate_temporal, ref["rate_temporal"])
        assert (comparison[comparison.effective_m.eq(1)].rate_spatial == 0).all()
    run("summarize_spatial.py", "--root", out, "--output-dir", out/"summary")
    print(f"SMOKE PASSED: {out}", flush=True)


if __name__ == "__main__":
    main()
