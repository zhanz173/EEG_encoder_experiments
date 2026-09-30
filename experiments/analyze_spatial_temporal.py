"""Temporal geometry of pilot spatial codes on five stratified validation EEGs.

Uses the exact saved, untouched evaluation windows; no cross-window lag pairs.
All reported inference uses the pilot's BF16 setting. No training or test access.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.spatial_runtime import load_checkpoint, amp_context, runtime

LAGS = [1, 2, 4, 8, 16, 32, 64]
METRICS = ['latent_cosine', 'centered_latent_cosine', 'matrix_cosine',
           'centered_matrix_cosine', 'subspace_affinity']


def cosine(a, b):
    denom = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    values = np.full(denom.shape, np.nan)
    np.divide((a*b).sum(-1), denom, out=values, where=denom > 1e-12)
    return values


def geometry(z, a, q, lag):
    # z and a are dictionaries containing raw and recording-centered arrays.
    def pair(v):
        return v[:, :-lag], v[:, lag:]
    vals = {}
    for key, v in {**z, **a}.items():
        x, y = pair(v)
        vals[key] = cosine(x, y).ravel()
    x, y = q[:, :-lag], q[:, lag:]
    # Rank-normalized overlap: 1 for identical full-rank subspaces, 0 for orthogonal.
    vals['subspace_affinity'] = ((np.swapaxes(x, -1, -2) @ y)**2).sum((-1,-2)).ravel()/q.shape[-1]
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/spatial_pilot')
    p.add_argument('--manifest', default='H:/EEG/FHA/Resting/preprocessed/manifests/recordings.parquet')
    p.add_argument('--shards', default='H:/EEG/FHA/Resting/preprocessed/shards')
    p.add_argument('--output', default='runs/spatial_temporal_analysis')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    reference = root/'factorized_k8_lambda0.02_seed0'
    records = pd.read_csv(reference/'posthoc_val/original/recordings.csv').sort_values(['wave_nmse','sha256_id'])
    selected = []
    for i, quantile in enumerate([.1,.3,.5,.7,.9]):
        row = records.iloc[round(quantile*(len(records)-1))].to_dict()
        row.update(eeg=f'EEG {i+1}', selection_quantile=quantile)
        selected.append(row)
    selected = pd.DataFrame(selected)
    assert selected.patient_id.nunique() == 5
    manifest = pd.read_parquet(args.manifest)
    selected = selected.merge(manifest, on='sha256_id', how='left', validate='one_to_one')
    assert selected.shard_name.notna().all()
    windows = pd.read_csv(reference/'evaluation_val/windows.csv')
    selected.to_csv(out/'selected_recordings.csv', index=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    runtime(device, 'bf16', threads=4)
    rng = np.random.default_rng(42)
    lag_rows, controls, norms, series, checks = [], [], [], [], []
    for lam in [.02,.1]:
        run = root/f'factorized_k8_lambda{lam}_seed0'
        config = json.loads((run/'posthoc_val/evaluation_config.json').read_text())
        digest = hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        assert digest == config['checkpoint_sha256']
        model, ck = load_checkpoint(run/'best.pt',device)
        for rec in selected.to_dict('records'):
            w = windows[windows.sha256_id.eq(rec['sha256_id'])].sort_values('start')
            assert (w.untouched_windows == 1).all()
            starts = w.start.to_numpy(int)
            assert len(starts) > 0 and (np.diff(starts) >= 2048).all()
            with h5py.File(Path(args.shards)/rec['shard_name'],'r') as f:
                x = np.stack([f['signals'][int(rec['index_in_shard']),:,s:s+2048] for s in starts]).astype(np.float32)
            assert np.isfinite(x).all() and x.shape[1:] == (20,2048)
            # Verify source identity/units against additive saved pilot window energies.
            energy = (x.astype(np.float64)/ck['scale'])**2
            energy = energy.sum((1,2))
            relative_error = np.max(np.abs(energy-w.wave_energy.to_numpy()) / w.wave_energy.to_numpy())
            assert relative_error < 2e-5, relative_error
            zz, aa, pre = [], [], []
            with torch.inference_mode():
                for start in range(0,len(x),8):
                    batch = torch.from_numpy(x[start:start+8]).to(device)/ck['scale']
                    with amp_context(device,'bf16'):
                        patches = batch.unfold(-1,16,16).permute(0,2,1,3).flatten(2)
                        zp = model.spatial_encoder(patches).float()
                        z = zp.round()
                        a = model.decode_spatial(z.transpose(1,2))
                    zz.append(z.cpu().numpy()); aa.append(a.cpu().numpy()); pre.append(zp.cpu().numpy())
            z, a, zp = np.concatenate(zz),np.concatenate(aa),np.concatenate(pre)
            u, sv, _ = np.linalg.svd(a,full_matrices=False)
            rank = (sv > np.maximum(sv[..., :1]*1e-5,1e-8)).sum(-1)
            assert (rank==8).all(), 'Subspace affinity requires rank-aware handling for deficient frames'
            flat = a.reshape(*a.shape[:2],-1)
            zs={'latent_cosine':z,'centered_latent_cosine':z-z.mean((0,1),keepdims=True)}
            matrices={'matrix_cosine':flat,'centered_matrix_cosine':flat-flat.mean((0,1),keepdims=True)}
            for lag in LAGS:
                for metric, vals in geometry(zs,matrices,u,lag=lag).items():
                    lag_rows.append(dict(eeg=rec['eeg'],lambda_rate=lam,lag_frames=lag,lag_ms=lag*62.5,
                        metric=metric,mean=float(np.nanmean(vals)),median=float(np.nanmedian(vals)),
                        p10=float(np.nanquantile(vals,.1)),p90=float(np.nanquantile(vals,.9)),
                        pairs=int(np.isfinite(vals).sum()),undefined_pairs=int(np.isnan(vals).sum())))
            # Permute full sequences within each crop; preserve all frame marginals.
            for repeat in range(20):
                order = np.stack([rng.permutation(z.shape[1]) for _ in range(len(z))])
                shuffled_z = {k:np.take_along_axis(v,order[...,None],axis=1) for k,v in zs.items()}
                shuffled_a = {k:np.take_along_axis(v,order[...,None],axis=1) for k,v in matrices.items()}
                shuffled_q = np.take_along_axis(u,order[...,None,None],axis=1)
                for metric, vals in geometry(shuffled_z,shuffled_a,shuffled_q,lag=1).items():
                    controls.append(dict(eeg=rec['eeg'],lambda_rate=lam,repeat=repeat,metric=metric,mean=float(np.nanmean(vals))))
            zn = np.linalg.norm(z,axis=-1)
            zstep = np.linalg.norm(np.diff(z,axis=1),axis=-1)
            astep = np.linalg.norm(np.diff(flat,axis=1),axis=-1)
            norms.append(dict(eeg=rec['eeg'],lambda_rate=lam,latent_norm_mean=float(zn.mean()),
                latent_norm_p10=float(np.quantile(zn,.1)),latent_norm_p90=float(np.quantile(zn,.9)),
                latent_step_mean=float(zstep.mean()),matrix_step_mean=float(astep.mean()),
                mean_vector_energy_fraction=float((z.mean((0,1))**2).sum()/(z*z).sum(-1).mean()),
                identical_adjacent_codes=float((np.diff(z,axis=1)==0).all(-1).mean()),
                zero_latent_frames=int((zn==0).sum()), min_rank=int(rank.min()),
                median_condition=float(np.median(sv[...,0]/sv[...,-1]))))
            # A deterministic middle crop supports an inspectable trajectory view.
            mid=len(starts)//2
            vals=geometry({k:v[mid:mid+1] for k,v in zs.items()},
                          {k:v[mid:mid+1] for k,v in matrices.items()},u[mid:mid+1],lag=1)
            for j in range(127):
                series.append(dict(eeg=rec['eeg'],lambda_rate=lam,time_s=(j+1)*.0625,
                    start_sample=int(starts[mid]),latent_norm=float(zn[mid,j+1]),
                    **{k:float(v[j]) for k,v in vals.items()}))
            np.savez_compressed(out/f"eeg{rec['eeg'].split()[-1]}_lambda{lam}_spatial.npz",
                z_quantized=z,z_prequant=zp,a=a,starts=starts,sfreq=256,
                sha256_id=rec['sha256_id'],checkpoint_sha256=digest)
            checks.append(dict(eeg=rec['eeg'],lambda_rate=lam,windows=len(starts),frames=int(zn.size),
                source_energy_max_relative_error=float(relative_error),checkpoint_sha256=digest))
            print(f"{rec['eeg']} lambda={lam}: {len(starts)} windows, {zn.size} frames",flush=True)
    lagdf=pd.DataFrame(lag_rows); ctrl=pd.DataFrame(controls); normdf=pd.DataFrame(norms)
    lagdf.to_csv(out/'lag_similarity.csv',index=False)
    ctrl.to_csv(out/'shuffled_controls.csv',index=False)
    normdf.to_csv(out/'latent_norms.csv',index=False)
    pd.DataFrame(series).to_csv(out/'example_trajectories.csv',index=False)
    summary=lagdf.groupby(['lambda_rate','metric','lag_ms'],as_index=False)['mean'].mean()
    sh=ctrl.groupby(['lambda_rate','metric'],as_index=False)['mean'].mean().rename(columns={'mean':'shuffled_mean'})
    summary=summary.merge(sh,on=['lambda_rate','metric'])
    summary['excess_over_shuffle']=summary['mean']-summary.shuffled_mean
    summary.to_csv(out/'equal_recording_summary.csv',index=False)
    (out/'analysis_config.json').write_text(json.dumps(dict(arguments=vars(args),checks=checks,
        selection='Five distinct-patient recordings nearest the 10/30/50/70/90% recording NMSE ranks of factorized lambda .02 validation',
        inference='BF16 spatial encoder/decoder, float32 normalization; saved quantized codes',
        centering='Subtract each recording mean across all its accepted frames, separately per model',
        controls='20 independent within-crop permutations; no cross-crop pairs',
        aggregation='Equal recordings; exploratory selected cases, not a population estimate',
        subspace='Squared Frobenius overlap of orthonormal bases divided by rank 8; full rank verified'),indent=2))
    plot(out,lagdf,ctrl,pd.DataFrame(series))
    print(summary[summary.lag_ms.isin([62.5,250,1000,4000])].round(4).to_string(index=False))
    print(normdf.round(4).to_string(index=False))


def plot(out,df,ctrl,series):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=['latent_cosine','centered_latent_cosine','matrix_cosine','centered_matrix_cosine','subspace_affinity']
    titles=['Quantized latent: raw cosine','Quantized latent: recording-centered cosine',
            'Ordered spatial matrix: raw cosine','Spatial matrix: recording-centered cosine','Spatial subspace overlap']
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(5,2,figsize=(12,16),layout='constrained')
    for col,lam in enumerate([.02,.1]):
        for row,(metric,title) in enumerate(zip(metrics,titles)):
            ax=axes[row,col]
            for i,eeg in enumerate(sorted(df.eeg.unique())):
                g=df[(df.lambda_rate==lam)&(df.metric==metric)&(df.eeg==eeg)].sort_values('lag_ms')
                ref=ctrl[(ctrl.lambda_rate==lam)&(ctrl.metric==metric)&(ctrl.eeg==eeg)]['mean'].mean()
                ax.plot(g.lag_ms,g['mean'],'o-',color=colors[i],label=eeg,ms=3)
                ax.axhline(ref,color=colors[i],ls=':',alpha=.65,lw=1)
            limits = (.8,1.005) if metric=='subspace_affinity' else ((.9,1.005) if metric=='matrix_cosine' else (-1,1.02))
            ax.set(title=f'{title} | λ={lam}',xscale='log',ylim=limits,ylabel='Similarity')
            ax.set_xticks([62.5,250,1000,4000],['62.5','250','1000','4000'])
            ax.grid(alpha=.2)
            if row==0: ax.legend(ncol=3,fontsize=8)
            if row==4: ax.set_xlabel('Time lag (ms)')
    fig.suptitle('Temporal spatial-code geometry in five validation EEGs\nSolid: observed lag pairs • dotted: within-window shuffled reference',fontsize=14)
    fig.supxlabel('Each curve uses all saved clean 8 s windows of one recording. Pairs never cross crop boundaries.\nCentered means subtracting the recording mean; subspace overlap ignores basis rotations/signs.',fontsize=10)
    fig.savefig(out/'temporal_similarity.png',dpi=150);fig.savefig(out/'temporal_similarity.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,lam in zip(axes,[.02,.1]):
        for i,eeg in enumerate(sorted(df.eeg.unique())):
            g=df[(df.lambda_rate==lam)&(df.metric=='centered_latent_cosine')&(df.eeg==eeg)].sort_values('lag_ms')
            ax.plot(g.lag_ms,g['mean'],'o-',color=colors[i],label=eeg)
        ref=ctrl[(ctrl.lambda_rate==lam)&(ctrl.metric=='centered_latent_cosine')]['mean'].mean()
        ax.axhline(ref,color='black',ls=':',label='Shuffled mean')
        ax.set(title=f'Factorized K=8, λ={lam}',xscale='log',ylim=(-1,1.02),
            xlabel='Time lag (ms)',ylabel='Recording-centered latent cosine')
        ax.set_xticks([62.5,125,250,500,1000,2000,4000],['62.5','125','250','500','1000','2000','4000'])
        ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.suptitle('Spatial latent direction has short-timescale structure that differs across EEGs',fontsize=13)
    fig.supxlabel('Five selected validation recordings • 43 clean 8 s windows each • no cross-window pairs',fontsize=10)
    fig.savefig(out/'latent_similarity_summary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(12,12),layout='constrained')
    for i,eeg in enumerate(sorted(df.eeg.unique())):
        for col,lam in enumerate([.02,.1]):
            g=series[(series.eeg==eeg)&(series.lambda_rate==lam)]
            ax=axes[i,col]
            ax.plot(g.time_s,g.centered_latent_cosine,label='Centered latent',color='#2166ac',lw=1)
            ax.plot(g.time_s,g.matrix_cosine,label='Spatial matrix',color='#b2182b',lw=1)
            ax.set(title=f'{eeg} | λ={lam} | crop start {g.start_sample.iloc[0]/256:g} s',ylim=(-1,1.02),ylabel='Adjacent-frame cosine')
            ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=8)
            if i==4:ax.set_xlabel('Seconds within middle accepted crop')
    fig.suptitle('Example temporal trajectories: adjacent frames 62.5 ms apart',fontsize=14)
    fig.savefig(out/'example_trajectories.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
