"""Fit database-only models, run pinned RaBitQ, calibrate M without test selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from prepare_truth import dump, fbin, sha, write_fbin


def assign(x, centers):
    out=np.empty(len(x),dtype=np.uint32)
    for b in range(0,len(x),4096):
        v=np.asarray(x[b:b+4096],dtype=np.float64)
        d=(v*v).sum(1)[:,None]+(centers*centers).sum(1)[None,:]-2*v@centers.T
        out[b:b+4096]=np.argmin(d,axis=1)
    return out


def kmeans(x, seed):
    rng=np.random.default_rng(seed)
    centers=[x[rng.integers(len(x))].copy()]
    closest=np.full(len(x),np.inf)
    for _ in range(15):
        closest=np.minimum(closest,((x-centers[-1])**2).sum(1))
        if closest.sum()==0:raise ValueError('Fewer than 16 unique fit vectors')
        centers.append(x[rng.choice(len(x),p=closest/closest.sum())].copy())
    centers=np.asarray(centers)
    for step in range(100):
        labels=assign(x,centers)
        updated=np.array([x[labels==c].mean(0) if np.any(labels==c) else centers[c] for c in range(16)])
        shift=float(np.max(np.linalg.norm(updated-centers,axis=1)))
        scale=1+float(np.max(np.linalg.norm(centers,axis=1)))
        centers=updated
        if shift<=1e-6*scale:break
    return centers,{'iterations':step+1,'relative_max_shift':shift/scale,'empty_clusters':int(sum(not np.any(labels==c) for c in range(16)))}


def fit(a):
    root=Path(a.data);out=root/f'seed_{a.seed}';out.mkdir(exist_ok=False)
    start=time.monotonic();x=fbin(root/'base.fbin');fit=np.load(root/'split_ids.npz')['fit_positions']
    train=np.asarray(x[fit],dtype=np.float64)
    mu=train.mean(0);centered=train-mu
    vals,basis=np.linalg.eigh(centered.T@centered)
    order=np.argsort(vals)[::-1];basis=basis[:,order];vals=vals[order]
    # Fix arbitrary eigenvector signs to improve reproducibility of serialized models.
    signs=np.sign(basis[np.argmax(np.abs(basis),axis=0),np.arange(basis.shape[1])]);basis*=signs
    centers,info=kmeans(train,a.seed)
    centers=centers.astype(np.float32)
    labels=assign(x,centers.astype(np.float64))
    write_fbin(out/'centers.fbin',centers);labels.tofile(out/'labels.u32')
    np.savez(out/'global_reference.npz',mu=mu,basis=basis,eigenvalues=vals)
    info.update({'seed':a.seed,'fit_objects':len(fit),'fit_seconds':time.monotonic()-start,
                 'cluster_counts':np.bincount(labels,minlength=16).tolist(),
                 'artifacts':{p.name:sha(p) for p in [out/'centers.fbin',out/'labels.u32',out/'global_reference.npz']}})
    dump(out/'fit_manifest.json',info)
    print('fit',a.data,a.seed,info,flush=True)


def run_score(a):
    root=Path(a.data);out=root/f'seed_{a.seed}'
    binary=Path(a.binary).resolve()
    cmd=[str(binary),str(root/'base.fbin'),str(root/'queries.fbin'),str(out/'centers.fbin'),str(out/'labels.u32'),str(a.seed),str(a.bits),str(out/f'rq{a.bits}')]
    with open(out/f'rq{a.bits}.log','x') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
    if p.returncode:raise RuntimeError(f'RaBitQ adapter failed; preserve {log.name}')
    rq=out/f'rq{a.bits}'
    meta=json.loads((rq/'metadata.json').read_text())
    assert meta['n']==100000 and meta['query_n']==1000 and meta['code_scored_count_per_query']==100000 and not meta['raw_rerank']
    dump(rq/'provenance.json',{'binary_sha256':sha(binary),'source_commit':'dd6aa230c49082bf4aad4a5a56f513d87a2084ef','artifacts':{p.name:sha(p) for p in rq.iterdir() if p.is_file()}})
    print('score gate',a.data,a.seed,a.bits,'PASS',flush=True)


def calibrate(a):
    root=Path(a.data);out=root/f'seed_{a.seed}'
    if (out/'front_frozen.json').exists():raise ValueError('Front already frozen')
    scores=fbin(out/'rq1/scores.fbin');s=np.load(root/'split_ids.npz');ids=s['base_ids']
    gt=np.load(root/'truth/top100.npz')['ids']
    grid=[128,256,512,1024,2048,4096,8192,16384,32768,65536,len(ids)]
    hits=np.zeros(len(grid),dtype=np.int64)
    for qi in s['calibration_positions']:
        order=np.lexsort((ids,scores[qi]));ranks=np.empty(len(ids),dtype=np.int64);ranks[order]=np.arange(len(ids))
        tr=ranks[np.searchsorted(ids,gt[qi,:10])]
        hits+=np.array([np.count_nonzero(tr<m) for m in grid])
    recall=hits/(len(s['calibration_positions'])*10)
    M=next(m for m,r in zip(grid,recall) if r>=0.9995)
    dump(out/'front_frozen.json',{'M':M,'N':len(ids),'M_over_N':M/len(ids),'calibration_grid':[{'M':m,'recall':float(r)} for m,r in zip(grid,recall)],
         'score_sha256':sha(out/'rq1/scores.fbin'),'selection':'smallest_calibration_feasible','test_used_for_selection':False})
    candidates=np.lib.format.open_memmap(out/'candidates.npy',mode='w+',dtype=np.uint32,shape=(len(scores),M))
    for qi in range(len(scores)):
        candidates[qi]=np.lexsort((ids,scores[qi]))[:M]
    candidates.flush();del candidates
    dump(out/'candidate_manifest.json',{'sha256':sha(out/'candidates.npy'),'ID_encoding':'positions in sorted base_ids; not original IDs','M':M,'all_methods_share_this_file':True})
    print('F0 frozen',a.data,a.seed,'M',M,'calibration recall',float(recall[grid.index(M)]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['fit','score','calibrate']);p.add_argument('--data',required=True);p.add_argument('--seed',type=int,required=True,choices=[17,29,43]);p.add_argument('--bits',type=int,default=1);p.add_argument('--binary',default='build-library/front_score')
    a=p.parse_args();{'fit':fit,'score':run_score,'calibrate':calibrate}[a.phase](a)
