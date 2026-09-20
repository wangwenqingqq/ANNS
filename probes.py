"""Stage A linear probes. Oracle access is diagnostic, never deployable search."""
import argparse
import csv
import json
from pathlib import Path
import resource
import time
import numpy as np
from prepare_truth import dump, fbin, sha

METHODS=('G','L','Q','H')


def mix64(x):
    x=np.asarray(x,dtype=np.uint64)
    x=x+np.uint64(0x9e3779b97f4a7c15)
    x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x^(x>>np.uint64(31))


def orthogonal(v,p,r):
    v=v-p@(p.T@v)
    u,s,_=np.linalg.svd(v,full_matrices=False)
    if len(s)<r or s[r-1]<1e-10:raise ValueError('Rank-deficient projected basis')
    # QR preserves ordered principal directions; SVD would rotate a rank-32 span
    # arbitrarily and invalidate taking its first 16 principal directions.
    q,_=np.linalg.qr(v)
    return q[:,:r]


def directions(s,p,r=32):
    vals,v=np.linalg.eigh(s)
    return orthogonal(v[:,np.argsort(vals)[::-1][:r]],p,r)


def sketch(x,mu,p,v,labels):
    # All transform arithmetic uses FP64, including for reloaded FP32 models.
    # Main persistent fields are rounded to FP32 and then read back before use.
    y=np.asarray(x,dtype=np.float64)-np.asarray(mu,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    a=y@p;e=y-a@p.T
    b=np.empty((len(x),v.shape[-1]));s=np.empty(len(x))
    for c in np.unique(labels):
        where=np.flatnonzero(labels==c);vc=v[0 if len(v)==1 else c]
        b[where]=e[where]@vc
        residual=e[where]-b[where]@vc.T
        s[where]=np.linalg.norm(residual,axis=1)
    return np.column_stack((a,b,s,(y*y).sum(1)))


def fit(a):
    root=Path(a.data);run=root/f'seed_{a.seed}';out=run/'probes';out.mkdir(exist_ok=False)
    start=time.monotonic();x=fbin(root/'base.fbin');queries=fbin(root/'queries.fbin')
    split=np.load(root/'split_ids.npz');ids=split['base_ids'];qids=split['query_ids'];fitpos=split['fit_positions']
    labels=np.fromfile(run/'labels.u32',dtype=np.uint32)
    c0=np.load(run/'candidates.npy',mmap_mode='r')
    global_model=np.load(run/'global_reference.npz');mu=global_model['mu'];global_basis=global_model['basis']
    r0=16 if x.shape[1]==128 else 64;p=global_basis[:,:r0];g=global_basis[:,r0:r0+32]
    y=np.asarray(x,dtype=np.float64)-mu;e=y-(y@p)@p.T;del y
    eq=np.asarray(queries[:256],dtype=np.float64)-mu;eq-= (eq@p)@p.T
    pairs=[];rng=np.random.default_rng(a.seed)
    for qi in range(256):
        chosen=c0[qi][np.argsort(mix64(ids[c0[qi]].astype(np.uint64)^mix64(np.array([qids[qi]+20260920],dtype=np.uint64))[0]),kind='stable')[:256]]
        for c in range(16):
            hard=chosen[labels[chosen]==c];universe=np.flatnonzero(labels==c)
            random=rng.choice(universe,len(hard),replace=False)
            for hp,qp in zip(hard,random):pairs.append((qi,int(hp),int(qp),c))
    pairs=np.asarray(pairs,dtype=np.int64)
    np.savez(out/'training_pairs.npz',query_positions=pairs[:,0],query_ids=qids[pairs[:,0]],hard_positions=pairs[:,1],hard_ids=ids[pairs[:,1]],random_positions=pairs[:,2],random_ids=ids[pairs[:,2]],cluster_ids=pairs[:,3])
    learned={};counts={};zeros={}
    for method,col in [('H',1),('Q',2)]:
        z=eq[pairs[:,0]]-e[pairs[:,col]];norm=np.linalg.norm(z,axis=1);valid=norm>0
        per_query=np.bincount(pairs[valid,0],minlength=256)
        z[valid]/=norm[valid,None]
        z[valid]/=np.sqrt(per_query[pairs[valid,0],None])
        bases=[];counts[method]=[]
        for c in range(16):
            rows=valid&(pairs[:,3]==c);counts[method].append(int(rows.sum()))
            if rows.sum()>=128:bases.append(directions(z[rows].T@z[rows],p))
            elif rows.sum()>=64:
                short=directions(z[rows].T@z[rows],p,16)
                # Only the first 16 are used; rank32 falls back to G below.
                bases.append(np.column_stack((short,np.zeros((x.shape[1],16)))))
            else:bases.append(g.copy())
        learned[method]=np.array(bases);zeros[method]=int((~valid).sum());del z
    local=[]
    for c in range(16):
        train=e[fitpos[labels[fitpos]==c]]
        if len(train)<33:raise ValueError('Local PCA training cluster rank insufficient')
        train=train-train.mean(0);local.append(directions(train.T@train,p))
    learned['L']=np.array(local);learned['G']=g[None,:,:]
    fallback={}
    for r in (16,32):
        sparse=[c for c in range(16) if min(counts['H'][c],counts['Q'][c])<4*r];fallback[str(r)]=sparse
        for method in METHODS:
            v=learned[method][:,:,:r].copy()
            if method in ('H','Q'):
                for c in sparse:v[c]=g[:,:r]
            np.savez(out/f'{method}_r{r}_reference.npz',mu=mu,p=p,v=v)
            # Freeze model, read it back, then build its own summaries.
            np.savez(out/f'{method}_r{r}_fp32.npz',mu=mu.astype(np.float32),p=p.astype(np.float32),v=v.astype(np.float32))
            for precision in ('reference','fp32'):
                m=np.load(out/f'{method}_r{r}_{precision}.npz')
                sx=sketch(x,m['mu'],m['p'],m['v'],labels)
                np.save(out/f'{method}_r{r}_{precision}_sketch.npy',sx.astype(np.float32 if precision=='fp32' else np.float64))
                del sx
    dump(out/'fit_manifest.json',{'seed':a.seed,'r0':r0,'ranks':[16,32],'pair_counts':counts,'zero_training_pairs':zeros,'shared_fallback_clusters':fallback,
        'training_pair_count':len(pairs),'each_query_nonzero_weight_sum':1,'fit_seconds':time.monotonic()-start,
        'artifacts':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print('probes fitted',a.data,a.seed,'seconds',time.monotonic()-start,flush=True)


def read_manifest(path):return json.loads(Path(path).read_text())


def query_fields(q,model,cluster,r0):
    mu,p,v=(np.asarray(model[k],dtype=np.float64) for k in ('mu','p','v'))
    y=np.asarray(q,dtype=np.float64)-mu;a=y@p;e=y-a@p.T;qc=[]
    for c in cluster:
        vc=v[0 if len(v)==1 else c];b=e@vc;s=np.linalg.norm(e-b@vc.T)
        qc.append(np.r_[a,b,s,y@y])
    return np.array(qc)


def summarize(values):
    if len(values)==0:return {'mean':None,'p05':None,'p50':None,'p95':None}
    return dict(zip(('mean','p05','p50','p95'),map(float,[np.mean(values),*np.quantile(values,[.05,.5,.95])])))


def common_reads(root,run,M):
    meta=read_manifest(run/'rq1/metadata.json');dim=fbin(root/'base.fbin').shape[1]
    # Logical read model, not DRAM: one packed full scan, models/IDs, then C0 IDs.
    return meta['persistent_payload_bytes'] + 4*dim + 4*M


def evaluate(a):
    root=Path(a.data);run=root/f'seed_{a.seed}';models=run/'probes'
    split=np.load(root/'split_ids.npz');ids=split['base_ids'];queries=fbin(root/'queries.fbin')
    qlist=np.arange(512) if a.split=='calibration' else split['test_positions']
    if a.split=='test':
        if not (run/'selection_frozen.json').exists():raise ValueError('Freeze selection before test')
        frozen=read_manifest(run/'selection_frozen.json')
        for name,expected in frozen['frozen_artifacts'].items():
            if sha(run/name)!=expected:raise ValueError('Frozen input changed: '+name)
    for name,expected in read_manifest(models/'fit_manifest.json')['artifacts'].items():
        if sha(models/name)!=expected:raise ValueError('Model/sketch/pair changed: '+name)
    results=run/f'eval_{a.split}';results.mkdir(exist_ok=False)
    labels=np.fromfile(run/'labels.u32',dtype=np.uint32);c0=np.load(run/'candidates.npy',mmap_mode='r')
    gt=np.load(root/'truth/top100.npz');oracle=np.load(root/'truth/all_d2.npy',mmap_mode='r')
    fm=read_manifest(models/'fit_manifest.json');r0=fm['r0'];dim=queries.shape[1];N=len(ids);M=c0.shape[1]
    frontbytes=common_reads(root,run,M);rows=[];memory=[];energy_summary=[]
    for r in (16,32):
        for method in METHODS:
            for precision in ('reference','fp32'):
                m=np.load(models/f'{method}_r{r}_{precision}.npz')
                sx=np.load(models/f'{method}_r{r}_{precision}_sketch.npy',mmap_mode='r')
                refm=np.load(models/f'{method}_r{r}_reference.npz')
                refs=np.load(models/f'{method}_r{r}_reference_sketch.npy',mmap_mode='r')
                modelbytes=sum(m[k].nbytes for k in ('mu','p','v'))
                budget=4*N*(r0+r+2)+4*dim*r0+4*16*dim*r+4*dim
                persistent=modelbytes+sx.nbytes
                memory.append({'method':method,'rank':r,'precision':precision,'model_payload_bytes':modelbytes,'sketch_payload_bytes':sx.nbytes,'extra_persistent_bytes':persistent,'extra_budget_bytes':budget,'budget_pass':persistent<=budget if precision=='fp32' else None,'serialized_model_bytes':(models/f'{method}_r{r}_{precision}.npz').stat().st_size,'serialized_sketch_bytes':(models/f'{method}_r{r}_{precision}_sketch.npy').stat().st_size})
                energies={}
                for qi in qlist:
                    candidates=c0[qi];cl=labels[candidates];active,inverse=np.unique(cl,return_inverse=True)
                    qfields=query_fields(queries[qi],m,active,r0)
                    if precision=='fp32':qfields=qfields.astype(np.float32)
                    xf=np.asarray(sx[candidates],dtype=np.float64);qf=np.asarray(qfields[inverse],dtype=np.float64)
                    t=((xf[:,:r0+r]-qf[:,:r0+r])**2).sum(1)
                    lo=t+(xf[:,-2]-qf[:,-2])**2;hi=t+(xf[:,-2]+qf[:,-2])**2
                    guard=1e-6*(1+xf[:,-1]+qf[:,-1]);tau=float(gt['tau'][qi]);d=np.asarray(oracle[qi,candidates]);neg=d>tau
                    finite=np.isfinite(xf).all(1)&np.isfinite(qf).all(1)&np.isfinite(lo)&np.isfinite(hi)&(xf[:,-2]>=0)&(qf[:,-2]>=0)
                    prune=finite&(lo>tau+guard);keep=~prune
                    # Exact offline verification for every pruned object, not sampled.
                    false=int(np.count_nonzero(prune&~neg))
                    retained=candidates[keep];rd=d[keep];order=np.lexsort((ids[retained],rd))[:10];answer=ids[retained[order]]
                    correct=np.isin(answer,gt['ids'][qi,:10]).sum();candidate_hits=np.isin(gt['ids'][qi,:10],ids[candidates]).sum()
                    # Tie-aware metric requires all strict neighbors and any needed boundary IDs.
                    strict_recovered=int(np.count_nonzero(rd[order]<tau));boundary_recovered=int(np.count_nonzero(rd[order]==tau))
                    tie_hits=strict_recovered+min(boundary_recovered,max(0,10-int(gt['strictly_inside'][qi])))
                    # FP64 residual-pair energy, paired with actual FP32 pruning outcomes.
                    qref=query_fields(queries[qi],refm,active,r0)[inverse];xref=np.asarray(refs[candidates])
                    z2=d-((qref[:,:r0]-xref[:,:r0])**2).sum(1)
                    energy_num=((qref[:,r0:r0+r]-xref[:,r0:r0+r])**2).sum(1)
                    valid=z2>np.finfo(np.float64).eps*np.maximum(1,d)*32
                    energy=energy_num[valid]/z2[valid]
                    splitname='train' if qi<256 else ('calibration' if qi<512 else 'test')
                    energies.setdefault((splitname,'all'),[]).append(energy)
                    nactive=1 if method=='G' else len(active)
                    transform=4*dim+4*dim+4*dim*r0+4*nactive*dim*r
                    readbytes=frontbytes+transform+4*M+4*M*(r0+r+2)+int(keep.sum())*(4*dim+8)
                    row={'seed':a.seed,'split':'train' if qi<256 else ('calibration' if qi<512 else 'test'),'query_position':int(qi),'query_id':int(split['query_ids'][qi]),'method':method,'rank':r,'precision':precision,'C0':M,'candidate_recall':float(candidate_hits/10),'zero_recall':int(candidate_hits==0),'recall':float(correct/10),'tie_aware_recall':tie_hits/10,'tau':tau,'tau_zero':int(tau==0),'full_refinements':int(keep.sum()),'hard_negative_kept':int(np.count_nonzero(keep&neg)),'false_prunes':false,'ambiguous':int(np.count_nonzero(finite&(lo<=tau+guard)&(hi>tau))),'known_inside_still_refined':int(np.count_nonzero(finite&(hi<=tau)&keep)),'fallback_invalid':int((~finite).sum()),'shared_fallback_candidates':int(np.count_nonzero(np.isin(cl,fm['shared_fallback_clusters'][str(r)]))) if method in ('H','Q') else 0,'logical_read_bytes':int(readbytes),'transform_logical_reads':transform,'transform_flops':int(4*dim*r0+4*nactive*dim*r),'energy_zero_or_rounding_pairs':int((~valid).sum()),'extra_persistent_bytes':persistent,'budget_pass':int(persistent<=budget) if precision=='fp32' else 1}
                    for key,value in summarize(energy).items():row['energy_'+key]=value
                    for name,mask in [('inside',d<=tau),('near_negative',(d>tau)&(d<=1.1*tau)),('far_negative',d>1.1*tau)]:
                        row[name+'_count']=int(mask.sum());row[name+'_kept']=int(np.count_nonzero(mask&keep))
                        select=mask&valid
                        row[name+'_energy_mean']=float(np.mean(energy_num[select]/z2[select])) if select.any() else None
                        energies.setdefault((splitname,name),[]).append(energy_num[select]/z2[select])
                    rows.append(row)
                for (splitname,stratum),parts in energies.items():
                    values=np.concatenate(parts)
                    energy_summary.append(dict(method=method,rank=r,precision=precision,split=splitname,stratum=stratum,pair_count=len(values),**summarize(values)))
                print('evaluated',a.data,a.seed,a.split,method,r,precision,flush=True)
    with open(results/'per_query.csv','x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(results/'memory.json',memory)
    dump(results/'energy_summary.json',energy_summary)
    dump(results/'manifest.json',{'stage':'oracle_radius_A','host_maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'device_used_bytes':0,'device_reserved_bytes':0,'reference_and_oracle_workspace_not_deployable':True,'front_logical_read_bytes_per_query':frontbytes,'all_pruned_candidates_audited':True,'per_query_sha256':sha(results/'per_query.csv'),'model_manifest_sha256':sha(models/'fit_manifest.json'),'candidate_manifest_sha256':sha(run/'candidate_manifest.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['fit','evaluate']);p.add_argument('--data',required=True);p.add_argument('--seed',type=int,required=True,choices=[17,29,43]);p.add_argument('--split',choices=['calibration','test'],default='calibration')
    a=p.parse_args();{'fit':fit,'evaluate':evaluate}[a.phase](a)
