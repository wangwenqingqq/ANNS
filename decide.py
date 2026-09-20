"""Freeze calibration choices and report oracle-radius A only. Never launches B."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from prepare_truth import dump, fbin, sha
from probes import common_reads


def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))

def sums(rs):
    return {'recall':float(np.mean([float(x['recall']) for x in rs])),
            'hard_negative_kept':sum(int(x['hard_negative_kept']) for x in rs),
            'logical_read_bytes':sum(int(x['logical_read_bytes']) for x in rs),
            'full_refinements':sum(int(x['full_refinements']) for x in rs),
            'false_prunes':sum(int(x.get('false_prunes',0)) for x in rs),
            'budget_pass':all(int(x.get('budget_pass',1)) for x in rs),
            'query_n':len(rs)}


def selected(rs,method,r,split):
    return [x for x in rs if x['method']==method and int(x['rank'])==r and x['precision']=='fp32' and x['split']==split]


def rq_row(qi,root,run,candidates,retained,truth,ids,split,rank,bits,extra):
    dist=np.asarray(truth[qi,retained]);tau=float(np.partition(truth[qi],9)[9])
    gt=np.load(root/'truth/top100.npz');order=np.lexsort((ids[retained],dist))[:10];answer=ids[retained[order]]
    hits=int(np.isin(answer,gt['ids'][qi,:10]).sum())
    dim=fbin(root/'base.fbin').shape[1];M=len(candidates);front=common_reads(root,run,M)
    pd=read(run/'rq1/metadata.json')['padded_dim']
    extra_reads=0 if bits==1 else M*(pd*(bits-1)//8+8+4)+4*pd
    return {'query_position':int(qi),'query_id':int(split['query_ids'][qi]),'split':'train' if qi<256 else ('calibration' if qi<512 else 'test'),
            'method':'FULL' if bits==1 else 'RQ_PLUS','rank':rank,'precision':'fp32','bits':bits,'M1':len(retained),'C0':M,
            'candidate_recall':int(np.isin(gt['ids'][qi,:10],ids[candidates]).sum())/10,'recall':hits/10,
            'tie_aware_recall':(int(np.count_nonzero(dist[order]<tau))+min(int(np.count_nonzero(dist[order]==tau)),max(0,10-int(gt['strictly_inside'][qi]))))/10,
            'full_refinements':len(retained),'hard_negative_kept':int(np.count_nonzero(dist>tau)),
            'logical_read_bytes':int(front+extra_reads+len(retained)*(4*dim+8)),'extra_persistent_bytes':extra,'budget_pass':1,
            'false_prunes':0,'false_prunes_applicability':'not_bound_pruning','ambiguous':None,'seed':int(run.name.split('_')[1])}


def baselines(a):
    root=Path(a.data);run=root/f'seed_{a.seed}';split=np.load(root/'split_ids.npz');ids=split['base_ids']
    c0=np.load(run/'candidates.npy',mmap_mode='r');truth=np.load(root/'truth/all_d2.npy',mmap_mode='r');gt=np.load(root/'truth/top100.npz')
    N=len(ids);D=fbin(root/'base.fbin').shape[1];r0=16 if D==128 else 64;M=c0.shape[1]
    f0=read(run/'rq1/metadata.json');choices={};grid=[]
    for r in (16,32):
        budget=4*N*(r0+r+2)+4*D*r0+4*16*D*r+4*D
        feasible=[]
        for bits in range(2,10):
            expected=N*(f0['padded_dim']*(bits-1)//8+8)
            if expected>budget:continue
            path=run/f'rq{bits}'
            if not (path/'metadata.json').exists():raise ValueError(f'Mandatory supported RQ width missing: {bits}')
            meta=read(path/'metadata.json');extra=meta['persistent_payload_bytes']-f0['persistent_payload_bytes']
            if extra>budget:continue
            if sha(path/'codes.bin')!=sha(run/'rq1/codes.bin'):raise ValueError('RQ+ sign codes are not identical to F0')
            scores=fbin(path/'scores.fbin');mgrid=sorted(set(m for m in [10,32,64,128,256,512,1024,2048,4096,M] if m<=M))
            hits=np.zeros(len(mgrid),dtype=np.int64)
            for qi in split['calibration_positions']:
                cand=c0[qi];order=np.lexsort((ids[cand],scores[qi,cand]))
                true=np.isin(ids[cand[order]],gt['ids'][qi,:10]);cum=np.cumsum(true)
                hits+=np.array([cum[m-1] for m in mgrid])
            recall=hits/(256*10)
            for m,rec in zip(mgrid,recall):grid.append({'rank_budget':r,'bits':bits,'M1':m,'recall':float(rec),'extra_bytes':extra})
            ok=[(m,float(rec)) for m,rec in zip(mgrid,recall) if rec>=.999]
            if ok:
                m,rec=ok[0];logical=common_reads(root,run,M)+M*(f0['padded_dim']*(bits-1)//8+8+4)+4*f0['padded_dim']+m*(4*D+8)
                feasible.append({'bits':bits,'M1':m,'recall_calibration':rec,'extra_bytes':extra,'logical_reads_per_query':logical})
        if not feasible:raise ValueError('RQ+ has no feasible calibrated configuration')
        choices[str(r)]=min(feasible,key=lambda x:(x['M1'],x['logical_reads_per_query'],x['extra_bytes']))
    dump(run/'rq_calibration.json',{'choices':choices,'grid':grid,'test_used':False,'rq_logical_work_model':'reuse F0 intermediate inner products; only selected C0 extra-bit scores; offline exporter scores all N for adapter validation; not a timing claim','shared_f0_intermediate_workspace_bytes':4*N})
    print('RQ calibration',a.data,a.seed,choices,flush=True)


def baseline_eval(root,run,splitname):
    split=np.load(root/'split_ids.npz');ids=split['base_ids'];qlist=np.arange(512) if splitname=='calibration' else split['test_positions']
    c0=np.load(run/'candidates.npy',mmap_mode='r');truth=np.load(root/'truth/all_d2.npy',mmap_mode='r');choices=read(run/'rq_calibration.json')['choices'];result=[]
    for r in (16,32):
        ch=choices[str(r)];bits=ch['bits'];scores=fbin(run/f'rq{bits}/scores.fbin')
        for qi in qlist:
            cand=c0[qi];keep=cand[np.lexsort((ids[cand],scores[qi,cand]))[:ch['M1']]]
            result.append(rq_row(qi,root,run,cand,cand,truth,ids,split,r,1,0))
            result.append(rq_row(qi,root,run,cand,keep,truth,ids,split,r,bits,ch['extra_bytes']))
    path=run/f'eval_{splitname}/baselines.csv'
    with open(path,'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    return result


def freeze(a):
    root=Path(a.data);run=root/f'seed_{a.seed}'
    if (run/'selection_frozen.json').exists():raise ValueError('Already frozen')
    rs=rows(run/'eval_calibration/per_query.csv');baseline_eval(root,run,'calibration')
    comparisons=[]
    for r in (16,32):
        controls=[]
        for method in ('G','L','Q'):
            for cr in (16,32):
                if cr>r:continue
                s=sums(selected(rs,method,cr,'calibration'))
                if s['false_prunes']==0 and s['recall']>=.999 and s['budget_pass']:controls.append((method,cr,s))
        if not controls:raise ValueError('No quality-qualified internal control')
        method,cr,c=min(controls,key=lambda v:(v[2]['hard_negative_kept'],v[2]['logical_read_bytes'],v[0],v[1]))
        h=sums(selected(rs,'H',r,'calibration'))
        gain=1-h['hard_negative_kept']/c['hard_negative_kept'] if c['hard_negative_kept'] else None
        bytegain=1-h['logical_read_bytes']/c['logical_read_bytes']
        comparisons.append({'rank':r,'control_method':method,'control_rank':cr,'calibration_hard_negative_gain':gain,'calibration_logical_read_gain':bytegain,'H':h,'control':c})
    eligible=[c for c in comparisons if c['H']['false_prunes']==0 and c['H']['budget_pass'] and c['H']['recall']>=.999]
    if not eligible:raise ValueError('H calibration quality/numerics failed; no test promotion')
    choice=max(eligible,key=lambda c:(-np.inf if c['calibration_hard_negative_gain'] is None else c['calibration_hard_negative_gain'],c['calibration_logical_read_gain'],-c['rank']))
    freeze={'seed':a.seed,'selected':choice,'all_rank_calibration':comparisons,'rq_choices':read(run/'rq_calibration.json')['choices'],'test_used':False,
        'frozen_artifacts':{str(p.relative_to(run)):sha(p) for p in [run/'front_frozen.json',run/'candidates.npy',run/'probes/fit_manifest.json',run/'eval_calibration/per_query.csv',run/'rq_calibration.json']}}
    dump(run/'selection_frozen.json',freeze);print('selection frozen',a.data,a.seed,choice,flush=True)


def bootstrap(h,c,key,seed):
    hv=np.array([float(x[key]) for x in h]);cv=np.array([float(x[key]) for x in c]);assert [x['query_position'] for x in h]==[x['query_position'] for x in c]
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(h),size=(2000,len(h)));den=cv[ix].sum(1);valid=den>0
    if not valid.any():return None
    vals=1-hv[ix[valid]].sum(1)/den[valid];return {'p025':float(np.quantile(vals,.025)),'p975':float(np.quantile(vals,.975)),'resamples':2000,'valid_resamples':int(valid.sum()),'paired_unit':'query ID','duplicate_query_groups_reported_in_input_manifest':True}


def decide(a):
    root=Path(a.data);run=root/f'seed_{a.seed}';frozen=read(run/'selection_frozen.json')
    for path,expected in frozen['frozen_artifacts'].items():
        if sha(run/path)!=expected:raise ValueError('Frozen artifact changed: '+path)
    rs=rows(run/'eval_test/per_query.csv');br=baseline_eval(root,run,'test');r=frozen['selected']['rank'];cm=frozen['selected']['control_method'];cr=frozen['selected']['control_rank']
    hrows=selected(rs,'H',r,'test');crows=selected(rs,cm,cr,'test');h=sums(hrows);c=sums(crows)
    rq=sums([x for x in br if x['method']=='RQ_PLUS' and x['rank']==r])
    gain=1-h['hard_negative_kept']/c['hard_negative_kept'] if c['hard_negative_kept'] else None
    bgain=1-h['logical_read_bytes']/c['logical_read_bytes']
    dominated=rq['recall']>=h['recall']-1e-14 and rq['full_refinements']<=h['full_refinements'] and rq['logical_read_bytes']<=h['logical_read_bytes']
    gates={'quality':h['recall']>=.999,'zero_added_false_prunes':h['false_prunes']==0,'budget':h['budget_pass'],'negative_reduction_20pct':gain is not None and gain>=.20,'logical_reads_reduction_10pct':bgain>=.10,'not_rq_dominated':not dominated}
    allmethods=[]
    for method in ('G','L','Q','H'):
        for rank in (16,32):
            for precision in ('fp32','reference'):
                subset=[x for x in rs if x['method']==method and int(x['rank'])==rank and x['precision']==precision]
                st=sums(subset);st.update(method=method,rank=rank,precision=precision,refinements_p50=float(np.median([int(x['full_refinements']) for x in subset])),refinements_p95=float(np.quantile([int(x['full_refinements']) for x in subset],.95)))
                allmethods.append(st)
    result={'scope':'oracle_radius_A_only','dataset':root.name,'seed':a.seed,'selected_rank':r,'control_method':cm,'control_rank':cr,'H':h,'control':c,'RQ_PLUS':rq,'hard_negative_reduction':gain,'logical_read_reduction':bgain,'hard_negative_ci':bootstrap(hrows,crows,'hard_negative_kept',a.seed),'logical_reads_ci':bootstrap(hrows,crows,'logical_read_bytes',a.seed),'rq_dominates':dominated,'gates':gates,'seed_A_pass':all(gates.values()),'all_probe_results':allmethods,'B':'NOT_STARTED','novelty':'NOT_ESTABLISHED','artifacts':{str(p.relative_to(run)):sha(p) for p in [run/'selection_frozen.json',run/'eval_test/per_query.csv',run/'eval_test/baselines.csv']}}
    dump(run/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='all_probe_results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['baselines','freeze','decide']);p.add_argument('--data',required=True);p.add_argument('--seed',type=int,required=True,choices=[17,29,43]);a=p.parse_args();globals()[a.phase](a)
