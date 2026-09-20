"""Independent final provenance, split, quality and resource admission checks."""
import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import numpy as np
from prepare_truth import dump, fbin, sha


def read(p):return json.loads(Path(p).read_text())
def check(condition,message):
    if not condition:raise ValueError(message)


def check_hashes(directory,manifest):
    count=0
    for name,expected in manifest['artifacts'].items():
        check(sha(directory/name)==expected,'Artifact hash mismatch: '+str(directory/name));count+=1
    return count


def model_capacity(model):
    check(np.core.multiarray.get_handler_name()=='default_allocator','Unknown NumPy allocator')
    lib=ctypes.CDLL(None);lib.malloc_usable_size.argtypes=[ctypes.c_void_p];lib.malloc_usable_size.restype=ctypes.c_size_t
    owners={}
    for name in model.files:
        owner=model[name]
        while not owner.flags.owndata and isinstance(owner.base,np.ndarray):owner=owner.base
        check(owner.flags.owndata and owner.base is None,'Unrecognized model allocation owner')
        pointer=owner.ctypes.data
        capacity=int(lib.malloc_usable_size(pointer))
        check(capacity>=owner.nbytes,'Invalid allocator capacity')
        # Retain owners; otherwise Python may release/reuse a pointer on next iteration.
        owners[pointer]=(owner,capacity)
    return sum(capacity for _,capacity in owners.values())


def verify_dataset(root):
    im=read(root/'input_manifest.json');hash_checks=check_hashes(root,im)
    hash_checks+=check_hashes(root/'truth',read(root/'truth/manifest.json'))
    split=np.load(root/'split_ids.npz');base_ids=split['base_ids'];qids=split['query_ids'];q=fbin(root/'queries.fbin')
    check(len(base_ids)==100000 and len(np.unique(base_ids))==100000 and np.all(np.diff(base_ids)>0),'Base ID contract')
    check(np.array_equal(split['fit_ids'],base_ids[split['fit_positions']]) and len(split['fit_ids'])==32768,'Fit ID map')
    sets=[]
    for phase,count in [('train',256),('calibration',256),('test',488)]:
        positions=split[phase+'_positions'];check(len(positions)==count,'Split size')
        sets.append({tuple(row.tolist()) for row in q[positions]})
    check(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'Identical query crosses splits')
    resource_rows=[];seed_rows=[]
    for seed in (17,29,43):
        run=root/f'seed_{seed}';frozen=read(run/'selection_frozen.json');summary=read(run/'summary.json')
        for name,expected in frozen['frozen_artifacts'].items():
            check(sha(run/name)==expected,'Frozen selection input changed');hash_checks+=1
        hash_checks+=check_hashes(run,read(run/'fit_manifest.json'))
        hash_checks+=check_hashes(run/'probes',read(run/'probes/fit_manifest.json'))
        cm=read(run/'candidate_manifest.json');check(sha(run/'candidates.npy')==cm['sha256'],'C0 changed')
        for name,expected in summary['artifacts'].items():check(sha(run/name)==expected,'Summary input changed')
        for bits in (range(1,10) if root.name=='sift128' else range(1,5)):
            rq=run/f'rq{bits}';hash_checks+=check_hashes(rq,read(rq/'provenance.json'))
            meta=read(rq/'metadata.json');check(meta['code_scored_count_per_query']==100000 and not meta['raw_rerank'],'Front scorer contract')
        candidates=np.load(run/'candidates.npy',mmap_mode='r');labels=np.fromfile(run/'labels.u32',dtype=np.uint32)
        check(candidates.shape==(1000,cm['M']),'C0 shape')
        check(int(candidates.max())<100000,'C0 out-of-range position')
        for row in candidates:check(len(np.unique(row))==len(row),'Duplicate C0 position')
        pairs=np.load(run/'probes/training_pairs.npz');qi=pairs['query_positions'];hp=pairs['hard_positions'];qp=pairs['random_positions'];cl=pairs['cluster_ids']
        check(np.all((qi>=0)&(qi<256)),'Test/calibration pair leakage')
        check(np.array_equal(pairs['query_ids'],qids[qi]),'Pair query mapping')
        check(np.array_equal(pairs['hard_ids'],base_ids[hp]) and np.array_equal(pairs['random_ids'],base_ids[qp]),'Pair base mapping')
        check(np.array_equal(labels[hp],cl) and np.array_equal(labels[qp],cl),'Matched H/Q clusters')
        for pos in range(256):
            mask=qi==pos;check(mask.sum()<=256 and np.isin(hp[mask],candidates[pos]).all(),'H training candidate scope')
        with open(run/'eval_test/per_query.csv',newline='') as f:rs=list(csv.DictReader(f))
        expected=set(map(int,split['test_positions']));false_total=0
        for method in ('G','L','Q','H'):
            for rank in (16,32):
                for precision in ('reference','fp32'):
                    group=[r for r in rs if r['method']==method and int(r['rank'])==rank and r['precision']==precision]
                    check(len(group)==488 and set(int(r['query_position']) for r in group)==expected,'Incomplete/repeated test observations')
                    check(all(int(r['query_id'])==qids[int(r['query_position'])] for r in group),'Test query ID mapping')
                    false_total+=sum(int(r['false_prunes']) for r in group)
        choice=frozen['selected'];control=[r for r in rs if r['method']==choice['control_method'] and int(r['rank'])==choice['control_rank'] and r['precision']=='fp32']
        control_quality=np.mean([float(r['recall']) for r in control])>=.999 and sum(int(r['false_prunes']) for r in control)==0
        check(len(control)==488,'Control observations')
        D=q.shape[1];r0=16 if D==128 else 64
        for method in ('G','L','Q','H'):
            for rank in (16,32):
                model=np.load(run/f'probes/{method}_r{rank}_fp32.npz')
                sketch=np.load(run/f'probes/{method}_r{rank}_fp32_sketch.npy',mmap_mode='r')
                model_bytes=sum(model[k].nbytes for k in model.files);capacity=model_capacity(model)
                page=os.sysconf('SC_PAGE_SIZE');mapped=len(sketch._mmap);reserved=math.ceil(mapped/page)*page
                budget=4*100000*(r0+rank+2)+4*D*r0+4*16*D*rank+4*D
                row={'dataset':root.name,'seed':seed,'method':method,'rank':rank,'raw_common_bytes':4*100000*D,'common_stable_ids_bytes':8*100000,'common_cluster_ids_bytes':4*100000,
                     'sketch_payload_bytes':sketch.nbytes,'model_payload_bytes':model_bytes,'extra_budget_bytes':budget,'model_allocator_usable_bytes':capacity,
                     'sketch_mapping_length_bytes':mapped,'sketch_page_reserved_bytes':reserved,'diagnostic_reserved_lower_bound_bytes':reserved+capacity,
                     'payload_budget_pass':int(model_bytes+sketch.nbytes<=budget),'diagnostic_reserved_budget_pass':int(reserved+capacity<=budget),
                     'device_used_bytes':0,'device_reserved_bytes':0,'stage':'CPU_oracle_A'}
                resource_rows.append(row)
        selected_resource=next(r for r in resource_rows if r['seed']==seed and r['method']=='H' and r['rank']==choice['rank'])
        mechanism_pass=summary['seed_A_pass'] and bool(control_quality)
        seed_rows.append({'seed':seed,'complete_test_groups':16,'query_count_each':488,'all_test_added_false_prunes':false_total,
                          'frozen_control_test_quality_pass':bool(control_quality),'representation_payload_pass':bool(selected_resource['payload_budget_pass']),
                          'diagnostic_reserved_lower_bound_pass':bool(selected_resource['diagnostic_reserved_budget_pass']),
                          'resource_gate_status':'FAIL_LOWER_BOUND_EXCEEDS_BUDGET' if not selected_resource['diagnostic_reserved_budget_pass'] else 'UNKNOWN_UPPER_BOUND_UNMEASURED',
                          'resource_gate_verified_positive':False,
                          'A_admitted':False,
                          'seed_decision':'NO_GO' if not mechanism_pass or not selected_resource['diagnostic_reserved_budget_pass'] else 'UNVALIDATED_RESOURCE',
                          'negative_mechanism_gate_independent_of_resource_overhead':not summary['gates']['negative_reduction_20pct'] or not summary['gates']['logical_reads_reduction_10pct']})
    with open(root/'memory_v2.csv','x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(resource_rows[0]));writer.writeheader();writer.writerows(resource_rows)
    result={'dataset':root.name,'verification':'PASS','recursive_artifact_hash_checks':hash_checks,'split_leakage':False,'training_candidate_scope_checked':True,
            'seed_gates':seed_rows,'A':'NO_GO' if sum(s['seed_decision']=='NO_GO' for s in seed_rows)>=2 else 'INCONCLUSIVE',
            'resource_scope':'Native glibc usable bytes for owning NumPy model blocks plus page-rounded file-backed sketch map; allocator metadata and shared arena overhead excluded, so this is a conservative lower bound. The CPU diagnostic is not a packed GPU index.',
            'B':'NOT_STARTED'}
    dump(root/'VERIFIED_v2.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);a=p.parse_args();verify_dataset(Path(a.data))
