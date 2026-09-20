"""Whitelist small reproducibility evidence; never publish raw paths or vectors."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
from prepare_truth import dump, sha

p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--out',default='results');a=p.parse_args()
runs=Path(a.runs);out=Path(a.out);out.mkdir(exist_ok=False)
for dataset in ('sift128','gist960'):
    root=runs/dataset;dest=out/dataset;dest.mkdir()
    m=json.loads((root/'input_manifest.json').read_text())
    keep={k:v for k,v in m.items() if k not in ('source_base','source_query')}
    dump(dest/'inputs.json',keep)
    shutil.copy2(root/'truth/manifest.json',dest/'truth.json')
    shutil.copy2(root/'VERIFIED_v2.json',dest/'VERIFIED.json')
    (dest/'memory.csv').write_text((root/'memory_v2.csv').read_text())
    for seed in (17,29,43):
        run=root/f'seed_{seed}';sd=dest/f'seed_{seed}';sd.mkdir()
        ship=['summary.json','front_frozen.json','candidate_manifest.json','fit_manifest.json','rq_calibration.json','selection_frozen.json','probes/fit_manifest.json',
              'eval_calibration/manifest.json','eval_calibration/memory.json','eval_calibration/energy_summary.json',
              'eval_test/manifest.json','eval_test/memory.json','eval_test/energy_summary.json']
        for f in ship:
            target=sd/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/f,target)
        for phase in ('eval_calibration','eval_test'):
            for name in ('per_query.csv','baselines.csv'):
                src=run/phase/name;target=sd/phase/(name+'.gz')
                with open(src,'rb') as source,open(target,'wb') as destf,gzip.GzipFile(filename='',mode='wb',fileobj=destf,mtime=0) as compressed:
                    shutil.copyfileobj(source,compressed)
        for bits in (range(1,10) if dataset=='sift128' else range(1,5)):
            rd=sd/f'rq{bits}';rd.mkdir()
            for name in ('metadata.json','provenance.json'):shutil.copy2(run/f'rq{bits}'/name,rd/name)
    summaries=[json.loads((dest/f'seed_{seed}/summary.json').read_text()) for seed in (17,29,43)]
    dump(dest/'dataset_decision.json',{'dataset':dataset,'A':json.loads((root/'VERIFIED_v2.json').read_text())['A'],'passing_seeds':sum(s['seed_A_pass'] for s in summaries),'required_seeds':2,'B':'NOT_STARTED','summaries':[{'seed':s['seed'],'rank':s['selected_rank'],'control':s['control_method'],'hard_negative_reduction':s['hard_negative_reduction'],'logical_read_reduction':s['logical_read_reduction'],'recall':s['H']['recall'],'false_prunes':s['H']['false_prunes'],'rq_dominates':s['rq_dominates']} for s in summaries]})
dump(out/'MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
print('Curated manifests and compressed per-query evidence; no raw vectors or private paths')
