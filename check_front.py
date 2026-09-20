"""Small deterministic integration gate; synthetic checks are not workload results."""
from pathlib import Path
import argparse
import json
import subprocess
import tempfile
import numpy as np
from prepare_truth import write_fbin, fbin, sha

p=argparse.ArgumentParser();p.add_argument('--binary',default='build-library/front_score');a=p.parse_args()
binary=str(Path(a.binary).resolve())
results=[]
with tempfile.TemporaryDirectory(dir='runs',prefix='adapter-check-') as td:
    root=Path(td)
    for d in (128,960):
        rng=np.random.default_rng(11)
        centers=rng.normal(size=(3,d)).astype(np.float32)
        labels=np.arange(99,dtype=np.uint32)%3
        x=centers[labels]+rng.normal(size=(99,d)).astype(np.float32)
        x[0]=centers[0];x[1]=x[2]  # Zero residual and distinct duplicate IDs.
        q=np.array([x[0],x[1],np.zeros(d),*rng.normal(size=(4,d))],dtype=np.float32)
        write_fbin(root/'base.fbin',x);write_fbin(root/'queries.fbin',q);write_fbin(root/'centers.fbin',centers);labels.tofile(root/'labels.u32')
        for bits in (1,2,4,9):
            directories=[]
            for repeat in (0,1):
                out=root/f'{d}-{bits}-{repeat}';directories.append(out)
                subprocess.run([binary,str(root/'base.fbin'),str(root/'queries.fbin'),str(root/'centers.fbin'),str(root/'labels.u32'),'17',str(bits),str(out)],check=True)
            s=fbin(directories[0]/'scores.fbin')
            assert s.shape==(7,99) and np.isfinite(s).all()
            for artifact in ('scores.fbin','codes.bin','excodes.bin','rotation.bin','rotated_centers.f32','batch_ids.u32','batch_centers.u32'):
                assert sha(directories[0]/artifact)==sha(directories[1]/artifact),(d,bits,artifact)
            m=json.loads((directories[0]/'metadata.json').read_text())
            assert m['code_scored_count_per_query']==99 and m['batch_capacity']==192
            results.append({'dimension':d,'bits':bits,'repeat_bit_identical':True,'zero_residual_duplicate_and_tail_checked':True,'max_normalized_reference_error':m['reference_max_normalized_error']})
print(json.dumps({'status':'PASS','cases':results,'binary_sha256':sha(binary)},indent=2))
