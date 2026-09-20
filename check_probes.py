"""Deterministic geometry/serialization regression check, not a workload result."""
import tempfile
from pathlib import Path
import numpy as np
from probes import orthogonal, sketch, query_fields, mix64
rng=np.random.default_rng(37);d=128;r0=16;r=32
p=np.linalg.qr(rng.normal(size=(d,r0)))[0]
v=np.array([orthogonal(rng.normal(size=(d,r)),p,r) for _ in range(3)])
mu=rng.normal(size=d);x=rng.normal(size=(99,d));labels=np.arange(99)%3
x[1]=x[0];q=np.vstack((x[:3],rng.normal(size=(4,d)),mu))
assert np.max(np.abs(v@np.swapaxes(v,1,2)))>0
assert np.max(np.abs(p.T@v[0]))<1e-12
for vc in v: assert np.allclose(vc.T@vc,np.eye(r),atol=1e-12)
with tempfile.TemporaryDirectory() as td:
    path=Path(td)
    for precision in (np.float32,np.float64):
        np.savez(path/'m.npz',mu=mu.astype(precision),p=p.astype(precision),v=v.astype(precision))
        model=np.load(path/'m.npz');sx=sketch(x,model['mu'],model['p'],model['v'],labels).astype(precision)
        np.save(path/'s.npy',sx);sx=np.load(path/'s.npy').astype(np.float64)
        for query in q:
            sq=query_fields(query,model,np.arange(3),r0).astype(precision).astype(np.float64)[labels]
            t=((sx[:,:r0+r]-sq[:,:r0+r])**2).sum(1)
            lo=t+(sx[:,-2]-sq[:,-2])**2;hi=t+(sx[:,-2]+sq[:,-2])**2
            exact=((x-query)**2).sum(1);guard=1e-6*(1+sx[:,-1]+sq[:,-1])
            assert np.all(lo<=exact+guard)
            assert np.all(hi>=exact-guard)
            for tau in (0.,float(np.sort(exact)[9])):
                prune=lo>tau+guard
                assert not np.any(prune&(exact<=tau))
assert np.array_equal(mix64(np.arange(100)),mix64(np.arange(100)))
print('probe geometry, duplicate/tie, FP32 serialization and guard check PASS')
