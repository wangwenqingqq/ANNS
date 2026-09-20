"""Stage A input/truth gates. NumPy only; never use a search index as the oracle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def fbin(path, dim=None):
    shape = np.fromfile(path, dtype='<i4', count=2)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('Invalid fbin header')
    n, d = map(int, shape)
    if dim is not None and d != dim:
        raise ValueError('Unexpected dimension')
    if Path(path).stat().st_size != 8 + 4*n*d:
        raise ValueError('Invalid fbin size')
    return np.memmap(path, mode='r', dtype='<f4', offset=8, shape=(n, d))


def write_fbin(path, x):
    with open(path, 'wb') as f:
        np.asarray(x.shape, dtype='<i4').tofile(f)
        np.asarray(x, dtype='<f4').tofile(f)


def groups(x):
    out = {}
    for i, row in enumerate(x):
        row = row.copy()
        row[row == 0] = 0  # Numeric equality includes +0 == -0.
        out.setdefault(row.tobytes(), []).append(i)
    return list(out.values())


def split_queries(x, rng, counts=(256, 256, 488)):
    gs = groups(x)
    remaining = [gs[i] for i in rng.permutation(len(gs))]
    split_ids = []
    # Exact subset-size selection; never split an identical-vector group.
    for target in counts:
        paths = {0: ()}
        for i, g in enumerate(remaining):
            for total, chosen in sorted(list(paths.items()), reverse=True):
                new = total + len(g)
                if new <= target and new not in paths:
                    paths[new] = chosen + (i,)
            if target in paths:
                break
        if target not in paths:
            raise ValueError('Exact grouped query split infeasible; revise protocol before running')
        selected = set(paths[target])
        split_ids.append(np.array([j for i in paths[target] for j in remaining[i]], dtype=np.int64))
        remaining = [g for i, g in enumerate(remaining) if i not in selected]
    return split_ids


def direct_distances(q, x):
    diff = q[:, None, :].astype(np.float64) - x[None, :, :].astype(np.float64)
    np.square(diff, out=diff)
    return diff.sum(axis=-1, dtype=np.float64)


def top_ids(dist, ids, k=100):
    # Include all partition-boundary ties before applying the stable ID rule.
    kth = np.partition(dist, k-1)[k-1]
    keep = np.flatnonzero(dist <= kth)
    order = np.lexsort((ids[keep], dist[keep]))[:k]
    return keep[order]


def self_test():
    x = np.array([[0., -0.], [-0., 0.], [1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=np.float32)
    splits = split_queries(x, np.random.default_rng(9), (2, 2, 2))
    keys = [{tuple(x[i]) for i in s} for s in splits]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    d = direct_distances(x[:2], x)
    ref = np.array([[math.fsum((float(a)-float(b))**2 for a,b in zip(q,v)) for v in x] for q in x[:2]])
    assert np.array_equal(d, ref)
    ids = np.array([7, 3, 2, 8, 9, 1])
    assert list(ids[top_ids(d[0], ids, 2)]) == [3, 7]
    assert len(np.unique(np.concatenate(splits))) == 6
    print('prepare/truth self-test PASS', flush=True)


def prepare(a):
    started = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    raw, query = fbin(a.base, a.dim), fbin(a.queries, a.dim)
    for label, data in [('base', raw), ('query', query)]:
        for start in range(0, len(data), 4096):
            if not np.isfinite(data[start:start+4096]).all():
                raise ValueError(label + ' contains NaN/Inf')
    if len(raw) < 100000 or len(query) < 1000:
        raise ValueError('Insufficient data for frozen A contract')
    rng = np.random.default_rng(20260920)
    ids = np.sort(rng.choice(len(raw), 100000, replace=False)).astype(np.int64)
    fit = np.sort(rng.choice(len(ids), 32768, replace=False)).astype(np.int64)
    qs = split_queries(query, rng)
    qids = np.concatenate(qs)
    selected = np.asarray(raw[ids], dtype=np.float32)
    qselected = np.asarray(query[qids], dtype=np.float32)
    np.savez(out/'split_ids.npz', base_ids=ids, fit_positions=fit, fit_ids=ids[fit], query_ids=qids,
             train_positions=np.arange(256), calibration_positions=np.arange(256,512), test_positions=np.arange(512,1000))
    write_fbin(out/'base.fbin', selected)
    write_fbin(out/'queries.fbin', qselected)
    manifest = {
        'stage': 'prepared', 'dataset': a.dataset, 'dimension': a.dim,
        'source_base': str(Path(a.base).resolve()), 'source_query': str(Path(a.queries).resolve()),
        'source_base_shape': list(raw.shape), 'source_query_shape': list(query.shape),
        'source_base_sha256': sha(a.base), 'source_query_sha256': sha(a.queries),
        'selected_n': 100000, 'query_n': 1000, 'fit_n': 32768, 'seed': 20260920,
        'base_duplicate_extra_ids': len(selected)-len(groups(selected)),
        'query_group_count': len(groups(query)), 'selected_query_group_count': len(groups(qselected)),
        'split_counts': [len(s) for s in qs],
        'artifacts': {p.name: sha(p) for p in [out/'split_ids.npz',out/'base.fbin',out/'queries.fbin']},
        'elapsed_seconds': time.monotonic()-started,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'machine': platform.machine(),
                    'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                    'maxrss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    }
    dump(out/'input_manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if not k.startswith('source_')}), flush=True)


def truth(a):
    started = time.monotonic()
    out = Path(a.out)
    manifest = json.loads((out/'input_manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        if sha(out/name) != expected:
            raise ValueError('Prepared input hash changed: ' + name)
    x, q = fbin(out/'base.fbin'), fbin(out/'queries.fbin')
    ids = np.load(out/'split_ids.npz')['base_ids']
    td = out/'truth'
    td.mkdir(exist_ok=False)
    matrix = np.lib.format.open_memmap(td/'all_d2.npy', mode='w+', dtype=np.float64, shape=(len(q),len(x)))
    indices = np.empty((len(q),100), dtype=np.int64)
    distances = np.empty((len(q),100), dtype=np.float64)
    ties = []
    maxerr = 0.0
    # Independent scalar fsum oracle cross-check before the full computation.
    small = direct_distances(q[:3],x[:127])
    for qi in range(3):
        reference = np.array([math.fsum((float(u)-float(v))**2 for u,v in zip(q[qi],v)) for v in x[:127]])
        maxerr = max(maxerr,float(np.max(np.abs(reference-small[qi]))))
        if not np.allclose(small[qi],reference,rtol=2e-14,atol=1e-15):
            raise ValueError('Independent FP64 distance oracle mismatch')
        assert np.array_equal(top_ids(small[qi],ids[:127]),top_ids(reference,ids[:127]))
    for qb in range(0,len(q),8):
        for xb in range(0,len(x),4096):
            matrix[qb:qb+8,xb:xb+4096] = direct_distances(q[qb:qb+8],x[xb:xb+4096])
        for qi in range(qb,min(qb+8,len(q))):
            pos = top_ids(matrix[qi],ids)
            indices[qi] = ids[pos]
            distances[qi] = matrix[qi,pos]
            tau = float(distances[qi,9])
            ties.append({'strictly_inside': int(np.count_nonzero(matrix[qi]<tau)),
                         'at_boundary': int(np.count_nonzero(matrix[qi]==tau))})
        if qb % 80 == 0:
            print('truth',a.dataset,'queries',min(qb+8,len(q)),'seconds',round(time.monotonic()-started,2),flush=True)
    matrix.flush()
    del matrix
    np.savez(td/'top100.npz', ids=indices, d2=distances, tau=distances[:,9],
             boundary_count=np.array([r['at_boundary'] for r in ties]),
             strictly_inside=np.array([r['strictly_inside'] for r in ties]))
    result = {'stage': 'truth_verified', 'metric':'squared_l2','arithmetic':'FP64_direct_difference_squared_sum',
              'independent_check':'scalar_python_math_fsum_3_by_127', 'independent_max_abs_error':maxerr,
              'query_block':8,'database_block':4096,'base_n':len(x),'query_n':len(q),'dimension':x.shape[1],
              'boundary_tie_queries':int(sum(r['at_boundary']>1 for r in ties)),
              'top10_cut_tie_queries':int(sum(r['strictly_inside']+r['at_boundary']>10 for r in ties)),
              'elapsed_seconds':time.monotonic()-started,'maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'input_manifest_sha256':sha(out/'input_manifest.json'),
              'artifacts':{p.name:sha(p) for p in [td/'all_d2.npy',td/'top100.npz']}}
    dump(td/'manifest.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['self-test','prepare','truth'])
    p.add_argument('--base');p.add_argument('--queries');p.add_argument('--dataset');p.add_argument('--dim',type=int);p.add_argument('--out')
    a=p.parse_args()
    if a.phase=='self-test':self_test()
    elif a.phase=='prepare':prepare(a)
    else:truth(a)
