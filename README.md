# ANNS: hard-candidate residual diagnostic

This repository tests a **bounded mechanism hypothesis**, not a new index:
can residual directions learned from a fixed front's surviving training pairs
reduce full-distance work on independent queries beyond ordinary global/local
PCA, query-aware controls, and higher-bit RaBitQ at the same byte budget?

The supplied [protocol](PROTOCOL.md) and [configuration](config.yaml) are
unaltered design inputs. Their thresholds are not experimental results.
[Execution record](EXECUTION.md) documents pinned choices and prior-art limits.

## Gates

1. Original FP32 SIFT128/GIST960, 100K stable database IDs; duplicate vectors under distinct IDs retained.
2. Exact grouped 256/256/488 external query split; FP64 direct-difference truth.
3. Pinned official RaBitQ all-vector scorer; calibrate and freeze M without test.
4. FULL / RQ+ / G / L / Q / H; serialized FP32 summaries, all-prune audit,
   actual memory and logical-byte accounting, paired query bootstrap.
5. Only A-GO permits the oracle-free GPU stage and 1M extension.

**Stage A decision: NO_GO on both datasets (0/3 passing seeds each).**
The hard-conditioned probe keeps more hard negatives than calibration-selected
local PCA in every seed. Both ranks and all six groups are retained; see the
[numeric decision](results/DECISION.md). This rejects only this fixed linear
route, not all ANN methods. Stage B was not started.

## Reproduce the input and scorer gates (Linux x86-64)

Requirements: Python 3 with NumPy, CMake >=3.15, C++17 compiler, OpenMP. No
system/global package installation is performed. Clone the official source:

```sh
mkdir -p vendor runs
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git vendor/RaBitQ-Library
git -C vendor/RaBitQ-Library checkout dd6aa230c49082bf4aad4a5a56f513d87a2084ef
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
./build.sh
python3 prepare_truth.py self-test
python3 check_front.py
python3 prepare_truth.py prepare --dataset sift128 --dim 128 \
  --base "$SIFT_BASE" --queries "$SIFT_QUERIES" --out runs/sift128
python3 prepare_truth.py truth --dataset sift128 --out runs/sift128
python3 front_gate.py fit --data runs/sift128 --seed 17
python3 front_gate.py score --data runs/sift128 --seed 17 --bits 1
python3 front_gate.py calibrate --data runs/sift128 --seed 17
```

Use GIST960/dimension960 and the corresponding inputs for the second dataset.
Repeat model seeds 29 and 43, but never regenerate input splits or truth per
seed. Commands refuse to overwrite existing result directories. Keep failed
runs; diagnose before starting a distinctly named replacement run.

Raw data, full distance matrices, candidate matrices, vendor source, build
products, private host paths, and machine configuration are excluded from Git.
Source/file hashes and curated manifests make evidence traceable. The pinned
third-party library retains its Apache-2.0 license in its original repository;
no vendored library is redistributed here.

A uses the true tenth-neighbor radius for diagnosis. Its CPU/offline duration
is **not** deployable nearest-neighbor search time, nor measured GPU traffic.

## Full stage A and evidence interpretation

After setting all four input-path environment variables above (and GIST
equivalents), `./run_A.sh` executes both datasets, all seeds/ranks, calibration,
freeze, test, independent verification and curation into `runs/curated_results`
(the shipped `results/` evidence remains untouched). It refuses to overwrite
existing evidence and never launches B. Full raw matrices and model/pair files
stay outside Git; their hashes are retained. Compressed per-query CSVs include
all train/calibration/test rows, not a selected favorable subset.

`seed_*/summary.json` is a preliminary payload-only decision. The final
per-dataset `VERIFIED.json` additionally checks actual artifact hashes, test
completeness, frozen-control test quality and resource admission. CPU mapping
and allocator usable sizes are a lower bound: they can prove a budget failure,
not a positive reserved-memory guarantee. Logical read counts are a declared
packed-FP32 work model, not observed Python/FP64 accesses or DRAM bytes. The
FP64 reference byte column uses that same counterfactual model; only FP32 rows
enter selections. No novelty, certified floating-point pruning, or GPU
end-to-end speedup is claimed.
