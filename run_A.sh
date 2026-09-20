#!/usr/bin/env bash
set -euo pipefail
: "${SIFT_BASE:?Set original FP32 fbin path}"
: "${SIFT_QUERIES:?Set original FP32 fbin path}"
: "${GIST_BASE:?Set original FP32 fbin path}"
: "${GIST_QUERIES:?Set original FP32 fbin path}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p runs
./build.sh
python3 prepare_truth.py self-test
python3 check_front.py
python3 check_probes.py
for dataset in sift128 gist960; do
  if [[ "$dataset" == sift128 ]]; then
    dim=128; base=$SIFT_BASE; queries=$SIFT_QUERIES; maxbits=9
  else
    dim=960; base=$GIST_BASE; queries=$GIST_QUERIES; maxbits=4
  fi
  data=runs/$dataset
  python3 prepare_truth.py prepare --dataset "$dataset" --dim "$dim" --base "$base" --queries "$queries" --out "$data"
  python3 prepare_truth.py truth --dataset "$dataset" --out "$data"
  for seed in 17 29 43; do
    python3 front_gate.py fit --data "$data" --seed "$seed"
    python3 front_gate.py score --data "$data" --seed "$seed" --bits 1
    python3 front_gate.py calibrate --data "$data" --seed "$seed"
    python3 probes.py fit --data "$data" --seed "$seed"
    python3 probes.py evaluate --data "$data" --seed "$seed" --split calibration
    for ((bits=2;bits<=maxbits;bits++)); do
      python3 front_gate.py score --data "$data" --seed "$seed" --bits "$bits"
    done
    python3 decide.py baselines --data "$data" --seed "$seed"
    python3 decide.py freeze --data "$data" --seed "$seed"
    python3 probes.py evaluate --data "$data" --seed "$seed" --split test
    python3 decide.py decide --data "$data" --seed "$seed"
  done
  python3 verify_results.py --data "$data"
done
python3 curate_results.py
# This runner deliberately never launches B, even if a future A result is positive.
