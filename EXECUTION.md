# Execution record — hard_candidate_residual_v1

## Frozen scope (2026-09-20, before measurements)

The supplied `PROTOCOL.md` and `config.yaml` are immutable design inputs.
Their SHA-256 values are in `source_pins.json`; null paths are resolved in
ignored per-run manifests, not by changing the supplied files. No GTS changes.
Stage A is an oracle-radius CPU mechanism diagnostic, not deployable search.
Stage B and new CUDA kernels remain disabled until all A-GO gates pass.

| Experiment | Hypothesis | Contract | Controls | Evidence | Decision |
|---|---|---|---|---|---|
| hcr_20260920_gate0 | Candidate-conditioned normalized residual pairs contain reusable information beyond ordinary local/query-aware structure | SIFT128/GIST960; 100K; independent train/calibration/test; seeds 17/29/43; ranks 16/32; actual FP32 serialization; all-C0 audit | FULL, official RQ+, G/L/Q | Pending | Designed, not measured |

## Cheap novelty screen

- [LoRANN §3.2–4.3](https://arxiv.org/html/2410.18926v1): local supervised
  reduced-rank scoring already trains on queries routed to each cluster and
  approximates query–corpus inner products, including GPU execution. Broad
  claims about query-aware local low-rank representations are subsumed.
- [DADE §3.2](https://www.vldb.org/pvldb/vol18/p812-zheng.pdf): PCA/data-aware
  projections and adaptive distance comparison are established. Ordinary
  variance capture and avoiding full dimensions are not new contributions.
- [ADSampling](https://arxiv.org/abs/2303.09855): reliable progressive distance
  comparison is established; the supplied empirical guard is not a substitute
  for its probabilistic guarantees.
- [Official RaBitQ](https://github.com/VectorDB-NTU/RaBitQ-Library): real
  low-bit quantization with correction factors is the required comparator.

Inference: the proposed normalized, F0-survivor-conditioned residual-direction
objective is not identical to the cited RRR score-loss objective. This limited
screen does NOT establish novelty or rule out other prior work. It permits
only the bounded A falsification experiment. If H fails against L/Q or is
Pareto-dominated by RQ+, do not rescue the direction by writing new kernels.

## Implementation resolutions, pre-test

- NumPy and the C++ standard library plus the pinned official library only.
  No new Python packages or replacement quantizer.
- Stable IDs are original zero-based file row indices. PCG64(20260920)
  selects sorted unique base IDs, then fit IDs and query groups in a declared
  deterministic order. Signed zeros are canonicalized for query equality.
  Identical-query groups must fit wholly in one split; infeasible exact counts
  block the run instead of leaking or dropping difficult queries.
- Truth blocks are exactly up to 8 by 4096. FP64 direct subtraction, square,
  sum; top-100 lexicographic distance/ID. A full FP64 distance matrix may be
  retained solely as an offline audit artifact, outside all scoring inputs.
- CPU-only preflight: x86-64 Xeon Gold 6530, 128 logical CPUs, NumPy 1.26.4;
  limit BLAS/OpenMP threads to 8 and compilation to 4. No GPU, GPU setting,
  foreign process, or existing GTS/TIDE file is modified.
- Source data are read-only original FP32 fbin files. The execution directory
  and raw paths are private run metadata excluded from Git. Retain hashes,
  dimensions, versions, commands, resource peaks, and failures.
- Stop on malformed/nonfinite inputs, split leakage, reference mismatch,
  unexplained scorer mismatch, allocation failure, or disk exhaustion.
  Rollback means stop only the recorded task-owned PID; preserve evidence.

## Claim ledger

| Claim | Status | Allowed wording |
|---|---|---|
| Exact FP64 truth for the selected workloads | unknown | Not generated yet |
| Official all-vector low-bit front is correctly integrated | unknown | Not validated yet |
| H generalizes / reduces hard-negative refinements / bytes | unknown | No evidence yet |
| Mature same-budget advantage | unknown | No evidence yet |
| Complete query speedup | unknown | B not started; no deployable timing claim |

### Front/database model implementation choices (before calibration)

- K-means++ initialization, NumPy PCG64(model seed), one run, at most 100
  Lloyd iterations, relative maximum center shift <= 1e-6. Fit only the
  32,768 database training objects; store centers in FP32 before assignment.
- F0 uses the author's packed 32-object `BatchDataMap` encoder and
  `SplitBatchQuery`/`split_batch_estdist`, HACC enabled, one sign bit per
  dimension. Tail capacity and three FP32 correction factors are counted.
  All 16 clusters/all N objects are scored, without IVF routing or reranking.
- Author FHT/Kac rotator; its native serialized sign bytes are populated by
  `std::mt19937(model seed)` with uniform bytes [0,255] and loaded through the
  public persistence API, then round-trip verified. No upstream source edits.
- Multi-bit adapter uses the author's exact enumeration encoder, rather than
  the faster approximate encoder. It reuses the same one-bit packed layout
  and the author's extra-bit estimator. Supported total bits: 2 through 9.
- Packed scoring is checked against the author's generic implementation and
  an independently accumulated scalar full-code/factor reference. HACC LUT
  quantization allows normalized absolute difference <= 5e-4 on this check;
  this is an adapter check, not a bound/correctness claim for exact pruning.

### Probe evaluation/accounting resolutions (before calibration evaluation)

- Fit H/Q second moments with unit nonzero residual differences and total
  weight one per query; retain both matched training-pair lists. Fixed splitmix64
  of original base/query IDs chooses H pairs without truth labels. Q uses
  PCG64(model seed), uniformly without replacement within each matched cluster.
- Train one ordered rank-32 basis; rank16 uses its leading directions. Project
  out P0 and QR-orthogonalize in order, not arbitrary SVD rotation of the span.
  Shared H/Q sparse fallback is evaluated separately at each rank.
- Main models and sketches are serialized FP32 and read back. A transforms and
  bound arithmetic use FP64 on the serialized values; query sketch fields are
  also rounded to FP32. FP64 reference uses its own FP64 model/sketch. This is
  not evidence about a future GPU's FP32 arithmetic; B must revalidate it.
- Logical reads (not DRAM) charge F0's packed codes, factors, rotated centers,
  rotation state and batch IDs; query input; C0 positions; probe cluster IDs;
  actual candidate summary fields; mu/P0/active local bases and query transform;
  full vector reads plus distance/ID reads for every refinement. Model reads are
  charged once per query/active cluster, not once per candidate. G reads one
  shared extra basis. Output writes, allocator bookkeeping and profiler traffic
  are not renamed as reads. Report persistent payload, reserved capacity,
  serialization bytes and peak diagnostic RSS separately.
- Common stable original IDs (8N bytes) and cluster assignment (4N bytes) are
  persistent across methods. They are not a free extra field for H. All methods
  share raw FP32 vectors and the F0 representation. A also has offline FP64
  oracle/reference workspaces, reported outside deployable index payload.
- Calibration selects the strongest G/L/Q at each rank budget by minimum total
  hard-negative survivors, breaking ties by total logical reads then name.
  H rank is chosen by its largest calibrated relative hard-negative reduction
  against that budget's control, then logical-read gain, then smaller rank.
  Only this frozen rank contributes to the primary per-seed A decision; both
  ranks remain visible as predeclared ablations, not test-set selection.
- RQ+ considers every supported width within each budget. For each width,
  choose the smallest prescribed M1 with calibration Recall@10 >=0.999;
  then select the width with smallest M1, breaking ties by logical reads and
  persistent bytes. No final-test adaptation or replacement of initial misses.

### Accounting and adapter limitations retained in the record

- RQ's offline scorer exporter evaluates all N codes to validate/reuse the
  official component. The same-C0 RQ+ quality result sorts only C0 and refines
  M1. Its logical byte model credits a realizable reuse of F0 intermediate
  inner products (4N workspace bytes) and reads extra codes only for C0.
  That model is favorable to the mature control, not measured memory traffic
  or a claim that the exporter itself touches only M objects. B would require
  the actual candidate-restricted implementation and full timing.
- `budget_pass` in raw A CSV/summary denotes the **persistent representation
  payload** (actual ndarray fields/model bytes), including explicitly stored
  factors and library batch padding. NumPy allocator slack/headers are not
  measured per index; peak process RSS includes oracle/reference scratch.
  Therefore this flag alone is not a fully verified allocator-reserved-memory
  gate and can never by itself admit B. No positive A decision will be issued
  with this resource gate unresolved. Independent failures of the refinement
  and logical-read gates are sufficient to reject A without GPU work.
- No A timing is a deployable query timing. Offline front/model/verification
  durations are separately labeled; GPU used/reserved is zero, no GPU kernels
  launched, no sanitizer claim made for an unimplemented GPU path.

### Post-run evidence hardening (no retuning)

- The final verifier checks every model/sketch/pair blob named by the frozen
  manifests, every score export, C0 uniqueness/hash, group-separated splits,
  training-pair membership and matched clusters, all 16 complete 488-query
  test groups, and the frozen control's test quality. This adds admission
  checks; it does not change models, ranks, scores, guards, or outcomes.
- Resource audit measures glibc usable capacities for model-owning blocks and
  page-rounded file-backed sketch mappings. These are a conservative lower
  bound, not a total allocator upper bound. Above-budget lower bounds prove
  failure; below-budget lower bounds NEVER prove resource admission.
- Raw `reference` rows use the same FP32 **counterfactual** byte model solely
  to pair geometry with the main rows. Their byte column is not measured FP64
  traffic and is excluded from all selection/decision logic. Reference actual
  FP64 payload and serialized sizes are separately listed in memory records.

The refinement logical-read model's extra 8 bytes mean one FP32 distance and
one uint32 candidate position. Base positions are monotone in sorted stable
IDs, so their tie order is identical. This is a proposed packed search-state
accounting model, **not the diagnostic Python/FP64 oracle's actual accesses**.
Oracle/reference I/O, Python copies, and sorting workspaces are separately
outside this mechanism read proxy. Thus the read model can disqualify this
specific budget gate, but cannot establish actual DRAM savings or query speed.
The negative hard-refinement result is independent of choosing 8 versus 16
bytes for candidate ranking state.

The resource verifier v2 supersedes the first SIFT verifier receipt. Original
receipts are retained in the raw run; no geometric measurements are overwritten.
A lower bound exceeding budget is a failure; a lower bound within budget remains
unknown rather than being promoted to a pass.

### Final measured outcome

All six dataset/seed points completed both ranks, FP64 reference and serialized
FP32 probes, FULL, and every budget-eligible official RQ+ width. Calibration
selected rank16 H versus rank16 L at every point. Every point fails the 20%
hard-negative reduction gate: H retains *more*, not fewer, hard negatives.
The matching byte proxy also increases. B was not started.

The GIST seed43 RQ+ test recall is below 0.999. Preserve that failed point;
it is not a qualified winning baseline, and it was not repaired by raising M1
on test. The independent H-versus-L negative result does not depend on RQ+.

Final decision and numeric breakdown are in `results/DECISION.md`; the
per-dataset `VERIFIED.json` is authoritative over preliminary payload-only
`seed_*/summary.json` resource flags. `build_verification.json` binds the clean
pinned source export to a byte-identical post-run rebuild of the scoring binary.
