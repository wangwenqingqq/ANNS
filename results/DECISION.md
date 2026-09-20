# Stage A decision: NO_GO on both datasets

The experiment completed SIFT128 and GIST960 at 100K with three model seeds, two ranks, all six groups, FP64 reference and serialized FP32 probes. All six selected H configurations fail the predeclared mechanism gates. Do not start B or new CUDA kernels for this fixed linear route.

## Primary independent-query results

Calibration selected H rank16 and local-PCA L rank16 at every point. Every row below uses 488 held-out test queries; seeds are not extra independent query samples. Negative reduction means H retains more hard negatives than L.

| Dataset | Seed | M | H / L mean full refinements | Hard-negative reduction H vs L (95% paired CI) | Logical-read proxy reduction | H Recall@10 | RQ+ bits / M1 / Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sift128 | 17 | 512 | 335.26 / 289.52 | -16.36% [-17.58, -15.18] | -0.6786% | 1.000000 | 4 / 32 / 1.000000 |
| sift128 | 29 | 512 | 333.76 / 288.64 | -16.19% [-17.43, -15.01] | -0.6691% | 0.999795 | 3 / 32 / 0.999590 |
| sift128 | 43 | 512 | 333.57 / 288.94 | -16.00% [-17.21, -14.85] | -0.6619% | 0.999795 | 3 / 32 / 0.999795 |
| gist960 | 17 | 128 | 124.97 / 124.38 | -0.52% [-0.75, -0.31] | -0.0154% | 0.999385 | 3 / 32 / 0.999385 |
| gist960 | 29 | 128 | 125.10 / 124.46 | -0.56% [-0.80, -0.35] | -0.0167% | 1.000000 | 3 / 32 / 1.000000 |
| gist960 | 43 | 128 | 125.08 / 124.41 | -0.59% [-0.85, -0.37] | -0.0176% | 0.999180 | 2 / 32 / 0.998156 |

All main/reference probe test rows have zero added false prunes. H and the frozen L controls meet Recall@10 >= 0.999 at all six points. SIFT seed29 RQ+ has slightly lower recall than H; GIST seed43 RQ+ fails the 0.999 target (0.998156). Those points are retained, not promoted as universal RQ+ dominance, and M1 was not increased after inspecting test. H fails against L independently of RQ+.

## Four separate questions

1. **Reusable structure outside training?** A useful H-specific advantage is not established. H energy capture drops from about 0.567 on train to 0.407 on test for SIFT, and from 0.346–0.358 to 0.092–0.094 for GIST; test capture is below both L and Q. This is consistent with substantial overfitting under the frozen training budget, not proof that residuals have no structure.
2. **Less information/work for the decision?** Not relative to the strongest ordinary control: H needs more hard-negative full refinements in every seed. It misses the required 20% reduction, with paired intervals entirely below zero.
3. **Better than ordinary/mature methods at the same budget?** No. L is stronger for the main criterion, and eligible RQ+ points often require far fewer refinements. Preserve the RQ+ quality failures and the separate allocator caveat.
4. **Full-query benefit?** Unknown, not measured. Stage B was correctly not started. No oracle-radius timing, CPU export duration or logical-byte proxy is a deployable GPU kNN speedup.

## Complete six-group refinement breakdown

Values below are mean full refinements/query on test. Both prescribed ranks are retained; only the calibration-frozen rank16 choices above enter the primary decision.

| Dataset | Seed | Rank | FULL | RQ+ | G | L | Q | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sift128 | 17 | 16 | 512 | 32 | 383.09 | 289.52 | 327.85 | 335.26 |
| sift128 | 17 | 32 | 512 | 32 | 194.97 | 117.72 | 146.53 | 151.51 |
| sift128 | 29 | 16 | 512 | 32 | 382.70 | 288.64 | 326.25 | 333.76 |
| sift128 | 29 | 32 | 512 | 32 | 194.08 | 117.58 | 145.42 | 150.91 |
| sift128 | 43 | 16 | 512 | 32 | 382.70 | 288.94 | 326.64 | 333.57 |
| sift128 | 43 | 32 | 512 | 32 | 194.17 | 117.90 | 145.48 | 149.90 |
| gist960 | 17 | 16 | 128 | 32 | 124.84 | 124.38 | 124.92 | 124.97 |
| gist960 | 17 | 32 | 128 | 32 | 123.14 | 122.61 | 123.84 | 123.97 |
| gist960 | 29 | 16 | 128 | 32 | 124.92 | 124.46 | 125.02 | 125.10 |
| gist960 | 29 | 32 | 128 | 32 | 123.31 | 122.61 | 123.90 | 124.05 |
| gist960 | 43 | 16 | 128 | 32 | 124.92 | 124.41 | 125.02 | 125.08 |
| gist960 | 43 | 32 | 128 | 32 | 123.28 | 122.66 | 123.90 | 124.04 |

## Energy generalization (all C0 residual pairs, reference geometry)

| Dataset | Seed | H train mean | H test mean | L test mean | Q test mean |
|---|---:|---:|---:|---:|---:|
| sift128 | 17 | 0.5668 | 0.4060 | 0.4643 | 0.4166 |
| sift128 | 29 | 0.5663 | 0.4073 | 0.4644 | 0.4176 |
| sift128 | 43 | 0.5687 | 0.4077 | 0.4641 | 0.4170 |
| gist960 | 17 | 0.3577 | 0.0925 | 0.1397 | 0.0974 |
| gist960 | 29 | 0.3455 | 0.0940 | 0.1397 | 0.0986 |
| gist960 | 43 | 0.3496 | 0.0930 | 0.1380 | 0.0971 |

## Correctness, resource and interpretation boundaries

- Original source files and splits are hashed; duplicate vectors retain distinct IDs. Grouped query splits do not leak identical vectors. Independent FP64 direct-difference truth is cross-checked against scalar `math.fsum` on a small sample. All pruned C0 objects are audited, not sampled.
- The final verifier checks 568 artifact hashes across both datasets, every H training pair against its training-query C0, matched cluster IDs, all 16 test groups per seed, and frozen control test quality. Per-dataset `VERIFIED.json` is authoritative over the preliminary payload-only budget flag.
- The rank16 H representation payload is exactly its prescribed extra budget. The diagnostic NumPy/file-backed implementation has extra mapping/allocator capacity, and its measured reserved lower bound exceeds that cap. This is an implementation-resource failure, not a theorem that a packed implementation cannot fit. Fixing this bookkeeping would not repair the independently negative refinement/byte-proxy gates.
- Logical reads use a declared packed-FP32 search-state model, including models/query transforms and common F0 costs. The additional ranking field cost is FP32 distance plus uint32 position. These are not measured DRAM traffic, actual Python/FP64 diagnostic reads, or a GPU memory upper bound. FP64-reference rows use the same counterfactual byte model and are not used for selection.
- RQ+ quality is measured on exactly the shared C0. The offline adapter exports full-library scores for all N; the logical RQ+ model credits reuse of F0 intermediate inner products and extra-bit reads only for C0. It does not claim the exporter has that actual access pattern.
- Three new processes, CUDA-event/wall-query timing, sanitizer, native multibit system and LoRANN RRR system comparisons belong to conditional B. They remain NOT STARTED, not failed or silently substituted.

## Decision and reopen condition

**Stop this fixed, hard-conditioned linear-probe route at A.** Do not rescue it by adding a new tree, GPU kernel, rank, test-tuned threshold, or more engineering. A different mechanism would require a new novelty screen and a fresh protocol showing why it should beat L and mature representations. This result does not reject all ANN, all nonlinear representations, or unrelated dynamic-index mechanisms.

Raw runs retain models, sketches, training-pair lists, full distances, candidate exports, failed RQ+ quality points and the initial verifier receipt. Git contains their hashes plus compressed per-query evidence, not raw vector data or private host paths.
