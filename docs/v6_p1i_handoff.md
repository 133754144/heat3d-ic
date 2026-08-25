# V6/P1i handoff

## Objective and phase evolution

V6 began by replacing loosely constrained thermal samples with literature-led,
traceable package physics. P1a–P1f calibrated power, package paths, asymmetric
dual-Robin cooling and deconfounding. P1g supplied the geometry-adaptive V6
layer baseline. P1h rebuilt the same cases on a shared, label-independent
1,024-node source-aware support and became the canonical V6-layer dataset.
P1i then created the continuous-physics formal random-block dataset, added a
240,825-node full-field sidecar, trained three frozen seeds, qualified
cross-resolution inference, and closed the accuracy/performance evidence.

## Completed state

- Canonical V6-layer dataset/model: P1h plus `V6_03_V5best_P1h`; this is a
  distinct role from P1i.
- Canonical formal random-block dataset: P1i formal1024_v1, 768/128/128
  train/valid/test, manifest SHA `f19987c…8514`, perfect interface contact.
- P1i model: V6_06 seed0 reference at point-global-best epoch 559
  (`51567afe…b90e`); V6_07/08 are seed1/2 replications.
- Deployment family: 1,024 source-aware conditioning anchors + E16384 query
  support + layer/interface-aware reconstruction to 240,825 nodes.
- E16384 role: reference operating point frozen by the preregistered accuracy
  non-inferiority + latency-Pareto decision chain, and the only route opened on
  corrected confirmatory test128.
- U-v2 role: bounded output-query extrapolation with frozen R2P nearest repair;
  valid-only parallel/direct inference characterization. It is not the test
  confirmation route.
- Resolution result: 16,384 + reconstruction is the main Pareto family.
  E@32,768 improved point-global RMSE only from 2.702272% to 2.672461% on the
  frozen valid32 resolution audit, while source and peak error worsened and
  latency/VRAM increased; it is exploratory evidence, not a production cell.
- Solver/performance contract: FVM240825 is the reference; the publication
  benchmark compares persistent services under the same in-memory
  `k/q/BC -> synchronized 240825 result` boundary and keeps fresh, resident,
  Q1/Q2, throughput and B16-to-B32 semantics separate.
- Paper evidence entry points:
  `docs/v6_p1i_publication_evidence_summary.md`,
  `docs/v6_p1i_p6a_publication_tables.md`,
  `docs/v6_p1i_error_tail_closeout.md`, and the compact CSV/JSON tables under
  `configs/heat3d_v6_p1i/`.

## Current issues and limitations

- Confirmatory full-field accuracy is strong, but test peak error has a
  high-energy absolute-error tail: peak RMSE 5.726285 K and 3.181269% of the
  frozen 180 K scale; top ten samples explain 95.45% of excess peak SSE versus
  the valid32 mean-SSE reference.
- P1h/P1i use perfect interface contact and cannot learn variable contact
  resistance.
- Evidence is bounded to the frozen generator, material/BC/source ranges,
  support family, checkpoint and reconstruction method. It is not a universal
  geometry/OOD claim.
- Neural timing is preprocessing-bound for the frozen workload. FVM retains
  conservation/physics fidelity and is the accuracy reference.
- `test_iid` is already-opened corrected confirmatory evidence. The separately
  preregistered sealed IID is a post-development final-confirmation set and
  remains ungenerated and unopened. The sealed IID set remains ungenerated and unopened.

## Errors encountered and durable lessons

- Training export once failed after epoch 600 because checkpoint metadata
  referenced an undefined `builder`. Checkpoint persistence was separated from
  diagnostics so metadata failures cannot destroy completed training outputs.
- A checkpoint/prediction reload audit once failed a 0.1 K max-absolute gate
  despite a 0.00199 K RMSE. Reload reporting now distinguishes max drift from
  field RMSE and preserves the original checkpoint.
- Dense physical-to-regional distance construction scaled as an N-by-R matrix
  and blocked high N. `sparse_kdtree_v1`, edge lists and graph caches replaced
  that bottleneck without changing frozen graph semantics.
- CPU/GPU graph topology byte equality was initially treated as a High-N hard
  gate. It was corrected to report-only while same-GPU graph/cache/prediction
  reproducibility remained hard.
- Benchmark attempts exposed padding-envelope overflow, empty-array lifecycle
  statistics, Q2 post-result audit contamination, JAX shape-compile
  contamination and incomplete failure observability. The final benchmark
  contract separates mode-specific schemas, service timing and untimed audits,
  and preserves every failed attempt instead of cherry-picking cells.
- U-v1 failed closed when full-grid query coordinates lay outside the native
  domain. U-v2 documents bounded extrapolation and output-side R2P coverage
  completion without changing native1024 encoder/processor graphs.

## Next steps

V6/P1i scientific development is closed. Remaining work is repository
integration and manuscript preparation using the frozen evidence. A future
contact-resistance dataset, new geometry/OOD study, or sealed IID opening must
start under a new preregistration and must not retroactively alter V6 claims.
The research branch should not be merged wholesale; use the strict allowlist in
`docs/v6_p1i_main_integration_plan.md`.
