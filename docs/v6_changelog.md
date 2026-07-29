# V6 changelog

## Dataset

- Qualified P1h as the canonical shared-support V6-layer dataset while
  retaining P1g as an archived geometry-adaptive baseline.
- Bound 1024 cases, 128 geometry groups, group-locked splits, shared support
  hashes, full-field archive provenance, and label-independent support rules.

## Model evidence

- Froze `V6_03_V5best_P1h` as the canonical configuration with seed0 as the
  reference and seeds 1/2 as replications.
- Retained `V6_04_V5best_P1h_DualAttention` as a noncanonical ablation.
- Limited claims to the P1h source-aware support family.

## Production inference

- Established Anchor-derived source-aware high-resolution inference and
  layer/interface-aware reconstruction.
- Added sparse KD-tree edge-list graph construction, versioned graph caches,
  a production bundle, run-level shared-graph reuse smoke, CPU/GPU timing, and
  legal structured-FVM mesh sensitivity results.
- Standardized resolution terminology: 4096 default/hotspot-oriented, 8192
  balanced full-field, 16384 maximum full-field accuracy.

## Governance and held-out evidence

- Reclassified the opened test split as a corrected confirmatory holdout.
- Registered the wrong-ladder protocol deviation and excluded temporary result
  hashes without changing the frozen decision.
- Preregistered and evaluated once a 16-case input-defined stress subset after
  label-free preflight. It is not distribution-shift OOD and did not affect
  selection or tuning.
- Recorded that canonical P1h has no available OOD role.

No training, checkpoint mutation, sampling change, graph-parameter change, or
reconstruction-method change was performed by this governance closeout.
