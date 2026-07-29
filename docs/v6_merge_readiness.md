# V6 merge readiness

Status: **ready for final audit; not merged**.

## Frozen scope

- Canonical dataset: `heat3d_v6_p1h_shared_support1024_v0`.
- Canonical model configuration: `V6_03_V5best_P1h`; seed0 e111 is the
  reference checkpoint and seeds 1/2 are replications.
- `V6_04_V5best_P1h_DualAttention` remains an ablation.
- Production workflow: 1024 source-aware conditioning anchors, Anchor-derived
  context/scale, sparse KD-tree edge-list cached graphs, and frozen
  layer/interface-aware full-field reconstruction.
- Resolution roles: 4096 default/hotspot-oriented, 8192 balanced full-field,
  and 16384 highest IID-average full-field accuracy. 32768 remains
  experimental.

## Governance status

- The opened `test_iid` is a corrected confirmatory holdout and was not used
  for model or resolution selection.
- The wrong-ladder temporary results are registered by SHA, excluded, and were
  not used for selection or formal reporting.
- The FVM comparison is a legal structured-FVM mesh sensitivity study with
  nonmatched degrees of freedom.
- The hard role is a preregistered 16-case IID stress subgroup within the
  already-opened corrected confirmatory holdout, not OOD, and was not used for
  selection or tuning.
- Canonical P1h has no registered distribution-shift OOD role; no OOD labels
  were opened.

## Merge gate

The branch is suitable for final audit after the deterministic governance,
dataset, production bundle, syntax, structured-data, hash, and diff checks
listed in the machine-readable readiness record pass. The final sweep checked
10,240 dataset files and 1,024 full-field rows; the production bundle contained
16 bound files, six graph caches, and six reconstruction maps. This report does
not authorize or perform a merge to `main`.

Machine record:
`configs/heat3d_v6/v6_merge_readiness.json`.
