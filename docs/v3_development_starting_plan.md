# Heat3D v3 Development Starting Plan

This is a planning document only. V3 implementation should start from the
latest `main` after v2 is merged, in a new branch/worktree such as
`research/v3-graph-coverage`.

## Planned Order

| priority | task | completion gate |
|---|---|---|
| P0 | Graph coverage A/B | Quantify physical-to-regional and regional-to-output coverage on fixed 1/4/16-sample cases. |
| P1 | Correct 3D support-radius and radius-cap logic | Remove isolated-node coverage defects without changing the dataset protocol. |
| P2 | Validate graph fixes on 4/16-sample memorization | Demonstrate a clear fitting improvement before full-data runs. |
| P3 | Audit the RIGNO model path and decoder | Isolate graph-to-output limitations against the pointwise baseline. |
| P4 | Run upstream RIGNO alignment tests | Verify graph construction, normalization, batching, and model-call semantics. |
| P5 | Evaluate a pointwise skip or local decoder | Use only if graph coverage fixes do not recover small-sample fitting. |
| P6 | Run a full-dataset controlled comparison | Proceed only after small-sample gates pass; keep `valid_stress` diagnostic only. |
| P7 | Revisit objective/loss design | Defer until graph/model-path limitations are addressed. |

V3 should preserve the v2 data protocol, stratified split, controlled-training
system, final/best export, and diagnostics contracts unless an explicit A/B
test requires a scoped change.
