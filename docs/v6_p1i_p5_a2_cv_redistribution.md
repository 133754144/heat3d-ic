# V6 P1i P5-A2 CV redistribution closeout

The frozen B8192 and E32768 adaptive supports were evaluated on all 32
development-valid samples. The reference uses one `cKDTree.query` worker; the
candidate uses SciPy's parallel batched query. Layer membership, nearest-node
tie behavior and `np.add.at` accumulation order are unchanged.

## Exact-equivalence

All 64 route/sample cells passed exact selected-CV arrays and SHA256, exact
nearest assignments, bitwise-equal volume sums and non-increased conservation
error. The production function was also checked against the candidate output.

## Timing

| Route | Reference median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| B8192 | 0.051114 | 0.039551 | 1.292x |
| E32768 | 0.070193 | 0.046199 | 1.519x |
| pooled | 0.066388 | 0.042804 | 1.551x |

Per-layer timing is retained in the machine-readable result. The parallel
batched query is promoted (`GO`). No target, prediction, test or sealed role was
used.
