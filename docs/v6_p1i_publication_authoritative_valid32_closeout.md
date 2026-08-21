# V6 Authoritative valid32 Publication Benchmark Closeout

## Decision

- `pre_measurement_seal = GO`
- `ready_for_authoritative_valid32 = GO`
- `publication_timing_freeze = NO_GO`

The formal entry verified the frozen protocol, E runner, U runner, unified
harness, collector, checker, and auxiliary artifact-manifest SHA values before
creating the formal output. The two-case FVM sanity had already established
true in-process persistent P1 Fresh/Q1 execution without ProcessPool or IPC.

## Fail-closed result

The authoritative matrix stopped in its first cell:

| Route | Seed | Mode | Intended samples | Completed cell | Failure |
|---|---:|---|---:|---|---|
| E16384 + reconstruction | 20260814 | serial | 32 | no | `p2r_edge_indices: 2911 exceeds frozen target 2905` |

The failure occurred while preparing the first timed sample's anchor graph.
It is a frozen static-padding capacity hard-gate failure, not an accuracy or
latency result. No route result JSON was produced. The other 29 independent
lifecycle cells were not started.

Per the preregistered contract, the cell was not rerun, the envelope was not
expanded, and the runner/graph/route semantics were not changed. A preliminary
shell launch using an incorrect explicit commit string had stopped before the
formal entry and created no output directory; it is not a measurement attempt.

## Collector and checker

The frozen collector was executed and correctly rejected the failed raw matrix
with `authoritative raw matrix did not pass hard gates`; therefore no latency,
speedup, bootstrap interval, or publication table was generated. The frozen
pre-measurement checker still passes, confirming that the seal itself did not
drift. The authoritative-result checker is ineligible because a 30-cell passed
matrix does not exist.

## Scope and interpretation

No training, accuracy tuning, test/sealed access, checkpoint modification,
dataset modification, graph-policy modification, or route change occurred.
No performance claim can be made from this run. Any future padding-envelope
repair must be a new explicit protocol amendment followed by a new complete
measurement; it cannot be retroactively inserted into this frozen attempt.

Evidence is preserved under
`configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_69be908_raw/`
and bound by
`configs/heat3d_v6_p1i/v6_p1i_publication_authoritative_valid32_failure_69be908_manifest.json`.
