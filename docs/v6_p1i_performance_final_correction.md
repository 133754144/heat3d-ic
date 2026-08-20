# V6 performance final correction

Status: **GO**. All rows use the same `in-memory k/q/BC -> synchronized 240825-node result` boundary. Accuracy is seed0 valid96; timing is frozen valid32 with three randomized orders. These populations are not pooled.

## Evidence correction

- E16384 `2.383 s`, E240825 `2.042 s`, and their derived speedups are **deprecated** because shared sample-varying edge shapes triggered CPU-JAX compilation inside the old timing span.
- Historical U-v2 `1.520 s` is relabeled **steady-shape fresh**, not unseen-shape first-hit.
- Native1024 encoder/P2R/R2R/regional nodes remain unchanged. U-v2 only extends output-query R2P coverage through bounded extrapolation and frozen nearest repair.
- Both the historical failed Q2 order and the pre-classification failure are retained. The residual hard gate was not relaxed; the missing host/scheduler span was made explicit, after which all three new randomized orders passed.

## Fixed 240825-output table

| strategy | PG % | raw K | fresh median/p95 s | resident core s | B16->B32 marginal s | Q2 submit/inter-completion/throughput | fresh/Q2 speedup vs FVM |
|---|---:|---:|---:|---:|---:|---:|---:|
| E16384_reconstruction | 3.367 | 2.480 | 0.597 / 0.909 | 0.0065 | 0.469 | 0.922 / 0.447 / 2.156 | 2.84x / 2.25x |
| U_v2_16384_reconstruction | 3.409 | 2.496 | 0.774 / 0.837 | 0.0040 | 0.559 | 1.102 / 0.538 / 1.794 | 2.19x / 1.87x |
| U_v2_direct240825 | 3.461 | 2.436 | 1.301 / 1.500 | 0.0597 | 0.842 | 1.652 / 0.770 / 1.190 | 1.30x / 1.24x |
| E240825_direct_control | 4.238 | 2.939 | 1.135 / 1.364 | 0.0957 | 0.687 | 1.385 / 0.679 / 1.422 | 1.49x / 1.49x |
| FVM240825_reference | — | — | 1.694 / 1.919 | 1.7659 | 1.043 | 2.013 / 0.999 / 0.958 | 1.00x / 1.00x |

`resident_core` is prepared neural inference or FVM prepared-system solve-only and is not E2E. FVM accuracy is reference/—. Qualification, hashes, equivalence, metrics and serialization are outside service timing.

## Interpretation

- The shared graph-runtime fix is graph/edge/hash byte-exact and removes route-specific shape prewarm. `unseen_shape_first_hit` and `steady_shape_fresh` are now reported separately.
- E16384 remains the production/reference route: controlled valid96 error with the best established architecture governance.
- U-v2@16384 and U-v2 direct240825 are parallel characterization strategies. Their results do not reopen valid32 tuning or replace E16384.
- E240825 direct remains an architecture control. FVM remains the physical reference and retains the accuracy/physics-consistency advantage.
- Publication performance claims are allowed for the corrected table because all Neural and FVM Q2 randomized orders passed their unchanged gates. Fresh E2E and Q2 throughput speedups are reported separately; resident ratios are not called E2E speedups.

## Provenance and access

Machine-readable closeout: `configs/heat3d_v6_p1i/v6_p1i_performance_final_correction_closeout.json`. Resolution accuracy: `docs/v6_p1i_performance_final_resolution_accuracy.csv`. No training, test, sealed, checkpoint, dataset, manifest, sampler, graph-policy or accuracy-driven tuning occurred.
