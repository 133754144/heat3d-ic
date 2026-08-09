# V6 P1i publication-grade GPU inference closeout

No training, checkpoint/data/binding changes, test/sealed access, batch inference, or full-GPU graph construction occurred.

## Clean B1 production timing

Qualification, hashes, metrics, labels, and serialization are excluded. GPU completion is synchronized.

| N | full PG (%) | oracle floor (%) | new-case median/p95 (ms) | warm-cache median/p95 (ms) | neural-forward median/p95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 4096 | 2.823750 | 1.488771 | 326.045/338.719 | 3.219/3.490 | 3.216/3.665 |
| 8192 | 2.742758 | 1.080746 | 346.083/359.568 | 4.327/4.638 | 4.397/4.715 |
| 16384 | 2.817371 | 0.708853 | 363.402/372.084 | 8.008/8.663 | 8.020/8.547 |
| 32768 | 2.939390 | 0.467668 | 412.983/423.625 | 18.057/18.399 | 18.047/18.477 |
| 65536 | 3.144189 | 0.317622 | 470.207/493.953 | 36.221/36.678 | 36.095/36.456 |

## Graph diagnosis

P1i full-field PG reaches its minimum at N=8192 (2.742758%), while the oracle floor keeps improving. From 8192 to 65536, P1i P2R regional degree falls from 11.084 to 8.083 and source-node P2R degree falls from 4.626 to 2.973. By contrast, P1h P2R regional degree remains 28.225 at 8192 and 27.503 at 32768.

The association is diagnostic rather than causal: the frozen graph method was not changed and no model was retrained.

## P2 reconstruction

All 160 cached-map cases pass CPU-vs-GPU apply equivalence; maximum absolute error is 3.203e-05 K and maximum RMSE is 6.853e-06 K. Warm model+GPU reconstruction latency is effectively the neural-forward latency at every resolution.

## Decision

**Next priority: graph reuse/fixed regional mesh.** The median new-case/warm-cache latency ratio is 45.4x, so cache/group/H2D preparation dominates B1. The frozen accuracy curve and declining regional/source coverage provide associated evidence for stabilizing and reusing the regional representation before optimizing graph build kernels or adding batch inference.

GPU graph optimization is secondary because cached steady-state does not build a graph. Batch inference is deferred because the dominant B1 new-case cost is preparation/cache transfer, not the already-small warm model+reconstruction apply.
