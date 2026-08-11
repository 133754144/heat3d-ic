# V6 P1i P5-A5 reconstruction-map closeout

The fixed 240825-node solver mesh is partitioned once into top, bottom, eight
interfaces and nine layer interiors (19 domains). Per-sample support KNN remains
domain-aware and uses the unchanged 2-D/3-D neighbor counts, exact-point rule,
inverse-distance weights and same-layer fallback. Only the partition is reused
and the batched `cKDTree.query` uses all CPU workers.

## Exact-equivalence

All 64 B8192/E32768 valid32 maps passed exact neighbor-index arrays, exact
weight arrays, exact domain codes, exact mapping hashes and identical
partition-of-unity error.

## Timing

The one-time partition preparation cost was 0.010897 s.

| Route | Reference map median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| B8192 | 0.131097 | 0.100834 | 1.300x |
| E32768 | 0.172518 | 0.110648 | 1.559x |
| pooled | 0.160053 | 0.107395 | 1.490x |

The partition cache and parallel KNN are promoted (`GO`). Reconstruction
semantics and all frozen scientific inputs remain unchanged.
