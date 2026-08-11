# V6 P1i P5-A1 support-ordering closeout

## Contract

- Baseline: `af5795a5`; frozen `valid_iid` 32 samples.
- Scientific inputs, sampler quotas, seed, anchor prefix and graph policy were unchanged.
- The reference uses the historical `list.pop(0)` deficit interleave; the candidate uses immutable queues plus cursors.
- Temperature, prediction, test and sealed roles were not accessed.

## Exact-equivalence gate

All 32 samples passed:

- complete selected-index arrays are bitwise equal;
- complete order SHA256 values are equal;
- each order remains a 240825-node permutation;
- the original ordered 1024-anchor prefix is exact.

## Timing (single sample, WSL2 Ryzen 7 9700X)

| Stage | Reference median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| Full deterministic order | 2.142250 | 0.716121 | 2.991x |
| SHA256 + sort | 0.150960 | 0.150762 | 1.001x |
| Weighted interleave | 1.966415 | 0.539131 | 3.647x |

The hash/sort cost is unchanged, as expected. Removing front-of-list shifts is
the material gain. The cursor implementation is promoted (`GO`).

Machine-readable evidence:
`configs/heat3d_v6_p1i/v6_p1i_p5_a1_support_ordering_result.json`.
