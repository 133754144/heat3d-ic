# V6 P1i P9 performance freeze

P9 uses persistent workers with untimed warmup on the same WSL2 host. Hashing, equivalence checks, metrics and serialization are outside production timing.

## Frozen metrics

| metric | value |
|---|---:|
| fresh B1 median / p95 | 0.851959 / 1.806525 s |
| warm resident B1 median | 0.013642 s |
| marginal added-case median | 0.323254 s |
| 2xB16 valid32 throughput | 2.624 samples/s |
| B32 valid32 throughput | 2.934 samples/s |
| saturated FVM P=2 | 0.863 samples/s |
| B32 neural/FVM throughput | 3.399x |

All registered preprocessing backends reproduce the complete anchor/query groups, inputs, graph, physics/context, selected CV and reconstruction map hashes exactly. The route is frozen after this valid32 closeout; no test/sealed role was accessed and no training occurred.
