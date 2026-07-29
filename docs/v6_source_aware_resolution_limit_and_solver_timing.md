# V6 source-aware resolution limit and solver timing

All formal model inference used local CPU, valid_iid only, batch=1. No training, test/hard access, or checkpoint mutation occurred.

- First failed resolution: 32768 (graph construction still incomplete after >3820 s; inference and field metrics were not reached)
- Largest formal source-aware resolution: 16384, the highest IID-average
  full-field accuracy mode
- Full physical solver: 240825 nodes; comparison is nonmatched-DOF.

## Support qualification

Every probe retains the original ordered 1024 P1h anchors and uses exact assigned-stratum ratios: source 50%, volume 25%, interface 12.5%, top 6.25%, bottom 6.25%. Selection is label-independent.

| N | min nodes/source box | p05 | median | 9 layers | 8 interfaces | top/bottom |
|---:|---:|---:|---:|:---:|:---:|:---:|
| 1024 | 4 | 5.6 | 10.0 | yes | yes | yes |
| 2048 | 8 | 11.0 | 20.0 | yes | yes | yes |
| 4096 | 17 | 25.0 | 42.0 | yes | yes | yes |
| 8192 | 36 | 51.0 | 82.0 | yes | yes | yes |
| 16384 | 76 | 105.6 | 162.5 | yes | yes | yes |
| 32768 | 174 | 211.6 | 325.0 | yes | yes | yes |

The next doubling (65536) cannot preserve the exact unique-node ratio: the source stratum requires 32768 nodes but has capacity 25308 (shortfall 7460).

## Seed0 upstream-like main path

| N | point-global % | sample-first % | raw K | graph s | warm s | E2E valid128 s | RAM GiB | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1024 | 0.902504 | 0.748941 | 0.386033 | 5.799 | 0.166306 | 31.789 | 1.207 | pass |
| 2048 | 2.549864 | 2.552716 | 1.089586 | 10.064 | 0.179557 | 37.323 | 1.130 | pass |
| 4096 | 3.550181 | 3.572683 | 1.516579 | 26.883 | 0.236518 | 60.353 | 1.874 | pass |
| 8192 | 4.175572 | 4.199658 | 1.783530 | 97.798 | 0.315319 | 143.861 | 1.843 | pass |
| 16384 | 4.486251 | 4.501761 | 1.915706 | 387.828 | 0.576250 | 466.293 | 4.387 | pass |
| 32768 | N/A | N/A | N/A | >3820 | N/A | >3820 | 0.866 observed | fail |

## Three-seed stability

| N | point-global mean±std % | sample-first mean±std % | raw mean±std K |
|---:|---:|---:|---:|
| 1024 | 0.930849 ± 0.044244 | 0.753959 ± 0.017191 | 0.398157 ± 0.018925 |
| 4096 | 3.648163 ± 0.457600 | 3.697738 ± 0.475325 | 1.558435 ± 0.195479 |
| 16384 | 4.603423 ± 0.541206 | 4.667760 ± 0.549923 | 1.965740 ± 0.231104 |

## Anchor-derived context and scale-pooling diagnostic

| N | seed0 point-global % | seed0 sample-first % | seed0 raw K | 3-seed point-global mean±std % |
|---:|---:|---:|---:|---:|
| 1024 | 0.902504 | 0.748941 | 0.386033 | 0.930849 ± 0.044244 |
| 2048 | 1.325161 | 1.238549 | 0.566256 | seed0 only |
| 4096 | 1.451917 | 1.400300 | 0.620235 | 1.574839 ± 0.182044 |
| 8192 | 1.648440 | 1.609235 | 0.704105 | seed0 only |
| 16384 | 1.836206 | 1.806438 | 0.784092 | 1.978708 ± 0.234285 |

## Same-host physical-solver comparison

The frozen 240825-node FVM replayed the archived valid sample with `max_abs_error=0 K`; warm solve mean was `1.651183 s`, cold mesh+assembly+first-solve was `2.900145 s`, and peak RAM was `0.334 GiB`.
The valid source-metadata audit found the smallest source-resolution-legal P1h candidate at `212097` nodes (60x60x56), still far above the model ladder and without a P1h mesh-convergence qualification; the frozen 240825-node solver is therefore the only accuracy-qualified replay comparator.

| model N | model warm batch1 s | warm solver/model ratio | model E2E/sample amortized s | cold solver/model ratio |
|---:|---:|---:|---:|---:|
| 1024 | 0.166306 | 9.929× | 0.248348 | 11.678× |
| 2048 | 0.179557 | 9.196× | 0.291583 | 9.946× |
| 4096 | 0.236518 | 6.981× | 0.471511 | 6.151× |
| 8192 | 0.315319 | 5.237× | 1.123917 | 2.580× |
| 16384 | 0.576250 | 2.865× | 3.642915 | 0.796× |

## Interpretation

- The main path is the upstream-like xinp=xout=N graph with context and joint pooling recomputed from N source-aware nodes.
- The lowest-error frozen inference scheme uses the canonical 1024 source-aware anchors for context and scale pooling, plus nested high-resolution source-aware query nodes. It does not change the frozen checkpoint.
- The upstream-like error increase is dominated by scale/context distribution shift: positive field bias and scale-log error rise with N. Anchor-derived context/scale removes most of it; the smaller remaining shape increase is consistent with regional-graph/query distribution shift.
- 32768 failed on graph-construction runtime before inference, so this is an engineering scaling limit, not evidence of non-finite prediction or point-global accuracy failure.
- Physical-solver speedups are nonmatched-DOF because the only frozen, replay-qualified solver mesh has 240825 nodes; no similarly sized mesh has an accuracy-equivalent mesh-convergence qualification.
