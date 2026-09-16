# V7 G2-P17：DeepOHeat full-minus-valid128 valid-only cohort

训练池为 frozen official pool 减去 Heat3D valid128；仅在被排除的 128 个 valid cases 上做 temperature-space validation。

| seed | train cases | best iter | best [%] | final [%] | wall h | peak device GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99872 | 40000 | 1.120713 | 1.505694 | 0.576 | 0.640 |
| 1 | 99872 | 40000 | 1.190702 | 1.727377 | 0.584 | 0.640 |
| 2 | 99872 | 60000 | 1.128649 | 1.290685 | 0.580 | 0.640 |

## Across seeds

best = 1.146688 ± 0.038323% (sample SD); 
final = 1.507919 ± 0.218354%; 
wall = 0.580 h/seed.

No test_iid, sealed, or DeepOHeat official100 data were accessed; this is not a same-information-budget comparison with Heat3D.
