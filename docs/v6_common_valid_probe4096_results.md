# V6 common-domain valid probe evaluation

The support is a frozen, label-independent set of 4096 original solver nodes.
Only `valid_iid` rows were read for q/temperature inference and metrics; test/hard
roles remained sealed.

| model | epoch | point-global CV % | sample-first CV % | raw CV RMSE K | peak RMSE K | source RMSE K | layer mean/drop K | top/bottom K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V6_02_V5best | 406 | 216.636406 | 218.865342 | 89.415544 | 99.541115 | 82.822451 | 87.915469/5.776467 | 96.483251/94.860895 |
| V6_03_V5best_P1h | 111 | 1.851389 | 1.855461 | 0.764151 | 1.541806 | 1.061957 | 0.216809/0.194917 | 0.415340/0.654476 |
| V6_04_V5best_P1h_DualAttention | 111 | 1.834161 | 1.813109 | 0.757040 | 1.461102 | 1.131307 | 0.205405/0.169444 | 0.445851/0.623619 |

## Conclusion

- Common-domain point-global ranking: V6_04_V5best_P1h_DualAttention < V6_03_V5best_P1h < V6_02_V5best.
- Canonical model candidate remains `V6_03_V5best_P1h`; `V6_04_V5best_P1h_DualAttention` remains an ablation regardless of
this diagnostic ranking.
- These values are a 4096-node solver-support diagnostic and do not replace
the historical 1024-node checkpoint-selection metrics.
