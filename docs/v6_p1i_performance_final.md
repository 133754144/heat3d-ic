# V6/P1i final performance closeout

Accuracy 复用冻结 valid32；P3 仅测静态缓存 production timing 与等价性。hash/equivalence/metrics 均不在 production span。

| route | PG % | raw K | source K | peak K | cold s | known-new-physics ms | replay ms | VRAM GiB | known speedup vs FVM | cache speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B8192_recon | 2.740 | 2.459 | 3.588 | 3.479 | 10.308 | 14.893 | 2.664 | 0.088 | 113.237x | 3.710x |
| E32768_recon | 2.672 | 2.209 | 3.904 | 4.052 | 9.851 | 37.683 | 8.731 | 0.391 | 44.753x | 2.148x |
| B240825_direct | 3.567 | 2.920 | 3.974 | 3.518 | 13.626 | 196.632 | 86.233 | 1.514 | 8.577x | 1.344x |
| E240825_direct | 3.067 | 2.488 | 4.735 | 7.207 | 10.899 | 324.697 | 93.360 | 3.888 | 5.194x | 1.210x |
| FVM240825 | 0.070 | 0.037 | 0.296 | 0.510 | 2.338 | 1686.431 | 1620.697 | 0.000 | 1.000x | N/Ax |

## Static cache equivalence and decision

- 四条路线的 CPU model-group tree、graph metadata/hash 均精确一致；预测差异均低于冻结 same-GPU replay envelope。
- 静态缓存四条路线均有明确收益，GO：运行时仅更新 k/q/BC/context/scale/QK；graph/map/structural packing 常驻。
- B8192 persistent JAX cache：首次 compile 5.375s，cache-hit 1.269s；compile 不进入 known-support steady span。
- hybrid GPU-tiled 未继续；true unseen-topology GPU builder 留待下一阶段。

## Pareto conclusions

- Process-cold：B8192-recon 严格支配 B240825-direct；E32768-recon 严格支配 E240825-direct。FVM240825 在 accuracy 与 cold latency 上仍占优。
- Known-support/new-physics：B8192-recon 最快；E32768-recon 提供更低 PG/raw/interface，但 source/peak 与 latency/VRAM 更差，二者均在 Pareto 前沿。
- 两条 direct full-grid 路线同时被各自 reconstruction 路线支配，不作为默认生产路线。
