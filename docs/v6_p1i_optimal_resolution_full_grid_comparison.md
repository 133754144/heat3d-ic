# V6/P1i P1 optimal-resolution + full-grid comparison

所有 accuracy 均为冻结 `seed0 + valid32`；本阶段没有重算历史 accuracy/FVM。模型推理与重建 apply 独立列出。

| route | PG % | raw K | source K | peak K | interface K | model ms | recon ms | cold s | fresh s | warm s | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B8192_recon | 2.7398 | 2.4588 | 3.5875 | 3.4786 | 0.5101 | 2.7468 | 2.4032 | 10.3081 | 0.4418 | 0.0027 | 0.1775 |
| E32768_recon | 2.6725 | 2.2087 | 3.9036 | 4.0515 | 0.3845 | 8.7102 | 2.6569 | 9.8509 | 0.5640 | 0.0087 | 0.4696 |
| B240825_direct | 3.5665 | 2.9201 | 3.9744 | 3.5175 | 0.6562 | 86.0196 | 0.0000 | 13.6256 | 3.2197 | 0.0862 | 2.2561 |
| E240825_direct | 3.0671 | 2.4884 | 4.7346 | 7.2073 | 0.4744 | 93.3570 | 0.0000 | 10.8995 | 1.5715 | 0.0934 | 4.9205 |
| FVM240825 | 0.0696 | 0.0366 | 0.2956 | 0.5097 | 0.0002 | N/A | 0.0000 | 2.3379 | 1.6864 | 1.6207 | 0.0000 |

## Pareto decision

- B8192-recon strictly dominates B240825-direct on all registered accuracy/latency/VRAM fields: **True**.
- E32768-recon strictly dominates E240825-direct on all registered accuracy/latency/VRAM fields: **True**.
- FVM is far more accurate. RIGNO process-cold is slower than FVM process-cold; warm replay speedups apply only to fixed-input resident semantics.
- Fresh graph has no semantically matched FVM unseen-topology state, so no fresh speedup is computed.

## Provenance

- New: only 10 independent B8192 process-cold timing runs.
- Reused: B/E accuracy, fresh/warm timing, E32768/B240825/E240825 process timing, and FVM accuracy/timing.
