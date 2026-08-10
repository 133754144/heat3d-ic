# V6/P1i full-grid performance closeout

本表不混算不同 timing protocol；空值或 `not_comparable` 表示没有语义匹配的历史状态。

## 240825-node matched summary

| policy | PG % | raw K | source K | peak K | process-cold s | fresh s | warm s | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 3.5665 | 2.9201 | 3.9744 | 3.5175 | 13.6256 | 3.2197 | 0.0862 | 2.256 |
| E | 3.0671 | 2.4884 | 4.7346 | 7.2073 | 10.8995 | 1.5715 | 0.0934 | 4.921 |
| FVM | 0.0696 | 0.0366 | N/A | N/A | 2.3379 | N/A | 1.6207 | 0 |

## Optimization decision

NO-GO for production graph replacement: shared reverse is exact and modestly improves fresh topology, but its independent process-cold bootstrap 95% CI includes zero, so the gain is not clear; GPU tiled is edge-exact but slower. Original B/E remain frozen production comparisons.

- GPU tiled exact：边可做到完全一致，但图构建更慢，NO-GO。
- P2R/R2P reverse reuse：仅在 fresh 路径有小幅收益；独立 process-cold bootstrap CI 含 0，因此未推广。
- padding/bucketing：未进入，因为前两步已定位主要瓶颈且无进一步预注册收益依据。

## Original first-inference bottleneck

- B@240825：连续 process-cold 14.052 s；CUDA init 0.601 s，graph 5.780 s，packing/padding 1.543 s，JIT+首 forward+sync 5.175 s。
- E@240825：连续 process-cold 10.930 s；CUDA init 0.450 s，graph 3.323 s，packing/padding 1.628 s，JIT+首 forward+sync 4.655 s。
- direct full-grid output 与 solver grid 同序，reconstruction map/build/apply 均为 0；同步已包含在首 forward 或 warm forward span。

## Timing interpretation

- process-cold：独立进程连续 wall-clock；与 FVM process-cold 比较。
- fresh-topology：进程已驻留、重新构图；历史 FVM 没有语义相同的 unseen-topology 状态，因此 speedup 标记 N/A。
- warm-resident：固定 support/graph/JIT 重复分析；只与 FVM fully-cached lower bound 比较。
- 历史 A/B/E 行保留各自 provenance，不与本轮 full-grid timing 合并统计。

## GO / NO-GO

- B/E@240825 implementation feasibility 均 GO；但 process-cold 相对 full FVM 仅 0.172x / 0.214x，不构成 cold/new-case production speedup。
- 固定 support 的 warm repeated-analysis 为 18.79x / 17.36x；该结论只适用于 fully-cached 语义。
- E 的 full-grid PG/raw 优于 B，但 source/peak 更差且 VRAM 更高；不据此替换既有 B@8192 推荐分辨率。
- graph optimization 总结为 NO-GO：不推广 shared-reverse、GPU tiled 或 padding/bucketing，不修改 frozen graph policy。

完整逐分辨率表见 `docs/v6_p1i_full_grid_performance_closeout.csv`。
