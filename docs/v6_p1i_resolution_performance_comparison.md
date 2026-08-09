# V6/P1i CPU FVM 与 GPU RIGNO 跨分辨率性能

硬件合同：同一 WSL2 主机的 RTX 5070 GPU 与 Ryzen 7 9700X CPU，B1。所有生产 GPU 区间以 `block_until_ready` 同步；metrics、hash、oracle 与 serialization 均在计时外。

评价域不混淆：RIGNO 在共同 240825-node full field 上重建评价；structured-FVM 则在各自合法结构网格上与 240825 参考场比较。

| N | RIGNO full PG % / raw K | RIGNO new / warm ms | FVM PG % / raw K | FVM new-physics / cached ms | new speedup | warm speedup | VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2.823750 / 2.638153 | 1787.428 / 3.219 | 1.190491 / 1.518489 | 7.612 / 6.258 | 0.004x | 1.944x | 150.7 |
| 8192 | 2.742758 / 2.491700 | 1750.683 / 4.327 | 1.191106 / 1.518798 | 13.810 / 12.109 | 0.008x | 2.799x | 155.2 |
| 16384 | 2.817371 / 2.534275 | 1743.902 / 8.008 | 1.191235 / 1.519003 | 27.974 / 25.245 | 0.016x | 3.152x | 233.4 |
| 32768 | 2.939390 / 2.646894 | 1775.330 / 18.057 | 1.191553 / 1.519157 | 67.232 / 62.941 | 0.038x | 3.486x | 448.4 |

在完整 240825 节点上，CPU FVM new-physics/cached median 为 1.697203/1.669673 s. 历史 1024-RIGNO 加完整场重建路线在 JIT-cached 新拓扑下为 1.763765 s、fully cached 为 0.010003 s；相对 240825 FVM，new-case 为 0.962x，重复已知样本为 166.92x。CSV 明确标记其评价域和较早 timing 协议。

`new_case_speedup_vs_fvm < 1` 表示 sample-varying 新 support 的 RIGNO 端到端更慢。warm-cache ratio 只比较重复已知样本 lower bound。CSV 单列 neural-core/FVM ratio，禁止称为 E2E speedup。

证据来源：A accuracy/graph、GPU clean cached timing、unified direct-wall timing 与 FVM field 全部是历史只读复用。本轮只新增 B/C valid32 accuracy/graph 与同步 candidate warm/new-case span；未重跑 FVM 或 A accuracy。
