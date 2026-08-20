# P1i U-v2 timing regression closeout

本报告替代旧 U-v2 `3.229151 s` fresh latency，并废弃所有由 serial trace 推算的 neural Q2。
未训练、未访问 test/sealed，E 架构、checkpoint、dataset、graph policy 均未改变。

## 根因与修复

- U-v1 历史控制恢复为 anchor `0.022513 s`、query `0.090316 s`、fresh `0.586322 s`。
- 回归来自 sample-varying CPU-JAX graph-shape compilation 被计入 production span；JIT/qualification/hash/I/O 现均在 span 外。
- U-v2 uncovered-only nearest repair 与冻结 R2P edge/hash 在 96/96 样本逐字节一致；repair median `2.352→0.500 s`（`4.70×`）。

## Corrected performance

| strategy | PG % | raw K | fresh med/p95 s | resident med s | B16→B32 marginal s | Q2 | fresh speedup vs FVM |
|---|---:|---:|---:|---:|---:|---|---:|
| E16384-reconstruction | 3.367458 | 2.479598 | 2.383024/3.625292 | 0.008487 | 2.643485 | deprecated_serial_trace_not_concurrent | 0.719× |
| E240825-direct-control | 4.237668 | 2.938547 | 2.042407/2.636335 | 0.094279 | 2.046555 | deprecated_serial_trace_not_concurrent | 0.839× |
| U-v2-direct240825 | 3.460815 | 2.435950 | 1.520111/1.727624 | 0.059666 | 1.539729 | not_qualified_one_pass_one_residual_gate_failure | 1.127× |
| FVM240825 | — | — | 1.713886/1.906241 | 1.498608 | 1.111950 | qualified_actual_persistent_process_pool | 1.000× |

## Q2 判定

真实 Q2 首个顺序通过，吞吐 `1.015156 sample/s`；第二顺序触发 residual hard gate，故 **未取得 publication qualification**。该通过值只作 exploratory，不能用于正式 speedup。
E16384/E240825 的旧 Q2 是 serial trace，统一标记 deprecated；FVM Q2 是真实 persistent process pool，继续有效。

## Repair error attribution

- seed0: covered `2.447 K`；repaired-inside `2.382 K`；repaired-outside `1.275 K`。distance/error Spearman `-0.0085`。
- seed1: covered `2.354 K`；repaired-inside `2.272 K`；repaired-outside `1.228 K`。distance/error Spearman `-0.0301`。
- seed2: covered `2.421 K`；repaired-inside `2.357 K`；repaired-outside `1.501 K`。distance/error Spearman `-0.0093`。

repair distance 与误差相关性接近零；outside-repair 体积占比极小，未形成主要误差来源。U-v2 仍仅是 valid96 diagnostic/characterization，E16384 保持 production/reference。
