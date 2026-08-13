# P1i U-v2 valid96 performance closeout

所有 accuracy 均为 frozen valid96 diagnostic/characterization；未访问 test/sealed，未训练。E16384 保持 production/reference，U-v2 是并列 direct inference strategy，E240825 仅作 architecture control。

| strategy | PG % | raw K | source K | peak K | interface K | fresh med s | resident med s | Q1 med s | Q2 throughput/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E16384-reconstruction | 3.367458 | 2.479598 | 4.087604 | 5.499716 | 0.386207 | 2.383024 | 0.008487 | 2.424122 | 0.414672 |
| U-v2-direct240825 | 3.460815 | 2.435950 | 4.228456 | 5.725792 | 0.387372 | 3.229151 | 0.056936 | 3.271916 | 0.310909 |
| E240825-direct-control | 4.237668 | 2.938547 | 6.010090 | 10.534402 | 0.526237 | 2.042407 | 0.094279 | 2.087861 | 0.504750 |
| FVM240825 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.713886 | 1.498608 | 1.713886 | 0.904681 |

## 解释

- `fresh_single_case` 与 Q1 是不同新 k/q/BC 的完整 in-memory compute service。
- neural Q2 使用相同固定深度 arrival rule 在不中断的实测逐 case service trace 上重放；FVM Q2 是两个 persistent workers 的直接 wall-clock。worker 数不同，submit latency 与 throughput 均保留，不混称相同 core latency。
- resident FVM 是 prepared-system solve-only，不是 E2E；surrogate 指标是相对 FVM reference field 的误差，绝不表示精度优于 FVM。
