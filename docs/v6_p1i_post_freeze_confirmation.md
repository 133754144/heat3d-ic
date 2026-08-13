# P1i post-freeze confirmation

所有主比较均输出 240825 nodes；test/sealed 未访问，未训练。

## Unified performance

| strategy | PG % | raw K | fresh median/p95 s | resident median/p95 s | streaming added median/p95 s | throughput samples/s |
|---|---:|---:|---:|---:|---:|---:|
| E16384_reconstruction | 2.702277119191781 | 2.2848024229282844 | 3.133268/3.544013 | 0.007229/0.008303 | 3.180663/3.588221 | 0.340977 |
| U_direct240825 | 2.81883076613177 | 2.3181013590341357 | 0.570639/0.601810 | 0.057747/0.106274 | 0.610898/0.642747 | 0.239973 |
| E240825_direct | 3.0670975720863662 | 2.488381921071613 | 2.403566/2.600600 | 0.093565/0.094630 | 2.445728/2.643123 | 0.411038 |
| FVM240825 | reference | reference | 1.646838/1.888548 | 1.626559/1.886173 | 18.470985/34.134949 | 0.899688 |

FVM resident-core 是 prepared-system solve-only，不是 E2E。batch-scale marginal 是预注册估计量；true streaming 行才是 persistent service 的 submit-to-result 实测。

## Independent valid96 confirmation (three-seed mean ± std)

| route | PG % | raw K | source K | peak K | interface K |
|---|---:|---:|---:|---:|---:|
| E16384_reconstruction | 3.356615 ± 0.061438 | 2.456785 ± 0.047238 | 4.095412 ± 0.079811 | 5.312322 ± 0.162981 | 0.419341 ± 0.036215 |
| E240825_direct | 4.267199 ± 0.028495 | 2.882178 ± 0.075719 | 6.136646 ± 0.127421 | 10.323850 ± 0.397988 | 0.573623 ± 0.051114 |

Paired bootstrap 95% CI 详见机器可读 JSON。valid96 在任何结果查看前由 formal valid128 减 frozen valid32 唯一确定。

U-direct 在 seed0 valid96 首个不满足 native-domain 覆盖的样本上 fail-closed；因此没有 valid96 accuracy/CI，也未继续 seed1/2。该结果是适用边界，不通过修改 route/packing/graph 绕开。

## Freeze

B/E/U 冻结为并列 inference strategy 定义；正式 production/reference 冻结为 E16384-reconstruction。U-direct 仅保留 valid32 架构证明，不具备 formal-valid population 部署资格。不再返回 valid32 做 route/graph/packing/model 优化。FVM 仍是物理 reference，surrogate 不声明比 FVM 更精确。
