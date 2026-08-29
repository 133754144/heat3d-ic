# V7 G1 scientific protocol freeze

当前状态是 `scientific_protocol_frozen_pre_G1`。e200 预算资格、全部注册变体的
non-publication qualification 以及 Full V6↔V7 semantic anchor 已完成；本文件
仍然不代表正式 G1 多种子实验已经开始，正式执行保持关闭。

## 历史证据边界

直接相关的历史 Full 参考是
`V6_06_V5best_P1i_seed0_reliable_B24`：它使用 e600 warmup-cosine，
point-global 最佳 epoch 为 559，sample-first 最佳 epoch 为 572。seed1/seed2
的曲线尚未在 devbox 定位，因此不推断其最佳 epoch。

e600 的前 200 个 epoch 不能作为 e200 收敛证据：相同 base/min LR 与 warmup
下，e600 在 epoch 200 仍处于 cosine 中段，而完整 e200 schedule 在 epoch
200 已衰减至 min LR；两者的 LR trajectory 不同。

## e200 candidate contract

候选预算为 200 epochs，warmup 10，base LR `5e-4`，min LR `5e-5`，完整
200-epoch warmup-cosine，AdamW、weight decay `1e-4`、gradient clipping `1.0`、
B24/sample-shuffle 和 sample-first valid_iid checkpoint selection 均保持冻结。
每个 qualification run 从随机初始化开始。

Full 与 vanilla RIGNO 各运行 seed0 的完整 e200 schedule，仅用于
`budget_qualification_only`；其结果不进入正式 G1 或 publication evidence。
Full 的 sample-first 最佳为 e173，Vanilla 为 e164；两者均未在 e195--e200
形成新的边界最佳点，且 e200 到达注册的最小 LR。因此 e200 通过预算资格判定，
不需要默认回退 e600，也没有启动 e300。完整诊断和原始 receipt 见
[v7_g1_budget_decision_receipt.json](v7_g1_budget_decision_receipt.json)。

## Support semantics and variant qualification

P1i 的 `retry_deterministic_geometry_only_v1` 实际产生的是
“source-layout-aware block/interface/surface and CV-weighted geometry support”。
其中历史字段 `local_regions` 是 block quota 的别名；native 1024 support 不读取
数值 q、temperature、label、solver output 或 model error，因此不能表述为
source-amplitude-aware support。完整审计见
[v7_g1_support_semantics_audit.md](v7_g1_support_semantics_audit.md)。

`generic_uniform_support` 使用冻结的 `generic_uniform_v1`，从固定 full-field
domain 均匀抽取 1024 个节点；`volume_only_support` 使用冻结的
`volume_only_v1`，在 interior volume 内按 control-volume 权重抽取 1024 个
节点。二者都不读取数值 q、temperature、label、solver 或 model error，且都
完成了 1-epoch `variant_qualification`。`no_context` 使用固定零 24-D global
context 并关闭 FiLM，同时保留 native shape-scale 路径，也完成了同样的资格
验证。三者均为 non-publication observation-only 证据。

`no_scale` 已按 physics-scale-only 语义完成 1-epoch `variant_qualification`；
physics scale 保留，仅关闭 learned residual correction，不是 direct-output
architecture。capacity-matched Vanilla 使用 node/edge latent width 100，也已
完成 1-epoch non-publication qualification。两项结果均不属于 scientific
evidence，receipt 见
[v7_g1_variant_qualification_receipt.json](v7_g1_variant_qualification_receipt.json)。

## Formal matrix boundary

正式候选基线为六个 registry variants，共享同一 batching、优化和 200-epoch
budget；seed bundle 固定为 `{0,1,2}`。Full/Vanilla 参数量差异为 7.4486%，
超过 5% 预注册阈值，因此第七个 capacity-matched Vanilla 已注册。冻结矩阵为
`7 × 3 × 200 = 21` 个正式 run；任何正式 run 仍未启动，且预算不得根据正式
seed0 结果事后调整。

上述七个 registry variant 的 provider/config delta 均已通过 bounded
non-publication qualification；未知 provider 仍 fail-closed，且不存在隐式
fallback。epoch budget、support semantics、variant implementations 和公平性
边界均已冻结。因此控制面达到 `V7_G1_SCIENTIFIC_READY=PASS`，但这只表示
正式科学协议可以打开；正式 G1 multi-seed execution 仍保持关闭，直到用户另行
授权。

## Full V6↔V7 semantic anchor and synchronized profiling

固定 CPU compatibility audit 对 prepared support/input、graph metadata/tensors、
global/native/q-k features、初始参数与 prediction、三步 gradients/updates/
optimizer state/parameter evolution、step loss scalar 和 validation prediction
均达到 exact。此前第 2 步观测到的 `1.9073486328125e-6` 来自比较器漏掉冻结
batch sample-count aggregation boundary；按同一 boundary 重放后差异消失，
没有注册新 tolerance，也没有改变 model/loss。semantic anchor 为 `PASS`；结构化
证据见
[v7_g1_full_p1i_semantic_anchor_receipt.json](v7_g1_full_p1i_semantic_anchor_receipt.json)。

Full 与 canonical Vanilla 的一次 CUDA 彩排已使用实际 post-step
`block_until_ready` 计时边界完成；truth/metrics 不在 step latency 内，publication
和 scientific evidence 均关闭。既有 V6 evidence 和性能数字未改动。详细数据见
[v7_g1_synced_profiling_receipt.json](v7_g1_synced_profiling_receipt.json)。

机器可读控制面见
`configs/heat3d_v7/v7_g1_epoch_budget_contract.json`、
`configs/heat3d_v7/v7_g1_budget_qualification.json`、
`configs/heat3d_v7/v7_g1_seed_bundle.json`、
`docs/v7_g1_budget_decision_receipt.json` 和本文件同名 JSON。

历史 V1–V6 runner 保留为 read-only historical oracle；V7 stable trainer 是
语义参考实现。test_iid、sealed labels、solver、模型选择和正式 G1 多种子执行
在本冻结阶段均关闭。
