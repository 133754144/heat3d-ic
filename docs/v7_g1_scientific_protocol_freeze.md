# V7 G1 scientific protocol freeze

当前状态是 `qualified_pre_G1`。e200 预算资格已经完成，但本文件仍然不代表
正式 G1 多种子实验已经开始；正式执行保持关闭。

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

## Formal matrix boundary

正式候选基线为六个 registry variants，共享同一 batching、优化和 200-epoch
budget；seed bundle 固定为 `{0,1,2}`。Full/Vanilla 参数量差异为 7.4486%，
超过 5% 预注册阈值，因此第七个 capacity-matched Vanilla 已注册。冻结矩阵为
`7 × 3 × 200 = 21` 个正式 run；任何正式 run 仍未启动，且预算不得根据正式
seed0 结果事后调整。

当前还不能报告 `G1 SCIENTIFIC READY`：generic-uniform 与 volume-only support
的 label-independent support provider/artifact 尚未冻结，no-context/no-scale
也尚未完成实现资格验证。它们被显式保留为 registry delta，并在 provider 缺失
时 fail-closed，不使用隐式 fallback。

机器可读控制面见
`configs/heat3d_v7/v7_g1_epoch_budget_contract.json`、
`configs/heat3d_v7/v7_g1_budget_qualification.json`、
`configs/heat3d_v7/v7_g1_seed_bundle.json`、
`docs/v7_g1_budget_decision_receipt.json` 和本文件同名 JSON。

历史 V1–V6 runner 保留为 read-only historical oracle；V7 stable trainer 是
语义参考实现。test_iid、sealed labels、solver、模型选择和正式 G1 多种子执行
在本冻结阶段均关闭。
