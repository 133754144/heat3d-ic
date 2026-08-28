# V7 G1 scientific protocol freeze

当前状态是 `qualification_pending`。本文件冻结 G1 的控制面，不代表正式
G1 多种子实验已经开始。

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
qualification 完成前不冻结最终 epoch budget，也不启动正式三种子矩阵。

## Formal matrix boundary

正式候选基线为六个 registry variants，共享同一 batching、优化和 epoch budget；
seed bundle 固定为 `{0,1,2}`。若参数公平性触发 capacity-matched Vanilla，
再注册第七个 variant。最终 18/21-run 数量必须在 qualification receipt 中由
明确的 e200/e300 decision 绑定，不能根据正式 seed0 结果事后调整。

机器可读控制面见
`configs/heat3d_v7/v7_g1_epoch_budget_contract.json`、
`configs/heat3d_v7/v7_g1_budget_qualification.json`、
`configs/heat3d_v7/v7_g1_seed_bundle.json` 和本文件同名 JSON。

历史 V1–V6 runner 保留为 read-only historical oracle；V7 stable trainer 是
语义参考实现。test_iid、sealed labels、solver、模型选择和正式 G1 多种子执行
在本冻结阶段均关闭。
