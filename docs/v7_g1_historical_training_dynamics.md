# V7 G1 历史训练动力学审计

本审计只读取冻结 V6/P1i 的配置和 devbox 上已保存的 `V6_06` train/valid
curve；没有访问 `test_iid` 或 sealed，也没有启动新训练。seed1/seed2 的配置已
核对，但对应 curve/checkpoint output 未在 devbox 定位，WSL2 当前不可达，因此
不推断它们的 best epoch。

## 当前 Full 参考

`V6_06_V5best_P1i_seed0_reliable_B24` 使用 RIGNO、96/96 latent、6 processor
steps、2 层 MLP、native shape--scale joint branch、24-D global context FiLM、
physics-plus-pooled-latent scale head、`sparse_safe_v2` q/k features 和显式
local-condition decoder bypass。loss 是冻结的四项 native objective，权重为
shape 1.5、log-scale 0.5、relative-field 1.0、raw-field 1.0；优化器是 AdamW，
base LR `5e-4`、min LR `5e-5`、warmup 10、gradient clipping 1.0、B24、e600
`warmup_cosine`，随机初始化。

已保存 curve 的 point-global best 为 e559，sample-first best 为 e572，最终
epoch 为 e600。e150→e200 validation loss 变差约 `1.3629e-4`，之后 e200→e300
仍有波动并继续改善；e559 之后到 e600 又回退约 `3.3047e-5`。这说明模型在
e200 附近并不是“已经证明收敛”的状态，同时也说明 e600 后段主要是平台/微小
波动，不能据此给 e200 schedule 下结论。

## 为什么不能截取 e600 前 200 epoch

相同 base/min LR 和 warmup 并不等于相同 schedule。e600 cosine 在 epoch 200
的 LR 仍约为 `3.943447e-4`；新的 e200 cosine 会在 epoch 200 到达
`5e-5`。因此 e600 的前 200 epoch 与完整 e200 run 的优化轨迹不同，前者不能
作为后者的收敛证据。旧 V2/V3 的 e200 结果也只作为不同模型/数据的背景，不作为
P1i Full 的直接证据。

## 冻结的单一候选

资格试验使用完整新 schedule：e200、warmup 10、AdamW、base LR `5e-4`、min
LR `5e-5`、cosine horizon 200、weight decay `1e-4`、clip 1.0、B24、随机
初始化，并以 valid_iid 的 `sample_first_relative_rmse_pct`、最早 epoch tie-break
作为 checkpoint selection metric。历史审计 JSON 记录了精确 milestones、SHA 和
尚未定位的 seed replication 状态。

由于历史 seed1/2 curve 缺失且 Vanilla 没有当前 Full 语义下的历史收敛证据，仍
需要 Full seed0 与 Vanilla seed0 的 `budget_qualification_only` e200 pilot；它们
不计入正式 G1 三种子结果。
