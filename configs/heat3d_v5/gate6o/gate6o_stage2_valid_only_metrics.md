# Gate 6O Stage 2 valid-only result

V39 在 WSL2、training commit
`cd21cd2e4ec97b94700f6ebcd466040e427b6ae9` 完成 e40。仅访问 `train` 和
`valid_iid`；`test/hard/sealed` 未访问。梯度有限，四类 checkpoint
保存并通过 reload。

| checkpoint | epoch | point-global % | sample-first % | raw CV K | shape CV | scale log |
|---|---:|---:|---:|---:|---:|---:|
| point-global best | 24 | 22.443992 | 20.014144 | 0.161237 | 0.139850 | 0.173165 |
| sample-first best | 0 | 22.727393 | 19.941262 | 0.163244 | 0.139844 | 0.175592 |
| base-MSE best | 24 | 22.443852 | 20.014120 | 0.161235 | 0.139848 | 0.173164 |
| final | 40 | 22.537004 | 19.990279 | 0.161877 | 0.139845 | 0.173717 |

相对 e543 初始化，e24 point-global 改善约 0.2834 个百分点、scale
log-RMSE 改善约 0.00243，但 sample-first 退化约 0.0729 个百分点；
sample-first best 因此正确保留在 epoch 0。V39 未恢复 e231 的
21.9446% point-global，结论为有限的 scale-path 改善与聚合指标权衡，
不是全面晋级。

冻结参数逐叶审计通过：所有非 `global_scale_*` 参数的最大差为严格
`0.0`；e24/final 仅四个 global scale MLP 叶发生变化。

Stage 2 的预注册初始化是 e543。point-global 与 base-MSE checkpoint
均落在 e24；sample-first checkpoint 落在 epoch 0，说明本次校准未在
sample-first 指标上超过初始化。V40/V41 seed1 配对配置仅完成准备，
本轮未启动。
