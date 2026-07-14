# V5 Gate 6B warm-start closeout

Gate 6B warm-start 路线已关闭。此次 closeout 只读取既有
`loss_summary.json` 中的 train/valid_iid 字段，没有运行 evaluator，也没有访问
test/hard；现有 FT 输出目录未被修改或覆盖。

| variant | host | base-MSE best | point-global best | train point-global | valid point-global | result |
|---|---|---:|---:|---:|---:|---|
| FT-L1 `1/1/0.5/1.5` | devbox | e0 | e0 | 4.3067% -> 3.2234% | 24.0753% -> 24.3565% | negative |
| FT-L2 `1.5/0.5/0.5/1.5` | wsl2 | e0 | e0 | 4.3067% -> 3.2247% | 24.0759% -> 24.3531% | negative |

两组 e100 都表现为 train error 下降而 valid point-global 退化，且两个 best
checkpoint 都保留 epoch 0。因此结论严格限定为：

> post-hoc loss reweighting failed；该结果不否定相同 loss 权重的 scratch training。

FT-L0 未运行，状态记录为 `closed_not_run`，不对其性能作任何推断。
