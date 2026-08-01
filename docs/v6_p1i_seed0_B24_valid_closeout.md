# V6 P1i seed0 B24 valid-only recovery closeout

训练已完成 600 epochs，但 checkpoint metadata 构造阶段因未定义 `builder` 崩溃。
两个 prediction archives 已完整写出，因此以下为冻结 `valid_iid` 的只读复算；
test 与 sealed IID 均未访问。checkpoint 参数未落盘，不能由预测反推恢复。

| archive | epoch | point-global true-RMS | sample-first CV | raw CV RMSE K | amp | corr | shape CV-RMSE | scale log-RMSE | legacy base MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| point_global_best | 542 | 2.109926% | 1.651605% | 1.337249 | 0.999631 | 0.992952 | 0.012125 | 0.012885 | 0.00310311 |
| final | 600 | 2.132820% | 1.665649% | 1.345393 | 1.000834 | 0.992854 | 0.012180 | 0.012892 | 0.00317082 |

## 判定

- `<20%` valid point-global gate: **PASS**。
- 该结论是 valid-only 模型质量诊断，不是 checkpoint 可复现性通过。
- e542 为预注册 point-global selection 对应的 `best_predictions.npz`；e600 为 final。
- 训练日志中的 `raw_rmse_K` 是未加 CV 的逐点 RMSE；表中 `raw CV RMSE K` 为冻结 V5 CV 口径。
- 训练日志 `best=e542/0.0031` 中 epoch 由 point-global 选择，斜杠后数值按既有日志合同显示该 epoch 的 valid base MSE。

## Best 到 final 诊断

- point-global SSE 增加 2.1819%；final 在 61/128 个样本上降低 SSE。
- sample-first relative RMSE 平均增加 0.014044 个百分点；final 在 57/128 个样本上改善。
- top-5 样本 SSE 占比由 28.136% 升至 29.505%，说明轻微后期退化伴随尾部集中。
- final 的 background bias 从接近零转为轻微正偏；shape 与 scale 指标也均小幅退化，因此 e542 优于 e600 的方向一致。
- 用户提供的末段日志未保存到注册 log 路径；其 e600 train/valid base MSE 分别为 0.000121/0.00317，约 26.2 倍 generalization gap，但绝对 valid 误差仍很低。

## 工件状态

- predictions: saved and SHA256-bound
- params checkpoints: missing because crash preceded checkpoint writes
- run_config/loss_summary: missing for the same reason
- retraining: not performed
