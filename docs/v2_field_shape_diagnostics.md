# Heat3D v2 Field-Shape Diagnostics

## P2 目标

P2 的目标是在继续替换 optimizer、扩大模型容量或调整 loss 之前，先补齐只读的 field-shape diagnostics。它只分析已经生成的 recovered-temperature predictions，并转换到 DeltaT 场上计算形状指标，不训练、不改 loss、不改 v1 runner。

## 为什么先做 field-shape diagnostics

v1 diagnostic baseline 已经暴露出 low-DeltaT background bias、final-vs-best 差异和 high-bin 表现差异。单看 RMSE/MAE 不足以判断模型是在学习三维温度场形状，还是只拟合了全局幅值或局部热点。P2 先把场形状诊断固定下来，后续 Optax AdamW、模型容量 ablation、hotspot/peak loss 才有更清晰的对比依据。

## 指标定义

- `true_mean / pred_mean / error_mean`：DeltaT 真值、预测和误差均值。
- `true_std / pred_std`：DeltaT 场标准差。
- `field_variance_ratio`：`var(pred_deltaT) / var(true_deltaT)`，用于观察预测场是否过平滑。
- `field_std_ratio`：`std(pred_deltaT) / std(true_deltaT)`。
- `centered_spatial_correlation`：去均值后的空间相关性，关注场形状是否一致。
- `uncentered_cosine_similarity`：未去均值的余弦相似度。
- `amplitude_ratio`：`(pred max-min) / (true max-min)`。
- `p95_error / p99_error`：绝对误差的 p95 / p99。
- `p95_amplitude_ratio / p99_amplitude_ratio`：预测和真值 DeltaT p95 / p99 的比例。
- `peak_true / peak_pred / peak_abs_error`：峰值 DeltaT 及峰值绝对误差。
- `top_k_overlap`：真值和预测的 top-k 热点索引重叠率，当前默认 `top_k=5`。

分母接近 0 的指标返回 `null` 并记录 warning。单个样本失败不会中断整体报告，会进入 failures / warnings。

## final / best 运行方式

同一个 run 目录分别对 final 和 best predictions 运行：

```bash
python scripts/analyze_heat3d_v2_field_shape_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 \
  --trained-predictions output/heat3d_v2_runs/frozen_v1_best_e050_seed0/predictions.npz \
  --prediction-label final \
  --output-json output/heat3d_v2_runs/frozen_v1_best_e050_seed0/field_shape_diagnostics_final.json \
  --output-md output/heat3d_v2_runs/frozen_v1_best_e050_seed0/field_shape_diagnostics_final.md \
  --stdout-mode compact
```

`best` 版本把 `--trained-predictions` 换成 `best_predictions.npz`，输出文件换成 `field_shape_diagnostics_best.*`。

## 当前边界

当前实现只是 research-stage 只读 diagnostics：

- 不训练；
- 不生成新数据；
- 不改 v1 runner；
- 不改模型；
- 不新增 loss；
- 不接 Optax；
- 不把结果称为 formal benchmark。

## 下一步验证

P2a 本地只跑 synthetic smoke。P2b 在 SSH 上 pull 当前 research 分支后，对已经存在的 strict e50 baseline output `output/heat3d_v2_runs/frozen_v1_best_e050_seed0` 分别生成 final/best field-shape diagnostics，并只报告关键指标，不提交 output。
