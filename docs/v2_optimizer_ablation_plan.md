# Heat3D v2 Optimizer Ablation Plan

## 目标

P3 的目标是在不改模型、不改 loss、不扩数据的前提下，把 v1 medium controlled runner 从单一 legacy manual full-batch gradient descent 扩展为可显式选择 optimizer 的 runner，并用严格对齐的 e50 seed0 配置做最小 optimizer ablation。

## 对比设计

- A0：已有 strict frozen V1 best reproduction，manual GD，`lr=1e-2`，`epochs=50`，`seed=0`。
- A1：Optax Adam，`lr=1e-3`，`weight_decay=0`，`gradient_clip_norm=1.0`。
- A2：Optax AdamW，`lr=1e-3`，`weight_decay=1e-4`，`gradient_clip_norm=1.0`。
- A3：Optax AdamW，`lr=3e-4`，`weight_decay=1e-4`，`gradient_clip_norm=1.0`。

A1/A2/A3 使用同一 dataset、epochs、seed、selection metric、loss 参数和当前 v1 model config，只改变 optimizer、lr、weight decay 和 gradient clipping。

## 成功标准

不能只看 overall RMSE / MAE。每组都必须同时报告：

- final / best overall RMSE / MAE；
- final / best valid RMSE / MAE；
- bin_0 bias / over_ratio；
- high-bin RMSE / MAE；
- field_variance_ratio；
- centered_spatial_correlation；
- amplitude_ratio；
- peak_abs_error；
- p95 / p99 error；
- top_k_overlap；
- final-vs-best 差异。

如果 AdamW 只改善 RMSE/MAE，但 field variance ratio、spatial correlation、amplitude ratio 或 top-k overlap 没有改善，不算真正解决 field-shape 问题。如果 field-shape 改善但 bin_0 bias / over_ratio 明显恶化，需要明确标注 tradeoff。

## 当前边界

- 不修改 `rigno/models/*`；
- 不新增 loss；
- 不改数据生成；
- 不做 multi-seed；
- 不扩数据；
- 不提交 output/data/log/checkpoint；
- 结果只作为 diagnostic / research-stage controlled training 对比，不声称 formal benchmark。

## 运行方式

本地只运行 config-to-command smoke。训练和 diagnostics 只在 SSH 服务器上运行，远程使用 `python`：

```bash
python scripts/check_heat3d_v2_optimizer_config_smoke.py
```

通过 smoke 后，按 A1/A2/A3 config 生成的 dry-run command plan 在 SSH 上依次运行 training，再运行 final/best baseline comparison、error bins、run summary、condition diagnostics 和 field-shape diagnostics。
