# Heat3D v2 Memory-Safe Batching Plan

## 背景

M1 = AdamW + latent64/steps4/mlp2/e50 在 SSH WSL RTX 5070 上以 full-batch 方式运行时 OOM。当前 v1 medium runner 会把 train/valid/all split 按 shape group 一次性打成大 batch，epoch 内对完整 train group 做 `value_and_grad`，再计算 grad norm、AdamW update、train/valid metrics 和 final/best prediction export。模型容量增大后，activation、gradient、AdamW moments 和 grad norm materialization 共同推高显存。

## Mini-Batch 与 Full-Batch 的区别

full-batch 每次参数更新使用整个 train split 的梯度，复现性和 strict V1 baseline 对齐较直接，但显存随样本数和模型容量放大。

mini-batch 每次参数更新只使用一部分样本，显存峰值更低，但 batch order、batch_size、drop_last、shuffle seed、loss quantile 计算范围都会影响结果。因此 mini-batch 结果不能直接视为与 full-batch strict baseline 数值等价，必须单独标注为 memory-safe capacity ablation。

micro-batch gradient accumulation 介于二者之间：多个 micro-batch 分别求梯度并按权重累积，最后做一次 optimizer update。它更接近 full-batch 语义，但实现复杂度更高。

## Upstream 启发

upstream RIGNO 使用 `--batch_size`、`Dataset.batches(...)`、`pmap/shard` 做 batch-wise training/evaluation。它的训练循环按 batch 更新 state，validation/test 也按 batch 汇总 metrics / predictions。这说明 Heat3D v2 不应该继续依赖 full-batch runner 来做容量实验。

Heat3D v2 的第一步不需要直接迁移 upstream 的 pmap；可以先做单 GPU sample mini-batch，再把 validation/prediction export 分批，保持 final/best `.npz` 文件契约不变。

## P4b 路线

P4b-1：batch config dry-run。

- 在 v2 config / command plan 中加入 `batch_size`、`micro_batch_size`、`validation_batch_size`、`prediction_batch_size`、`shuffle_train_batches`、`drop_last`。
- 本阶段只生成 planned command，不改训练行为。
- `micro_batch_size` 先标记为 future / unmapped。

P4b-2：mini-batch training smoke。

- 只做小样本 / 1 epoch smoke。
- 不改 `rigno/models/*`，不改 loss API。
- 重点验证 batch path、weighted aggregate、best-valid selection 和 final/best export contract。
- 当前实现让 `--batch-size > 0` 时按 shape group 切成 sample mini-batches，每个 mini-batch 单独 `value_and_grad` 和 optimizer update；`--batch-size 0` 保持 legacy full-batch。
- `--validation-batch-size` 和 `--prediction-batch-size` 分别控制 valid aggregation 和 final/best prediction export 的 batch group 大小；`.npz` 文件名和 sample_id key 契约不变。

P4b-3：M1-lite e50。

- 用 latent32/steps3/mlp2 + AdamW + batch path 先确认 memory-safe chain。
- 必须报告 field-shape diagnostics，不只看 RMSE。

P4b-4：M1 retry。

- 在 P4b-2/P4b-3 通过后再 retry latent64/steps4/mlp2。
- 仍不做 sweep、不做 multi-seed、不声称 benchmark。

## 本轮边界

P4b-1 只做 dry-run/config 接口；P4b-2 已将 batch CLI 接入 runner，但仍是 research-stage memory-safe training path，不是 strict full-batch reference。mini-batch 与 full-batch 使用相同 loss 公式，但 batch 内 quantile / update order 会改变数值轨迹，因此结果不能与 strict e50 full-batch A0 视为数值等价。

## SSH 验证递进策略

1. 先跑 M1-lite e1，验证 mini-batch path 能完成一次 epoch 和 final/best export。
2. 若 e1 不 OOM，再跑 M1-lite e3，确认多 epoch、best-valid selection 和 batch shuffling 没有明显问题。
3. 若 M1-lite e3 通过，再跑原 M1 e1，验证 latent64/steps4/mlp2 在 batch path 下是否可行。
4. 若原 M1 e1 通过，再跑原 M1 e50 并生成 final/best 全套 diagnostics。

如果 M1 e50 成功，下一步再比较 field-shape metrics、bin0、high-bin 与 A0/A2 的 tradeoff。如果 M1 e1 仍 OOM，下一步先减小 batch size，之后再考虑 micro-batch gradient accumulation；不要直接做更大模型或 sweep。

## 需要记录的可复现字段

P4b-2 接入 mini-batch 后，run config / loss summary 至少应记录：

- `batch_size`；
- `micro_batch_size`；
- `validation_batch_size`；
- `prediction_batch_size`；
- `shuffle_train_batches`；
- batch shuffle seed / epoch seed rule；
- `drop_last`；
- loss quantile 是 per-batch、per-group，还是 fixed full-train threshold；
- validation/prediction aggregate 是否按 sample count 加权。
