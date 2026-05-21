# Heat3D v2 RIGNO Batching Audit

## Scope

本轮只做 upstream / 本仓库源码审计，不改 runner、不改模型、不训练、不生成数据。upstream 源码临时 clone 到 `/tmp/rigno-upstream-batch-audit`，commit `3e4b307`，来源为 `https://github.com/camlab-ethz/rigno`。

## 1. Upstream RIGNO 是否使用 batch / mini-batch

结论：upstream RIGNO 使用 mini-batch 训练和 batch-wise evaluation，不是 full-batch 训练。

关键证据：

- `/tmp/rigno-upstream-batch-audit/rigno/train.py:72` 定义 `--batch_size`，默认 `2`，说明是 training samples 的 batch size。
- `/tmp/rigno-upstream-batch-audit/rigno/train.py:160-163` 要求 `n_train % batch_size == 0`，并计算 `num_batches = n_train // batch_size`、`batch_size_per_device = batch_size // NUM_DEVICES`。
- `/tmp/rigno-upstream-batch-audit/rigno/dataset.py:819-838` 的 `Dataset.batches(mode, batch_size, key)` 按 batch_size 生成 batch，支持 key permutation，并处理 remainder。
- `/tmp/rigno-upstream-batch-audit/rigno/train.py:211-217` 用 `jax.pmap` 定义 `_train_one_batch`。
- `/tmp/rigno-upstream-batch-audit/rigno/train.py:517-532` 在 `train_one_epoch` 中遍历 `batches`，对每个 batch 做 `shard(...)` 后调用 `_train_one_batch` 更新 state。
- `/tmp/rigno-upstream-batch-audit/rigno/train.py:727-781` validation/evaluation 也遍历 batches，而不是一次性评估整个 split。
- `/tmp/rigno-upstream-batch-audit/rigno/test.py:369-430` test prediction/export 侧也通过 `dataset.batches(mode='test', batch_size=FLAGS.batch_size)` 分批生成预测，再 concatenate。

没有发现独立的 `gradient_accumulation` CLI 参数。time-dependent 分支内部有 subbatch/valid-pair 逻辑：`train.py:467-503` 用 `jax.lax.fori_loop` 遍历 time-pair subbatch，并累加 loss / grads；但这不是通用跨 sample micro-batch gradient accumulation。对 time-independent `stepper=out` 场景，`train.py:456-458` 使 `num_valid_pairs = 1`，每个 sample batch 只做一次 batch update。

## 2. Upstream 训练循环结构

数据分 batch：

- `Dataset.batches(...)` 根据 mode 和 batch_size 取出 batch。
- training 每 epoch 用 `dataset.batches(mode='train', batch_size=FLAGS.batch_size, key=subkey_0)`，带随机 permutation。
- batch 在进入 pmapped step 前用 `flax.training.common_utils.shard` 拆成 `[NUM_DEVICES, batch_size_per_device, ...]`。

loss 聚合：

- 单个 batch 内 `_train_one_batch` 通过 `jax.value_and_grad` 得到 loss 和 grads。
- 多设备下用 `jax.lax.pmean` 同步 loss / grads。
- epoch loss 用 `loss[0] * batch_size / num_samples_trn` 累加。

validation：

- `evaluate(...)` 接收 batches iterable，循环每个 batch。
- 对每个 batch 做 pmap evaluation，然后把 batch metrics append 到列表。
- 最后 concatenate per-sample metrics，再取 median 等 aggregate。

prediction/export：

- `test.py:get_all_estimations` 通过 `_get_estimations_in_batches` 循环 test batches。
- 每个 batch 单独预测，undo shard 后 append，最后 concatenate。

多设备分片：

- upstream 明确使用 `replicate(state/stats)`、`jax.pmap`、`shard(batch.*)` 和 `shard_prng_key`。
- Heat3D v2 不必第一步迁移 pmap；单 GPU memory-safe 版本可以先做 sample mini-batch / micro-batch，再考虑 pmap。

## 3. Heat3D 当前 full-batch 瓶颈

当前 v1 medium runner 的 full-batch 发生在这些位置：

- `scripts/run_heat3d_v1_medium_controlled_training_export.py:1543-1566` 一次性构建 `train_groups`、`valid_groups`、`all_groups`。
- `scripts/check_heat3d_v1_small_train_valid_smoke.py:187-222` 的 `_make_batch_group` 把同一 shape signature 的全部 examples concatenate 成一个 group。medium1024 Gap-A 当前通常会形成 `train=768`、`valid=128`、`all=1024` 的大 group。
- `scripts/run_heat3d_v1_medium_controlled_training_export.py:1022-1025` 的 `loss_fn` 对完整 `train_groups` 调 `_loss_components`，再用 `jax.value_and_grad` 一次性求 full train split 梯度。
- `scripts/run_heat3d_v1_medium_controlled_training_export.py:1035-1038` 每个 epoch 更新后又 full train / full valid 计算 loss 和 metrics。
- `scripts/run_heat3d_v1_medium_controlled_training_export.py:1102-1112` 的 `_predict_temperatures` 对 `all_groups` 做 full all split prediction export。

M1 OOM 的直接触发点是 `scripts/check_heat3d_v1_small_train_valid_smoke.py:292-295` 的 `_global_norm`，它对每个 gradient leaf 做 `jnp.sum(jnp.square(leaf))` 并转换到 Python float。SSH 报错显示 epoch 1 中尝试额外分配约 `687.94MiB` 时 `RESOURCE_EXHAUSTED`。这说明 full-batch `value_and_grad` 已经把显存推到边缘，随后 grad norm materialization / Optax clip / AdamW moments 都会进一步增加压力。

AdamW 相比 manual GD 还需要 optimizer state，通常至少包含一阶/二阶矩，参数相关显存显著增加。M1 的 `latent64/steps4/mlp2` 同时放大：

- 每层 node / edge latent activation；
- processor message passing 的中间激活；
- MLP hidden activation；
- gradient tree 和 AdamW moment tree。

所以 M1 OOM 不是单个 flag 的问题，而是 full-batch training path 与更大模型容量叠加后的结构性内存瓶颈。

## 4. 可迁移方案对比

### A. quick fix：禁用 grad norm / clip，只做 feasibility

可以新增 `--disable-grad-norm-log` 或把 logging norm 改成可选，避免 `_global_norm` 在 Python 侧 materialize 全部 gradient norm。也可以临时关闭 `--gradient-clip-norm` 来减少 Optax clip 的 global norm 计算。

优点：改动最小，能确认是否只是 grad norm 触发最后一次分配。

缺点：不是根本解决方案。full-batch activation、gradient、AdamW state 仍在；即使绕过 `_global_norm`，后续 optimizer update、train/valid metrics 或 prediction export 仍可能 OOM。该方案只能作为 feasibility probe，不能作为正式 P4 方案。

### B. mini-batch training：按 sample/group 分 batch

参考 upstream `Dataset.batches(...)`，把 `train_examples` 先按 shape signature 分组，再在每组内按 `batch_size` 切 sample batch。每个 batch 独立构建 `inputs/graphs/target`，对该 batch 做一次 optimizer update。

优点：最贴近 upstream；每步 activation 显存随 batch_size 降低；AdamW state 仍保留但可控。

风险：当前 loss 中的 background / hotspot / pseudo-negative quantile 是在 full group 上计算的。若直接在 mini-batch 内计算 quantile，会改变 loss 语义，不能和 strict full-batch reference 混为同一 reference。要么明确记录 mini-batch 语义变化，要么预先固定 full-train threshold / per-split threshold。

### C. micro-batch gradient accumulation

把一个 logical batch 拆成多个 micro-batch，分别 forward/backward，按 sample count 加权累积 gradients，最后只做一次 AdamW update。

优点：更接近 full-batch 梯度语义；能降低 activation peak memory；适合作为 M1 retry 的 memory-safe 方向。

风险：实现复杂度更高。需要保证 loss aggregate、quantile threshold、grad scaling、Optax update、best-valid selection 都一致；还要避免累积 gradient tree 本身过大。当前 Heat3D v2 最小可行版本可以先做 mini-batch，再评估是否需要 strict accumulation。

### D. validation / prediction 分批

validation 和 prediction export 可以独立分批，不改变训练 update 语义。validation 只需对每个 batch 产生 sum/count 或 per-sample metrics，再全 split weighted aggregate。prediction export 只需保持 output `.npz` 的 sample_id keys 和 final/best 文件名不变。

这一步应该与 training batching 同时考虑，否则训练能跑但每 epoch valid metrics 或 final/best export 仍可能 OOM。

### E. M1-lite 容量降级

可以临时跑 `latent32/steps3/mlp2` 或 `latent64/steps3/mlp1` 验证容量方向，但这不解决 full-batch 架构问题。M1-lite 适合在 P4b memory-safe 机制未完成时作为风险较低的对照，不应替代 batching 方案。

## 5. 推荐给 Heat3D v2 的实施路线

P4b-0：memory audit / OOM doc。

- 本地完成即可。
- 记录 M1 OOM 堆栈、full-batch 代码位置、upstream batching evidence、推荐改造方案。

P4b-1：加 `--batch-size` / `--micro-batch-size` dry-run。

- 本地完成即可。
- 只扩 config schema、command builder、smoke，不跑训练。
- 同时增加 `validation_batch_size`、`prediction_batch_size` 草案字段。

P4b-2：mini-batch train smoke。

- 本地可做 py_compile 和小 synthetic / tiny smoke；medium1024 需要 SSH。
- 先不追求复现 strict reference，只验证 batch path 不改模型、不改 loss API、不改 final/best export contract。

P4b-3：M1-lite e50。

- 需要 SSH。
- 用较低容量验证 memory-safe path 和 diagnostics 文件链路。

P4b-4：M1 latent64/steps4 retry。

- 需要 SSH。
- 在 batch path 稳定后再重跑原 M1，不做 sweep，不做 multi-seed。

## 6. 风险与不改动边界

- 不改 `rigno/models/*`。
- 不改变 loss 语义，除非 config/run_config 明确标注 `batch_quantile` 或 `fixed_threshold` 等新语义。
- 不改变 final/best prediction export 语义：仍输出 `predictions.npz` / `best_predictions.npz`，sample_id keys 保持可被现有 diagnostics 读取。
- 不把 mini-batch 结果直接与 full-batch strict baseline 混为同一 reference。A0 strict full-batch 仍是 historical/reference baseline；mini-batch run 应标注为 memory-safe capacity ablation。
- mini-batch 引入 batch order、drop remainder / include remainder、seed、batch_size、gradient accumulation steps 等可复现性字段，必须写入 run_config。
- validation/prediction 若分批，需要严格 weighted aggregate，避免 batch 平均的平均导致小 remainder batch 权重错误。
