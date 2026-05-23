# Heat3D v2 graph cache and timing audit

## 为什么怀疑 graph build

M1 e50 已在 SSH 完成，但总训练时间接近 6 小时。当前 v2 runner 的 mini-batch 路径降低了单步显存压力，却仍可能在启动阶段重复构建图/分组：同一批样本会分别进入 train、valid、all prediction group，分组扫描阶段还会先构建 metadata 来判定 shape signature。若 graph metadata/build_graphs 是高成本区，训练总时长会被启动构图和预测导出前的全量 group build 放大。

## Upstream RIGNO 发现

审计 upstream `/tmp/rigno-upstream-graph-cache-audit`，commit `3e4b307`。

- `Dataset.build_graphs(builder=graph_builder)` 在训练开始前为 train/valid/test 样本预构 graph metadata，并把结果放入 `self.rigs` 内存字段。
- `Dataset.batches(mode, batch_size, key)` 返回 `Batch(u, c, x, t, g)`，其中 `g` 来自已缓存的 `self.rigs` 切片；batch 生成本身不重新调用 graph builder。
- train loop 每个 epoch 遍历 `dataset.batches(...)`，对 batch 做 `shard(batch.*)` 后更新 state；eval loop 同样按 batch 遍历。
- upstream 在 `dataset.metadata.fix_x` 为真时，每个 epoch 会用新 PRNG key 重新 `dataset.build_graphs(...)`，用于 rmesh 变化；否则主要复用预构结果。
- 未看到通用磁盘 graph cache；缓存是 dataset object 内的 in-memory metadata/rigs cache。
- test/export 路径也 batch-wise 生成预测；部分 test correction 路径会按 batch 临时 `_build_graph_metadata(...)`。

## 当前 Heat3D 构图路径

`scripts/run_heat3d_v1_medium_controlled_training_export.py` 使用 `Heat3DGraphBuilder` 和 small smoke helper：

- dataset load 后构造 `Heat3DV1NativeSupervisedDataset(..., k_encoding_mode="diag3")`。
- `_make_groups_with_progress(...)` 分别为 `train_examples`、`valid_examples`、`all_examples` 建 group。
- 分组扫描时对每个 sample 调用 `builder.build_metadata(example.condition.coords)` 取得 shape signature。
- 每个 group/batch 进入 `_make_batch_group(...)` 后再次为 batch 内样本 build metadata，再调用 `builder.build_graphs(metadata)`。
- mini-batch 后，epoch loop 复用启动阶段生成的 `train_groups`/`valid_groups`，不会每 epoch 重建 graph。
- prediction export 复用启动阶段生成的 `all_groups`，不在 export 时额外构 graph。
- diagnostics scripts 读取已有 prediction archives 和 JSON/NPZ，不走 runner 的 graph builder。

## 新增 timing hooks

runner 新增：

```bash
--profile-timing
--profile-timing-json output/heat3d_v2_runs/timing_smoke/profile_timing.json
```

默认不启用额外 profile 输出，不改变训练语义、final export 或 best export。启用后会打印/保存：

- dataset/load time；
- graph/group build time；
- train/valid/all group count；
- approximate graph metadata/build_graphs call count；
- per-epoch total/train/validation time；
- train/valid batch count；
- per-train-batch total/loss-grad/grad-norm/optimizer-update/other time；
- per-batch shape signature 和 possible recompile 标记；
- per-valid-batch time 和 shape signature；
- prediction export time。

本地 helper smoke：

```bash
python scripts/check_heat3d_v2_graph_build_timing_smoke.py
```

真实 timing smoke 应放到 SSH WSL，用 1 epoch、小 subset、小 batch，并带 `--profile-timing` 与 `--profile-timing-json`。

## P4b-time-1 M1 e5 profile

之前 1 epoch smoke 的 `group_build ~= 17s`、`epoch train ~= 13.77s` 只能说明小 subset 的图构建不是主要问题；它没有覆盖 M1 e50 的完整模型容量、batch_size=4、medium1024 full1024 split、AdamW、gradient clipping 和 5+ epoch 后的稳定批次行为。因此它不足以解释 e50 总耗时约 6 小时、每 epoch 7-8 分钟的问题。

本轮新增 `configs/heat3d_v2/frozen_v1_e005_adamw_m1_batch_profile.yaml`，目标是在 SSH WSL 上跑 M1 same config 的 5 epoch timing profile：

- model: latent64/edge64/processor_steps4/mlp2；
- optimizer: AdamW lr1e-3 weight_decay1e-4 gradient_clip_norm1.0；
- batch_size/validation_batch_size/prediction_batch_size 与 M1 e50 相同；
- dataset: `medium1024_gapA_full1024_v2`；
- save final/best prediction npz 关闭，profile focus 是 train/valid timing；runner 仍会按既有流程构建 final prediction arrays 并计入 `prediction_export`。

判断规则：

- first batch compile：每个 epoch 的 `first_train_batch_time` 明显高于 later median，且 later batches 稳定。
- later-batch recompile：later batch time 超过同 epoch later median 的 3 倍，`possible_recompile=true`。
- batch shape variation：比较 `batch_shape_signature_key` 的 unique count；若慢 batch 对应不同 shape 或 final small batch，说明固定 shape/drop_last/padding 可能有效。
- grad norm overhead：看 `grad_norm_time` 的 total/median 占比；若显著，应考虑增加 `--disable-grad-norm-report` 或降低记录频率。
- optimizer update overhead：看 `optimizer_update_time` 的 total/median 占比；若显著，优先确认 AdamW state/update 是否成为瓶颈。
- validation overhead：看 `validation_total_time` 和 per-valid-batch time；若占比高，考虑降低验证频率或分离 validation profile。

下一步决策：

1. 若 batch shape 变化导致 recompile：优先固定 shape、`drop_last` 或 padding。
2. 若 grad norm 慢：新增 `--disable-grad-norm-report` 或降低记录频率。
3. 若 group_build 仍显著但占比有限：做 in-memory sample/group cache。
4. 最后才考虑 optional disk graph cache。

## 若确认 graph build 是瓶颈

推荐路线：

1. 先做 in-memory graph/group cache：按 split/sample ids/shape signature 复用 scan metadata、batch metadata 和 build_graphs 结果。
2. 再做 optional disk cache：只缓存稳定、可验证的 metadata/group artifacts，不默认启用。
3. cache key 至少包含 subset path、sample ids、graph builder config、`k_encoding_mode`、代码版本。

主要风险：

- cache key 不完整导致 stale graph；
- subset 或 sample 文件变化后 cache 未失效；
- graph config/rmesh 参数变化但命中旧 cache；
- mini-batch shuffle 后 cached batch 与 sample 顺序不一致；
- cached group 粒度过大时会抵消 mini-batch 显存收益。
