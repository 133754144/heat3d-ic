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
- prediction export time。

本地 helper smoke：

```bash
python scripts/check_heat3d_v2_graph_build_timing_smoke.py
```

真实 timing smoke 应放到 SSH WSL，用 1 epoch、小 subset、小 batch，并带 `--profile-timing` 与 `--profile-timing-json`。

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
