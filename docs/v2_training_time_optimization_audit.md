# Heat3D v2 training time optimization audit

## 背景

M1 mini-batch e50 已能在 SSH WSL 上完整运行，但总耗时约 6 小时。P4b-time-1 的 M1 e5 profile 说明每个 epoch 约 7-8 分钟，主要不是 graph build 或 shape recompile。P4b-time-2 已新增 `train_metrics_schedule=half_and_final`，让 full train metrics 只在中点和最后计算。

P4b-time-2 的 e5 对照 profile：

- epoch loop: `2318.78s -> 1998.02s`；
- 平均 epoch: `463.76s -> 399.60s`；
- 节省 `320.77s`，约 `13.83%`；
- skip 的 epoch 1/2/4 各节省约 `106-110s`。

## 当前已确认的大头

- train batch loss/grad + optimizer update：稳定约 `332s/epoch`，首个 epoch 含 compile 约 `348s`。
- full train metrics：原先约 `110-111s/epoch`，已通过 `half_and_final` 降频。
- validation：约 `20s/epoch`，仍每 epoch 保留，用于 best-valid selection。
- group build：约 `193-195s`，是一次性启动成本，不是每 epoch 主因。
- prediction arrays build：即使 `save_predictions=False`，当前仍会构建 final prediction arrays，e5 profile 中约 `70s`。
- final train/valid metrics：最后一个 epoch 已 computed 时仍会在 loop 后再算一次，e5 profile 中约 `131s`。

## 当前 runner 审查

`scripts/run_heat3d_v1_medium_controlled_training_export.py` 当前可删减或降频项：

- `save_predictions=False` 时仍先调用 `_predict_temperatures(...)` 构建 final prediction arrays，然后只跳过保存。这不影响训练语义，可改为只有需要保存或诊断时才构建。
- `save_best_predictions=False` 时不会构建 best prediction arrays；该路径已在 `if args.save_best_predictions:` 内。
- final train/valid metrics 在 epoch loop 后总是重算；若最后一个 epoch 已按 schedule computed，可复用最后一轮 train metrics 和 validation metrics，避免重复。
- `_global_norm(grads)` 每个 batch 都执行，用于 `grad_norms` logging/finite check，不参与 optimizer update。
- Adam/AdamW path 已在 Optax transform 中使用 `optax.clip_by_global_norm(...)`，所以外部 `_global_norm` 不是 gradient clipping 必需项。
- profile timing 下对 loss/grads 和 params 做 `block_until_ready`，使细粒度 timing 更准确，但会强制同步；普通训练未开启 profile 时不走这些同步点。
- per-batch `bool(np.isfinite(grad_norm))` 和 `float(np.mean(batch_grad_norms))` 会把 grad norm 诊断带回 host；若降低 grad norm report 频率，可减少同步/host 开销。
- full-train evaluation 已从 every epoch 降到 `half_and_final`，仍可按任务改为 `final_only` 或 `none`，但会进一步稀疏 train 曲线。
- validation 必须每 epoch 保留，因为 best-valid selection 当前依赖每 epoch validation 指标。

## RIGNO upstream 对比

审查 upstream RIGNO `/tmp/rigno-upstream-graph-cache-audit`，commit `3e4b307`。

- batch size：训练前要求 `num_samples_trn % batch_size == 0`，且 `batch_size % NUM_DEVICES == 0`；每设备 batch 为 `batch_size // NUM_DEVICES`。
- graph reuse：`dataset.build_graphs(builder=graph_builder)` 预先构建 graph metadata 到 `self.rigs`，`dataset.batches(...)` 只按 idx 切片复用 graph。
- batch API：`dataset.batches(mode, batch_size, key)` 返回 `Batch(u, c, x, t, g)`；train/eval loop 直接迭代 batch。
- compiled update：`_train_one_batch` 使用 `jax.pmap`，loss、`jax.value_and_grad`、`state.apply_gradients` 在 compiled path 中完成。
- evaluation frequency：`evaluation_frequency = epochs // EVAL_FREQ`，不足时为 1；train/valid full evaluation 只在频率命中或最后一轮执行。
- train metrics：upstream 会低频跑 train 与 valid evaluation，不是每 epoch 固定 full train metrics。
- grad diagnostics：upstream 返回 grads 并计算一个 mean-abs-grad 类诊断用于 logging，不是外部 global grad norm，也不用于 update。
- prediction arrays：upstream 重点保存 checkpoint 和 metrics/plots；没有 Heat3D runner 这种 `save_predictions=False` 仍构建全量 prediction archive 的契约。
- 可迁移思想：低频 evaluation、graph 一次 build 后复用、compiled train batch update、把 diagnostics 与训练核心路径分离。
- 不适合直接迁移：`pmap` 多设备假设、time-dependent rollout 评估、upstream 数据集元数据结构和 rmesh 随机重建策略。

## CUDA/JAX 检查

SSH WSL 检查结果：

- remote HEAD: `5ab3752`；
- `jax_version`: `0.9.1`；
- `jax_default_backend`: `gpu`；
- `jax_devices`: `[CudaDevice(id=0)]`；
- device: `cuda:0`；
- device kind: `NVIDIA GeForce RTX 5070`；
- `local_device_count`: `1`。

结论：当前训练应已走 CUDA JAX。下一步优先优化代码路径，而不是先修 CUDA 环境。

## 推荐优化优先级

P0: 确认 GPU backend。已确认 `gpu`/`cuda:0`/RTX 5070。

P1: `save_predictions=False` 时跳过 prediction arrays build。预期可省约 `70s/run`，不改变训练、validation、best selection。

P2: 新增 `grad_norm_report_every` 或 `--disable-grad-norm-report`。外部 `_global_norm` 不参与 Optax clipping，降频风险主要是 grad diagnostic 变稀疏。

P3: final metrics reuse。若最后一个 epoch 已 computed full train metrics，可复用 epoch history 中的 train/valid metrics，避免重复约 `131s` 的收尾评估。

P4: `profile_sync_mode`。将 profile 分为 coarse 和 detailed；默认只 coarse sync，详细 per-batch sync 仅在诊断 recompile/shape 时使用。

P5: jitted train_step feasibility。把 loss/grad/update 合成 compiled train step 是最大方向，但本轮不实现；需要先确认 current Heat3D batch object 是否能稳定作为 JAX pytree/static args。

P6: in-memory group cache。对启动 group build 有帮助，但对 epoch 主体帮助有限。

P7: optional disk cache。最后考虑，风险是 cache key、数据变化和 graph config 失效。

## 风险排序

- 低风险：P1、P3。只删减不必要导出/重复评估，不影响 optimizer 或 validation。
- 中低风险：P2。保留 Optax clipping，减少外部诊断频率；需要保留 finite check 的替代策略。
- 中风险：P4。profile 数据解释会变化，需要明确 coarse/detailed 语义。
- 中高风险：P5。可能引入 JAX static arg、pytree、compile cache 和 shape 约束问题。
- 中高风险：P6/P7。cache 失效和数据/graph config 变化风险需要严格 key。

## P4b-time-4 P1/P2/P3 实现

本轮实现三个不改变训练语义的删减/降频项：

P1: `save_predictions=False` 时跳过 final prediction arrays build。

- 之前即使不保存 `predictions.npz`，runner 仍会调用 `_predict_temperatures(...)` 构建 1024 个 recovered prediction arrays；
- 现在只有 `--save-predictions` 为真时才构建 final predictions；
- `save_best_predictions=True` 时仍单独构建 best-valid predictions，不改变 `best_predictions.npz` 文件契约；
- run summary 记录 `final_prediction_export_skipped=true` 和 `final_prediction_export_skip_reason=save_predictions_false`；
- 预期节省约 `70s/run`。

P2: 新增 `--grad-norm-report-every`。

- 默认值为 `10`；
- `1` 表示旧行为，每个 train batch 都计算外部 `_global_norm(grads)`；
- `N>1` 表示只在 batch index 可被 `N` 整除时计算外部 grad norm；
- `0` 表示关闭外部 grad norm reporting；
- 这不关闭 `optimizer_config["gradient_clip_norm"]`，Adam/AdamW path 中的 `optax.clip_by_global_norm(...)` 仍保留；
- 不改变 grads、params update、loss、optimizer 类型或 batch size；
- `profile_timing` 中 skipped batch 的 `grad_norm_time=0`，并标记 `grad_norm_reported=false`；
- 风险是 grad norm 曲线变稀疏，finite check 只覆盖 reported grad norm 和最终 train/valid metrics。

P3: final metrics reuse。

- 当最后一个 epoch 已按 `train_metrics_schedule` 计算 full train metrics 时，缓存最后一轮的 train/valid loss components 和 metrics；
- epoch loop 后直接复用这些结果作为 final metrics，不重复调用 full train/valid forward；
- 记录 `final_metrics_reused=true` 和 `final_metrics_reuse_source=last_epoch_full_metrics`；
- 若 `train_metrics_schedule=none` 或最后一轮未 computed，则 fallback 到旧逻辑，确保 final metrics 仍存在；
- 预期节省约 `131s/run`。

本轮 M1 e5 对照 config：

```text
configs/heat3d_v2/frozen_v1_e005_adamw_m1_batch_profile_p1p2p3_timeopt.yaml
```

该 config 与上一轮 M1 e5 profile 保持相同 model、optimizer、loss、dataset、batch sizes、epochs 和 seed，仅差异为：

- `grad_norm_report_every=10`；
- `save_predictions=false` 时跳过 final prediction arrays build；
- 最后 epoch full train metrics 已 computed 时复用 final metrics。
