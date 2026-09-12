# G2-E2 Heat3D runtime / batch-scaling diagnostic

本报告仅记录 devbox 上 frozen G2 P1i workload 的 3-epoch runtime 诊断；不构成 accuracy、收敛或 formal-training 证据。训练数据为每 epoch 768 个 train samples，validation 为 128 个 valid_iid samples；所有 train batch 等长，valid 使用 B32，无 test/sealed 访问。

## 执行路径审计

当前 runner 的 `trainer.step` 已将 `jax.value_and_grad`、gradient transform、optimizer update/apply_updates 放进同一个 `jax.jit(_step_impl)` executable。control 路径仍在每个 train step 对完整输出块调用 `block_until_ready`，并把 loss、梯度/更新/参数有限性及范数转回 host；compiled 诊断路径保持同一数值 step，仅同步训练所需状态并省略范数诊断。两条路径都没有改变数学、数据、normalization、optimizer 或 checkpoint selection。forward/loss、backward 和 optimizer update 的 GPU 内部耗时未拆分，需 profiler 才能进一步区分。

## 统一结果

| 配置 | batch / train batches | 首次编译阶段 (s) | epoch 1 / 2 / 3 (s) | post-warm median step (s) | p95 (s) | post-warm samples/s | median valid (s) | GPU 显存峰值* | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current control B24 | 24 / 32 | 2459.73 | 2779.51 / 1729.35 / 1605.43 | 44.25 | 45.24 | 0.5548 | 296.44 | 11694/12227 MiB | PASS |
| compiled B24 | 24 / 32 | 2207.40 | 2497.78 / 1547.09 / 1544.84 | 39.82 | 40.59 | **0.6015** | 268.62 | 11694/12227 MiB | PASS |
| compiled B48 | 48 / 16 | 2469.40 | 2760.32 / 1558.32 / 1581.42 | 81.09 | 82.61 | 0.5919 | 274.15 | 11714/12227 MiB | PASS (near limit) |
| compiled B96 | 96 / 8 | 未完成 | — | — | — | — | — | 11468/12227 MiB sampled | STOPPED: ~45 min first JIT, no steady state |
| compiled B128 | 128 / 6 | — | — | — | — | — | — | — | NOT RUN |

\*显存为 `nvidia-smi` 采样峰值；B48 仅约 4.2% 采样余量。每个已完成配置均 loss/grad finite；epoch 1→3 的 loss 仅作为运行 sanity check，不作模型优劣判断。B96 在首个 executable 编译中安全停止，保留日志 SHA `01ec18f942a8f3ab7a3fc00c7b39db5641a172bc28282b2e45b3a60f3b89a6c9`，没有 OOM、JSON、progress 或 checkpoint；因此没有伪造 steady-state 指标，也未运行 B128。

### 显式计时与同步

| 配置 | dispatch/JIT 总计 (s) | explicit sync + host postprocess (s) | train sync calls | train host scalar extractions | 200e 单 seed runtime-only projection |
|---|---:|---:|---:|---:|---:|
| control B24 | 5153.46 | 74.73 | 96 | 384 | 96.08 h |
| compiled B24 | 4756.69 | 0.31 | 96 | 192 | 85.95 h |
| compiled B48 | 5061.33 | 0.15 | 48 | 96 | 87.86 h |

`dispatch` 包含 JIT/device execution；它远大于显式 host 诊断时间，说明主要瓶颈仍在执行/dispatch/compute 路径。compiled B24 相对 control 的 post-warm step 降低约 10.0%，吞吐提高 8.4%，median epoch wall 降低约 10.6%，但这不是 accuracy 结论。B48 将每步工作量约翻倍，吞吐反而比 compiled B24 低 1.6%，epoch wall 高 2.2%，且显存接近上限。

## 结论与边界

- 本诊断确认：单一 JIT 已包含 loss/grad/optimizer update；control 的 host scalar/norm 诊断有可测 overhead，但不是总 runtime 的主导部分。
- 推荐的**诊断/后续 bounded preflight**配置为 compiled B24；它是已完成档位中吞吐最高，且显存采样值略低于同样接近上限的 B48。该建议不改变 frozen science contract，也不释放 formal 长训练。
- B96 因首个 batch 的 XLA 编译约 45 分钟仍无 steady-state 而停止；B128 不运行。
- 未启动 profiler、GINO/Transolver formal 或任何长周期训练；未访问 P1i test/sealed、DeepOHeat official test，未修改 G1。

Machine-readable 明细及远端 artifact SHA 见 `docs/v7_g2_e2_heat3d_batch_scaling_receipt.json`；原始 JSON/checkpoint/log 均留在 devbox `/tmp`，不提交 Git。
