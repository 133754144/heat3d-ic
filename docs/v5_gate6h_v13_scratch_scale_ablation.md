# Gate 6H V13 scratch scale-path ablation

状态：V28/V30 已完成 e600 与 valid-only true-RMS 统一评估；V29 固定为 `failed_oom_validation_e18`；V31 已准备但未启动。本阶段只使用 train/valid_iid；test、hard roles 和 sealed IID 禁止访问。

## e600 结果收集（2026-07-17）

结果已写入 `configs/heat3d_v5/v5_gate6h_v13_scale_ablation_registry.csv`。下表是训练摘要中的 legacy `valid_rel_rmse_v4_pct`、normalized base MSE 与 raw DeltaT RMSE；它不是 Gate 5 true-RMS evaluator 结果。

| config | host | status | best epoch | best base MSE | best raw RMSE K | best legacy relative RMSE |
|---|---|---|---:|---:|---:|---:|
| V4P5_28_gate6h_v13_stopgrad_scratch_e600 | wsl2 | completed_e600 | 554 | 0.04082419 | 0.16094990 | 25.0443% |
| V4P5_29_gate6h_v13_scale_attention_scratch_e600 | wsl2 | failed_oom_validation_e18 | — | — | — | — |
| V4P5_30_gate6h_v13_deep_scale_head_scratch_e600 | devbox | completed_e600 | 275 | 0.03817912 | 0.15565122 | 24.2198% |

V29 的状态来自用户明确确认；2026-07-17 远程复核时 run 目录为空、配置日志路径不存在且无存续进程，因此不能伪造 traceback 或日志 hash。缺口、配置 hash 与 e1 memory 证据冻结在 `configs/heat3d_v5/gate6h/v29_oom_validation_e18_audit.json`。

## Epoch 25 结果

以下为训练 history 的 epoch 25 valid 行；四项 loss 顺序为 shape / log-scale / relative-field / raw-absolute。

| config | 四项 valid loss | total loss | base MSE | point-global true-RMS | amplitude ratio |
|---|---|---:|---:|---:|---:|
| V28 | 0.076514 / 0.162574 / 0.287529 / 0.126960 | 0.610547 | 0.157617 | 49.2084% | 1.18407 |
| V30 | 0.081233 / 0.067180 / 0.123451 / 0.067655 | 0.346544 | 0.090664 | 37.3228% | 0.94022 |

V30 在 e25 的 total、base MSE、point-global 以及 scale/relative/raw 分支上明显低于 V28；V28 只有 shape loss 略低。该行仅作训练轨迹描述，不替代 e600 统一评估。

## Valid-only true-RMS 统一评估

Evaluator commit=`4fdfb842244da1cc4c7353217b7b00d215a039bd`，只读取已有 `best_predictions.npz` / `predictions.npz` 与 `valid_iid=128` 标签、CV；未运行模型推理，未访问 test/hard/sealed。完整公式、hash 与 train-only context 证明见 `configs/heat3d_v5/gate6h/valid_only_true_rms_evaluation.json`。

| config | artifact | epoch | base MSE | point-global true-RMS | sample-first CV-relative | raw CV RMSE K |
|---|---|---:|---:|---:|---:|---:|
| V28 | best | 554 | 0.04082669 | 25.045622% | 23.527666% | 0.18150837 |
| V28 | final | 600 | 0.04392702 | 25.979191% | 25.087848% | 0.18875546 |
| V30 | best | 275 | 0.03817760 | 24.219438% | 21.070756% | 0.17266375 |
| V30 | final | 600 | 0.03908934 | 24.506929% | 20.892804% | 0.17529112 |

V30 在 best/final 的四项统一指标上均优于 V28；但最优 point-global 为 24.219438%，仍未达到 valid `<20%`。

## V29 OOM 峰值分析与 V31

冻结的 V29 e1 smoke 已观测 peak RSS=4611.16 MiB、live device=7019.33 MiB、device pool=8190.00 MiB，即 live 占 85.71%，只余 1170.67 MiB。V29 的 validation B128 是 train B28 的 4.57 倍；这与 epoch 18 validation OOM 一致。由于原始终端日志未落盘，无法给出失败瞬间峰值或 attempted allocation，以上数值只作为可复核下界，不能冒充 OOM 瞬时峰值。

V31 `V4P5_31_gate6h_v29_validation_b32_retry_e600` 严格继承 V29，仅把 validation/prediction batch 从 128 降到 32；train 仍为 B28，模型、loss、LR、optimizer 与全部 seeds 不变。resolved-config checker 将科学差异固定为：

- `run.validation_batch_size: 128 -> 32`
- `run.prediction_batch_size: 128 -> 32`

V31 状态为 `frozen_prepared / not_started`，启动策略仍为 `explicit_user_instruction_only`。

## 历史基线审计

历史 Gate 5/6 closeout 与 registry 显示，已完成 e600 的 B0/N0/N1/N3/L1/L2/V13 中，V13 是 valid_iid point-global 最低的单模型。三项直接对比如下：

| model | checkpoint | point-global relative RMSE | sample-first CV-relative RMSE | raw CV RMSE K |
|---|---:|---:|---:|---:|
| V13 | best e318 | 23.700678% | 20.316459% | 0.16798237 |
| L2 | best e353 | 23.728964% | 20.835389% | 0.16890419 |
| N3 | best e402 | 24.075573% | 20.658375% | 0.17188320 |

因此 V13 不仅按 primary point-global 最优，在这三个候选中 sample-first 与 raw CV RMSE 也最低。指标统一取自 `configs/heat3d_v5/gate6g/v13_closeout.json` 的 valid-only source metrics。

## V13 实际 LR 合同

以下字段直接读取 WSL2 V13 `run_config.json`，其 SHA256 为 `5c4d02d2af9389beeda94db56fb075366798eaf2d419784cc061730a43614afa`；不从旧 YAML 推断：

- optimizer=`adamw`，epochs=600，B28；
- `lr=5e-4`，schedule=`warmup_cosine`，warmup=10，min_lr=`5e-5`；
- `second_stage_epoch=0`，second_stage_lr=`1e-4`；
- `lr_init=1e-5`，lr_peak=`2e-4`，lr_base=`1e-5`，lr_lowr=`1e-6`；
- `pct_start=0.02`，pct_final=`0.1`；
- gradient clip=`1.0`，weight decay=`1e-4`。

## 新 scratch e600 候选

| order | config | 相对 V13 的唯一模型差异 |
|---:|---|---|
| 1 | `V4P5_28_gate6h_v13_stopgrad_scratch_e600` | `pooled_latent_stop_gradient=true` |
| 2 | `V4P5_29_gate6h_v13_scale_attention_scratch_e600` | V28 + `scale_attention_mode=physics_gate` |
| 3 | `V4P5_30_gate6h_v13_deep_scale_head_scratch_e600` | `scale_head_depth=3` |

三组均为 random initialization，继承 V13 的数据、模型、loss、B28、seeds 和实际 LR 合同。primary selection 为 point-global；同时保存 point-global、sample-first、base-MSE 与 final checkpoint。post-training valid diagnostics 开启，final probe 关闭。

初始 preflight 明确不准备 V22 e600；当前生命周期以 registry 为准：V28/V30 completed、V29 failed、V31 not_started。

## E1 execution smoke

三组配置均在 devbox 正式 P5 train=672、valid_iid=128、1024 nodes、B28 上完成 e1；这不是正式性能结果。

| config | params | peak RSS MiB | live device bytes | replay | valid diagnostics |
|---|---:|---:|---:|---|---|
| V28 stop-gradient | 853927 | 4322.66 | 7193244416 | 5/5 passed | completed / valid_iid_only |
| V29 scale attention | 893736 | 4611.16 | 7360296960 | 5/5 passed | completed / valid_iid_only |
| V30 deep scale head | 862247 | 4337.45 | 7186432768 | 5/5 passed | completed / valid_iid_only |

全部 smoke 均满足 finite loss/gradient、train-only context fit、summary-before-replay、forbidden roles 为空。该表只冻结启动前 e1 结论；当前 e600 状态见文首，不用 smoke 状态覆盖最终生命周期。

## 手动启动

V28/V30 已完成，V29 不得续跑。仅在后续明确决定启动 V31 时执行（本轮未执行）：

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_31_gate6h_v29_validation_b32_retry_e600.yaml
```
