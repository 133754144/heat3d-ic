# Gate 6H V13 scratch scale-path ablation

状态：V28/V30 已完成 e600 摘要收集，V29 仍在 WSL2 运行。本阶段只使用 train/valid_iid；test、hard roles 和 sealed IID 禁止访问。

## e600 结果收集（2026-07-17）

结果已写入 `configs/heat3d_v5/v5_gate6h_v13_scale_ablation_registry.csv`。下表是训练摘要中的 legacy `valid_rel_rmse_v4_pct`、normalized base MSE 与 raw DeltaT RMSE；它不是 Gate 5 true-RMS evaluator 结果。

| config | host | status | best epoch | best base MSE | best raw RMSE K | best legacy relative RMSE |
|---|---|---|---:|---:|---:|---:|
| V4P5_28_gate6h_v13_stopgrad_scratch_e600 | wsl2 | completed_e600 | 554 | 0.04082419 | 0.16094990 | 25.0443% |
| V4P5_29_gate6h_v13_scale_attention_scratch_e600 | wsl2 | running_e600 | — | — | — | — |
| V4P5_30_gate6h_v13_deep_scale_head_scratch_e600 | devbox | completed_e600 | 275 | 0.03817912 | 0.15565122 | 24.2198% |

V28/V30 的 `result_v5_required_metrics_complete=false` 是有意的：当前 run 只有 `loss_summary.json` 与 valid-only persisted diagnostics，尚未运行统一 true-RMS V5 evaluator；test/hard 未访问。V29 的结果收集留到训练自然结束后。

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

本阶段明确不准备 V22 e600，且 `long_training_started=false`。

## E1 execution smoke

三组配置均在 devbox 正式 P5 train=672、valid_iid=128、1024 nodes、B28 上完成 e1；这不是正式性能结果。

| config | params | peak RSS MiB | live device bytes | replay | valid diagnostics |
|---|---:|---:|---:|---|---|
| V28 stop-gradient | 853927 | 4322.66 | 7193244416 | 5/5 passed | completed / valid_iid_only |
| V29 scale attention | 893736 | 4611.16 | 7360296960 | 5/5 passed | completed / valid_iid_only |
| V30 deep scale head | 862247 | 4337.45 | 7186432768 | 5/5 passed | completed / valid_iid_only |

全部 smoke 均满足 finite loss/gradient、train-only context fit、summary-before-replay、forbidden roles 为空。正式 e600 仍为 `not_started`。

## 手动启动顺序

仅在明确决定启动正式 e600 后执行：

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_28_gate6h_v13_stopgrad_scratch_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_29_gate6h_v13_scale_attention_scratch_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_30_gate6h_v13_deep_scale_head_scratch_e600.yaml
```
