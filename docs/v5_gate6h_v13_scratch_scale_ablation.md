# Gate 6H V13 scratch scale-path ablation

状态：`frozen_prepared`，正式 e600 未启动。本阶段只使用 train/valid_iid；test、hard roles 和 sealed IID 禁止访问。

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
