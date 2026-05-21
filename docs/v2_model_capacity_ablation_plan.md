# Heat3D v2 Model Capacity Ablation Plan

## 目标

P4 的目标是在不改 `rigno/models/*`、不新增 loss、不扩数据的前提下，通过 runner CLI/config 显式控制 RIGNO 容量，验证更大的 latent width、processor steps 和 MLP depth 是否能缓解 over-smoothing / variance collapse。

## M1 设计

M1 使用与 A2 相同的 dataset、epochs、seed、selection metric、loss 参数和 AdamW 设置，只改变模型容量：

- optimizer：AdamW；
- lr：`1e-3`；
- weight_decay：`1e-4`；
- gradient_clip_norm：`1.0`；
- node_latent_size：`64`；
- edge_latent_size：`64`；
- processor_steps：`4`；
- mlp_hidden_layers：`2`。

对应 config：

`configs/heat3d_v2/frozen_v1_e050_adamw_lr1e3_wd1e4_m1_latent64_steps4_mlp2_seed0.yaml`

## 对照

- A0：historical/reference baseline，manual GD，legacy small model；
- A2：AdamW lr1e-3 wd1e-4，legacy small model；
- M1：AdamW lr1e-3 wd1e-4，larger model capacity。

这样可以区分 optimizer 改变和模型容量改变带来的影响。

## 成功标准

不能只看 overall RMSE / MAE。M1 必须同时看：

- valid RMSE / MAE；
- bin0 bias / over_ratio；
- high-bin RMSE / MAE；
- field_variance_ratio；
- centered_spatial_correlation；
- amplitude_ratio；
- peak_abs_error；
- p95 / p99 error；
- top_k_overlap；
- final-vs-best 差异。

如果 M1 明显改善 variance_ratio、correlation、amplitude_ratio、top-k overlap，且 bin0 bias 没有严重恶化，可以进入 M2 或容量 follow-up。如果 M1 仍然 variance collapse，下一步应检查 target / normalization / loss design，而不是盲目继续加大模型。

## 边界

本阶段仍是 diagnostic / research-stage controlled training，不是 formal benchmark。不要做 multi-seed、不要做 sweep、不要扩数据、不要提交 output/data/log/checkpoint。
