# V7 G2 P23 frozen model identities

本清单冻结 P23 使用的 checkpoint identity，不进行重训、重选 checkpoint 或
test 评估。P1i 数据合同为 `heat3d_v6_p1i_continuous_physics1024_v1`，valid-only
为 128 个 case、240,825 个物理节点；dataset/split/truth SHA 见同名 JSON。

| 模型 | P23 角色 | seed/checkpoint selection | 表示 | 参数量 | external pretraining |
|---|---|---|---|---:|---|
| Heat3D V6 | canonical native-1024；full-field candidate | V6 frozen point-global best | 1024 source-aware points；V6 full-field utility | 892,776 | 否 |
| GINO | native-1024 common-task | valid sample-first best | Open3D FRS + torch-scatter，32³ latent | 13,673,988 | 否 |
| Transolver | native-1024 common-task | valid sample-first best | Physics-Attention point tokens | 716,737 | 否 |
| Therm-FM | `COMPLETE_VALID_ONLY_3_SEEDS_PRETRAINED_TRANSFER` | minimum valid normalized loss；物理指标称 `metrics_at_loss_selected_checkpoint` | dense 65×65×57，741 输入通道 | 21,435,546 | Poseidon-T，是 |
| HCP | `REFERENCE_ONLY / OFFICIAL_ACCURACY_ASSETS_BLOCKED` | 不进入 P23 | — | — | — |

P23 full-field 统一比较只允许 Heat3D V6 与 Therm-FM；GINO/Transolver 保持
native-1024 角色，不强行外推到 240,825 节点。完整 SHA、selection policy、
evaluator role 和历史 valid-only evidence 见
[`v7_g2_p23_model_identity_manifest.json`](v7_g2_p23_model_identity_manifest.json)。

当前无法从本工作树取得 V6 seed1/seed2 的 per-sample prediction archive，因此
无法合法生成 3×3 paired comparison；这不是用历史 aggregate 代替 per-sample
证据的理由。
