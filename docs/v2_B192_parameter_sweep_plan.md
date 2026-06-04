# Heat3D v2 B192 Parameter Sweep Plan

本文设计 B192 optimization parameter sweep。执行策略是分阶段：本轮只先跑 3 个 pilot；根据结果最多再跑 2 个后续候选，不一次性无条件跑完整 10 组。

## Context

B192 full / base_mse / base_mse_hotspot 在 `lr=3e-4` 下均 best_epoch=1，说明 larger batch 和 simplified loss 都没有自动解决后续 valid_loss 退化。B192 e50 wall-clock 约 8-10 分钟，是快速 ablation 平台，但只有 4 updates/epoch、200 total updates/e50，不与 B4 e50 update-count 等价。

## Candidate Pool

| id | phase | config path | batch | loss | lr | weight_decay | grad_clip | updates/e50 | research question | criterion |
|---:|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | A | `configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr1e4_seed0.yaml` | 192 | `mse` | `1e-4` | `1e-4` | `1.0` | 200 | Does lower LR fix base-MSE early-best? | continue if best_epoch > 1 or final/best improves without large best loss regression |
| 2 | A | `configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e5_seed0.yaml` | 192 | `mse` | `3e-5` | `1e-4` | `1.0` | 200 | Is still lower LR needed for B192 stability? | continue if final degradation drops meaningfully |
| 3 | A | `configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_full_lr1e4_seed0.yaml` | 192 | `background_pseudo_negative` | `1e-4` | `1e-4` | `1.0` | 200 | Does full composite mainly need lower LR? | continue full-family only if it beats base_mse pilots |
| 4 | B | TBD | 192 | `background_pseudo_negative` | `3e-5` | `1e-4` | `1.0` | 200 | Test full composite at very low LR | run only if Pilot 3 is promising |
| 5 | B | TBD | 192 | `background_hotspot` | `1e-4` | `1e-4` | `1.0` | 200 | Test hotspot with lower LR | run if Pilot 1 improves stability |
| 6 | B | TBD | 192 | `background_hotspot` | `3e-5` | `1e-4` | `1.0` | 200 | Test hotspot at very low LR | run if Pilot 2 is stable but hotspot weak |
| 7 | B | TBD | 96 | `mse` | `1e-4` | `1e-4` | `1.0` | 400 | Does more update count help base MSE? | run if B192 lower LR still under-updates |
| 8 | B | TBD | 96 | `background_pseudo_negative` | `1e-4` | `1e-4` | `1.0` | 400 | Does B96 stabilize full composite? | run if full B192 remains early-best |
| 9 | B | TBD | 192 | `mse` | `1e-4` | `0.0` | `1.0` | 200 | Is weight decay causing amplitude/late drift? | run if Pilot 1 improves but still degrades |
| 10 | B | TBD | 192 | `mse` | `1e-4` | `1e-4` | `0.5` | 200 | Is grad clipping threshold too loose? | run if Pilot 1 shows instability |

## Phase A Decision Rules

Case A: If Pilot 1 or Pilot 2 gets best_epoch > 1, clearly lowers final/best degradation, and does not badly worsen best_valid_loss, continue with adjacent loss-family candidates such as B192 base_mse_hotspot lower LR or optimizer knobs.

Case B: If Pilot 1 and Pilot 2 still have best_epoch=1 but `lr=3e-5` strongly lowers final degradation, test B192 base_mse `lr=1e-5` or B96 base_mse `lr=3e-5`.

Case C: If all three pilots remain best_epoch=1 with large final degradation, stop the sweep. Conclude that lower LR alone does not fix B192 optimization and shift priority to upstream training gap, runner monitor, and update-count-equivalent design.

Case D: If full composite `lr=1e-4` clearly beats base_mse pilots, test full composite `lr=3e-5`. Otherwise do not continue full composite.

## Reporting Requirements

Every completed run must report:

- best_epoch;
- best_valid_loss;
- final_valid_loss;
- best/final raw_deltaT_mse;
- hotspot_mae;
- bg_bias;
- pn_over_ratio;
- final/best degradation;
- wall-clock;
- updates;
- whether best_epoch remains 1.

If predictions exist and field-shape diagnostics are missing, run only read-only field-shape diagnostics before comparing spatial metrics.
