# Heat3D v2 Next Training Config Index

All configs below are dry-run prepared YAML files. They use the existing
`medium1024_gapA_full1024_v2` stratified split, save final/best predictions,
and keep `valid_iid` as primary validation while `valid_stress` remains
diagnostic only.

## Priority Summary

| priority | purpose |
|---|---|
| P0 | highest-value next SSH runs: M1.5 continuation/lower LR/schedule, M1.25 first tests, M1 baseline seeds |
| P1 | regularization, light background controls, and secondary M1.25/M1 controls |
| P2 | very low LR, rapid decay, combined background controls, and extra M1.5 seeds |

## Configs

| priority | config | model | batch | epochs | lr/schedule | loss variant | purpose |
|---|---|---:|---:|---:|---|---|---|
| P0 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | extend M1.5 e100 to e200 |
| P0 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr1e4_stratified_seed0.yaml` | M1.5 | 96 | 200 | 1e-4 / constant | base MSE | lower LR overshoot control |
| P2 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e5_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-5 / constant | base MSE | very low LR stability check |
| P0 | `frozen_v1_e200_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / warmup_cosine | base MSE | schedule smoothing |
| P2 | `frozen_v1_e200_adamw_m15_B96_base_mse_rapid_decay_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / rapid_decay | base MSE | fast decay overshoot control |
| P1 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_clip0p5_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | stronger clip |
| P1 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_clip0p1_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | very strong clip |
| P1 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_wd1e3_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | stronger weight decay |
| P1 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_wd1e2_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | high weight decay stress test |
| P1 | `frozen_v1_e200_adamw_m15_B96_light_bg_bias_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | light background bias | low-DeltaT bias control |
| P1 | `frozen_v1_e200_adamw_m15_B96_light_bg_over_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | light background over | low-DeltaT overprediction control |
| P2 | `frozen_v1_e200_adamw_m15_B96_light_bg_l1_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | light background L1 | conservative background magnitude control |
| P2 | `frozen_v1_e200_adamw_m15_B96_light_bg_bias_over_stratified_seed0.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | light bias + over | combined background control |
| P0 | `frozen_v1_e200_adamw_m125_B128_base_mse_lr3e4_stratified_seed0.yaml` | M1.25 | 128 | 200 | 3e-4 / constant | base MSE | intermediate capacity first test |
| P1 | `frozen_v1_e200_adamw_m125_B128_base_mse_lr1e4_stratified_seed0.yaml` | M1.25 | 128 | 200 | 1e-4 / constant | base MSE | intermediate lower LR |
| P1 | `frozen_v1_e200_adamw_m125_B128_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.25 | 128 | 200 | 3e-4 / warmup_cosine | base MSE | intermediate schedule test |
| P0 | `frozen_v1_e200_adamw_m125_B96_base_mse_lr3e4_stratified_seed0.yaml` | M1.25 | 96 | 200 | 3e-4 / constant | base MSE | safer-memory intermediate test |
| P1 | `frozen_v1_e200_adamw_m125_B96_base_mse_lr1e4_stratified_seed0.yaml` | M1.25 | 96 | 200 | 1e-4 / constant | base MSE | safer-memory lower LR |
| P0 | `frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed1.yaml` | M1 | 192 | 200 | 3e-4 / constant | base MSE | baseline seed robustness |
| P0 | `frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed2.yaml` | M1 | 192 | 200 | 3e-4 / constant | base MSE | baseline seed robustness |
| P1 | `frozen_v1_e200_adamw_m1_B192_base_mse_lr1e4_stratified_seed0.yaml` | M1 | 192 | 200 | 1e-4 / constant | base MSE | lower-LR baseline control |
| P1 | `frozen_v1_e200_adamw_m1_B192_base_mse_warmup_cosine_stratified_seed0.yaml` | M1 | 192 | 200 | 3e-4 / warmup_cosine | base MSE | schedule baseline control |
| P2 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed1.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | M1.5 seed robustness |
| P2 | `frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed2.yaml` | M1.5 | 96 | 200 | 3e-4 / constant | base MSE | M1.5 seed robustness |

## Suggested Manual SSH Order

1. P0: M1.5 B96 e200 `lr=3e-4`, then `lr=1e-4`, then `warmup_cosine`.
2. P0: M1.25 B96/B128 `lr=3e-4`.
3. P0: M1 B192 seed1/seed2 to estimate baseline variance.
4. P1: clip/weight-decay and light background controls only after the P0 shape
   and bin0 tradeoffs are clear.
5. P2: very low LR, rapid decay, and extra M1.5 seeds if P0/P1 leave an
   actionable direction.

## Long Epoch / Overshoot-Control Configs

Context: M1.5 B96 warmup-cosine e200 is the current stable candidate, with
better scalar/stress loss than the M1 B192 e200 baseline but remaining stress
amplitude overshoot. These configs are prepared for manual SSH runs only.

| priority | config | model | batch | epochs | lr/schedule | regularization/loss variant | purpose |
|---|---|---:|---:|---:|---|---|---|
| P0 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | test whether warmup e200 is undertrained |
| P0 | `frozen_v1_e400_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 400 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | longer continuation if e300 still helps |
| P0 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed1.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | candidate stability seed1 |
| P0 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed2.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | candidate stability seed2 |
| P1 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_clip0p5_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | clip=0.5 | reduce stress amplitude/variance overshoot |
| P2 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_clip0p1_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | clip=0.1 | strong clipping stress test |
| P1 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-3 | regularization for overshoot control |
| P2 | `frozen_v1_e300_adamw_m15_B96_base_mse_warmup_cosine_wd1e2_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-2 | high regularization stress test |
| P2 | `frozen_v1_e300_adamw_m15_B96_light_bg_over_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | light bg over=0.01 | suppress low-DeltaT overprediction |
| P2 | `frozen_v1_e300_adamw_m15_B96_light_bg_bias_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | light bg bias=0.01 | suppress low-DeltaT signed bias |
| P2 | `frozen_v1_e300_adamw_m15_B96_light_bg_bias_over_warmup_cosine_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / warmup_cosine | light bg bias=0.01, over=0.01 | combined light background control |
| P0 | `frozen_v1_e300_adamw_m125_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.25 | 96 | 300 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | intermediate capacity, safer memory |
| P0 | `frozen_v1_e300_adamw_m125_B128_base_mse_warmup_cosine_stratified_seed0.yaml` | M1.25 | 128 | 300 | 3e-4 / warmup_cosine | wd=1e-4, clip=1.0, base MSE | intermediate capacity, B128 if memory allows |
| P1 | `frozen_v1_e300_adamw_m125_B96_base_mse_lr3e4_stratified_seed0.yaml` | M1.25 | 96 | 300 | 3e-4 / constant | wd=1e-4, clip=1.0, base MSE | intermediate constant-LR control |
| P1 | `frozen_v1_e300_adamw_m125_B128_base_mse_lr3e4_stratified_seed0.yaml` | M1.25 | 128 | 300 | 3e-4 / constant | wd=1e-4, clip=1.0, base MSE | intermediate B128 constant-LR control |
| P2 | `frozen_v1_e300_adamw_m15_B96_base_mse_lr3e4_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / constant | wd=1e-4, clip=1.0, base MSE | best-checkpoint route; inspect best not only final |
| P2 | `frozen_v1_e300_adamw_m15_B96_base_mse_lr3e4_clip0p5_stratified_seed0.yaml` | M1.5 | 96 | 300 | 3e-4 / constant | clip=0.5 | stabilize constant-LR best-checkpoint route |

## Long Config Suggested SSH Order

1. P0: M1.5 B96 warmup e300 seed0, then e400 only if e300 still improves
   without unacceptable stress amplitude/variance overshoot.
2. P0: M1.5 B96 warmup e300 seed1/seed2 if seed0 remains a candidate.
3. P0: M1.25 B96/B128 warmup e300 to test intermediate capacity.
4. P1: clip0.5 and wd1e-3 overshoot controls.
5. P2: light background and constant-LR best-checkpoint route after the
   warmup/capacity direction is clear.
