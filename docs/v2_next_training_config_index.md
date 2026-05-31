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

## M2 Capacity Probe and Long Training Configs

Context: current best candidate is M1.5 B96 base-MSE warmup-cosine e400 seed0.
M2 configs keep batch size at 96 and test whether larger capacity improves
scalar/stress loss without making stress amplitude or field variance overshoot
unacceptable.

| priority | config | model | node/edge/steps | batch | epochs | schedule | wd/clip | purpose | OOM risk | suggested SSH order |
|---|---|---|---:|---:|---:|---|---|---|---|---:|
| P0 | `frozen_v1_e005_adamw_m2lite_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-lite-width | 112/112/6 | 96 | 5 | warmup_cosine | 1e-4 / 1.0 | quick OOM probe for width-only scaling | low-medium | 1 |
| P0 | `frozen_v1_e005_adamw_m2width_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-width | 128/128/6 | 96 | 5 | warmup_cosine | 1e-4 / 1.0 | quick OOM probe for width 128 | medium | 2 |
| P0 | `frozen_v1_e005_adamw_m2depthlite_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-lite-depth | 112/112/8 | 96 | 5 | warmup_cosine | 1e-4 / 1.0 | quick OOM probe for extra processor steps | medium-high | 3 |
| P2 | `frozen_v1_e005_adamw_m2risk_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-risk | 128/128/8 | 96 | 5 | warmup_cosine | 1e-4 / 1.0 | high-risk OOM probe only | high | 4 |
| P0 | `frozen_v1_e400_adamw_m2lite_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-lite-width | 112/112/6 | 96 | 400 | warmup_cosine | 1e-4 / 1.0 | lowest-risk M2 long run | low-medium | 5 |
| P0 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-width | 128/128/6 | 96 | 400 | warmup_cosine | 1e-4 / 1.0 | test upstream-style latent width | medium | 6 |
| P0 | `frozen_v1_e400_adamw_m2depthlite_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-lite-depth | 112/112/8 | 96 | 400 | warmup_cosine | 1e-4 / 1.0 | test whether more processor steps help more than width | medium-high | 7 |
| P1 | `frozen_v1_e400_adamw_m2lite_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | M2-lite-width | 112/112/6 | 96 | 400 | warmup_cosine | 1e-3 / 1.0 | M2-lite with overshoot-control regularization | low-medium | 8 |
| P1 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | M2-width | 128/128/6 | 96 | 400 | warmup_cosine | 1e-3 / 1.0 | M2-width with overshoot-control regularization | medium | 9 |
| P2 | `frozen_v1_e300_adamw_m2risk_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-risk | 128/128/8 | 96 | 300 | warmup_cosine | 1e-4 / 1.0 | only after e005 risk probe passes with memory headroom | high | 10 |

M2 suggested SSH order:

1. Run e005 probes first: M2-lite-width, M2-width, M2-lite-depth, then M2-risk.
2. If probes do not OOM, run M2-lite e400 first.
3. Then run M2-width e400.
4. Then run M2-lite-depth e400.
5. If overshoot is high, run the wd1e-3 versions.
6. Consider M2-risk e300 only if all probes pass and GPU memory headroom is clear.
