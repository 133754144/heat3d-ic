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

## MLP-Depth Probe Configs

Context: upstream RIGNO exposes `mlp_hidden_layers`, with default/example value
1. Current Heat3D v2 M1.5 uses `mlp_hidden_layers=2`, so deeper MLPs are
exploratory rather than an upstream-default alignment.

| priority | config | model | node/edge/steps/mlp | batch | epochs | schedule | purpose | OOM risk | suggested SSH order |
|---|---|---|---:|---:|---:|---|---|---|---:|
| P0 | `frozen_v1_e002_adamw_m15_B96_base_mse_warmup_cosine_mlp3_probe_seed0.yaml` | M1.5-MLP3 | 96/96/6/3 | 96 | 2 | warmup_cosine | quick feasibility probe for one deeper MLP layer | medium | 1 |
| P1 | `frozen_v1_e400_adamw_m15_B96_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml` | M1.5-MLP3 | 96/96/6/3 | 96 | 400 | warmup_cosine | long run only because the e002 probe completed without OOM | medium | 2 |

MLP-depth note: `mlp_hidden_layers=4` was tested as a two-epoch feasibility
probe on SSH and OOMed during the first training gradient step, so no mlp4
e400 candidate is listed.

## Loss Reduction Next Configs

Context: M1.5 mlp3 e400 lowered `valid_stress` versus M1.5 mlp2 e400 but did
not beat M2-width on scalar loss. M2-width remains the strongest scalar-loss
path, while both routes still need amplitude and field-variance monitoring.

| priority | config | model | batch | epochs | schedule | regularization | purpose |
|---|---|---|---:|---:|---|---|---|
| P0 | `frozen_v1_e400_adamw_m15_B96_base_mse_warmup_cosine_mlp3_stratified_seed1.yaml` | M1.5 mlp3 | 96 | 400 | warmup_cosine | wd=1e-4, clip=1.0 | mlp3 seed stability |
| P0 | `frozen_v1_e400_adamw_m15_B96_base_mse_warmup_cosine_mlp3_stratified_seed2.yaml` | M1.5 mlp3 | 96 | 400 | warmup_cosine | wd=1e-4, clip=1.0 | mlp3 seed stability |
| P0 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed1.yaml` | M2-width | 96 | 400 | warmup_cosine | wd=1e-4, clip=1.0 | M2 scalar-path seed stability |
| P0 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed2.yaml` | M2-width | 96 | 400 | warmup_cosine | wd=1e-4, clip=1.0 | M2 scalar-path seed stability |
| P1 | `frozen_v1_e600_adamw_m15_B96_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml` | M1.5 mlp3 | 96 | 600 | warmup_cosine | wd=1e-4, clip=1.0 | check if mlp3 continues improving after e400 |
| P1 | `frozen_v1_e600_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-width | 96 | 600 | warmup_cosine | wd=1e-4, clip=1.0 | check if M2 scalar gain continues after e400 |
| P1 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | M2-width | 96 | 400 | warmup_cosine | wd=1e-3, clip=1.0 | overshoot-control regularization |
| P2 | `frozen_v1_e500_adamw_m15_B96_base_mse_warmup_cosine_minlr1e5_mlp3_stratified_seed0.yaml` | M1.5 mlp3 | 96 | 500 | warmup_cosine | min_lr=1e-5 | schedule variant for lower final loss |
| P2 | `frozen_v1_e500_adamw_m2width_B96_base_mse_warmup_cosine_minlr1e5_stratified_seed0.yaml` | M2-width | 96 | 500 | warmup_cosine | min_lr=1e-5 | schedule variant for lower final loss |
| P2 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_clip0p5_stratified_seed0.yaml` | M2-width | 96 | 400 | warmup_cosine | wd=1e-4, clip=0.5 | clipping control for amplitude/variance overshoot |

## B48 Capacity Probe and Long Configs

Context: B96 reached capacity limits for `steps=8` and `mlp_hidden_layers=4`.
B48 probes test whether a smaller batch restores memory headroom for wider,
deeper, or MLP-deeper candidates. Probe results are feasibility checks only,
not performance conclusions.

| priority | config | model | node/edge/steps/mlp | batch | epochs | purpose |
|---|---|---|---:|---:|---:|---|
| P0 | `frozen_v1_e002_adamw_m2width_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-width B48 | 128/128/6/2 | 48 | 2 | B48 control probe |
| P0 | `frozen_v1_e002_adamw_m25width_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2.5-width B48 | 160/160/6/2 | 48 | 2 | width 160 feasibility |
| P0 | `frozen_v1_e002_adamw_m3width_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M3-width B48 | 192/192/6/2 | 48 | 2 | high-risk width 192 feasibility |
| P0 | `frozen_v1_e002_adamw_m2depth_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2-depth B48 | 128/128/8/2 | 48 | 2 | test whether B48 restores steps8 feasibility |
| P0 | `frozen_v1_e002_adamw_m2width_B48_base_mse_warmup_cosine_mlp3_capacity_probe_seed0.yaml` | M2-width B48 mlp3 | 128/128/6/3 | 48 | 2 | test whether B48 restores M2 mlp3 feasibility |
| P1 | `frozen_v1_e400_adamw_m2width_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-width B48 | 128/128/6/2 | 48 | 400 | B48 long-run control |
| P1 | `frozen_v1_e400_adamw_m25width_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M2.5-width B48 | 160/160/6/2 | 48 | 400 | width 160 long-run candidate |
| P2 | `frozen_v1_e400_adamw_m3width_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M3-width B48 | 192/192/6/2 | 48 | 400 | high-risk width 192 long-run candidate |
| P1 | `frozen_v1_e400_adamw_m2depth_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M2-depth B48 | 128/128/8/2 | 48 | 400 | steps8 long-run candidate |
| P1 | `frozen_v1_e400_adamw_m2width_B48_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml` | M2-width B48 mlp3 | 128/128/6/3 | 48 | 400 | M2-width plus mlp3 long-run candidate |

B48 suggested SSH order: run the B48 e400 control first, then M2-depth B48
or M2.5-width B48, and only consider M3-width if memory/time budget remains.

## Large B48 Capacity Probe and Long Configs

Context: the first B48 probe batch showed that `192/192/steps6/mlp2`,
`128/128/steps8/mlp2`, and `128/128/steps6/mlp3` are feasible for two-epoch
checks. The larger probes below test the next capacity boundary. Probe results
remain feasibility-only; e400 configs are prepared only for successful probes.

| priority | config | model | node/edge/steps/mlp | batch | epochs | probe result | purpose |
|---|---|---|---:|---:|---:|---|---|
| P0 | `frozen_v1_e002_adamw_m3width_B48_base_mse_warmup_cosine_mlp3_capacity_probe_seed0.yaml` | M3+mlp3 B48 | 192/192/6/3 | 48 | 2 | passed | key feasibility check for M3 plus mlp3 |
| P0 | `frozen_v1_e002_adamw_m25depth_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M2.5-depth B48 | 160/160/8/2 | 48 | 2 | passed | test width 160 plus steps8 |
| P1 | `frozen_v1_e002_adamw_m3depth_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M3-depth B48 | 192/192/8/2 | 48 | 2 | passed | high-risk width 192 plus steps8 |
| P1 | `frozen_v1_e002_adamw_m35width_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml` | M3.5-width B48 | 224/224/6/2 | 48 | 2 | passed | test width 224 upper bound |
| P2 | `frozen_v1_e002_adamw_m35width_B48_base_mse_warmup_cosine_mlp3_capacity_probe_seed0.yaml` | M3.5+mlp3 B48 | 224/224/6/3 | 48 | 2 | OOM | extreme width plus mlp3; do not long-run |
| P1 | `frozen_v1_e400_adamw_m3width_B48_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml` | M3+mlp3 B48 | 192/192/6/3 | 48 | 400 | prepared | long-run candidate after probe success |
| P1 | `frozen_v1_e400_adamw_m25depth_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M2.5-depth B48 | 160/160/8/2 | 48 | 400 | prepared | steps8 long-run candidate |
| P2 | `frozen_v1_e400_adamw_m3depth_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M3-depth B48 | 192/192/8/2 | 48 | 400 | prepared | high-risk steps8 long-run candidate |
| P1 | `frozen_v1_e400_adamw_m35width_B48_base_mse_warmup_cosine_stratified_seed0.yaml` | M3.5-width B48 | 224/224/6/2 | 48 | 400 | prepared | widest successful steps6 long-run candidate |

Large B48 suggested SSH order: first run M3+mlp3 or M2.5-depth e400, then
M3.5-width e400 if memory/time allows. Do not run M3.5+mlp3 e400 because the
two-epoch probe OOMed.

## M2.5 B48 Follow-up Configs

Context: B48 capacity review found `M2.5 B48 = 160/160/steps6/mlp2` to be the
current strongest scalar candidate. These configs test whether that result is
stable across seeds and whether conservative overshoot controls help without
turning `valid_stress` into the primary validation split.

| priority | config | model | batch | epochs | schedule | variant | purpose |
|---|---|---:|---:|---:|---|---|---|
| P0 | `frozen_v1_e400_adamw_m25width_B48_base_mse_warmup_cosine_stratified_seed1.yaml` | 160/160/s6/m2 | 48 | 400 | warmup_cosine | seed1 | seed stability for current scalar candidate |
| P0 | `frozen_v1_e400_adamw_m25width_B48_base_mse_warmup_cosine_stratified_seed2.yaml` | 160/160/s6/m2 | 48 | 400 | warmup_cosine | seed2 | seed stability for current scalar candidate |
| P1 | `frozen_v1_e400_adamw_m25width_B48_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | 160/160/s6/m2 | 48 | 400 | warmup_cosine | wd=1e-3 | conservative regularization for amplitude/variance overshoot |
| P1 | `frozen_v1_e400_adamw_m25width_B48_light_bg_bias_over_warmup_cosine_stratified_seed0.yaml` | 160/160/s6/m2 | 48 | 400 | warmup_cosine | bg bias=0.01, bg over=0.01 | light low-DeltaT/bin0 overprediction control |

Suggested SSH order:

1. Run seed1 and seed2 first.
2. If seed stability is acceptable, run the `wd1e-3` variant.
3. Run the light background bias/over variant only after comparing scalar,
   field-shape, and bin0/le0.05 diagnostics for the first three runs.

## B96 Follow-up Configs After M2.5 Probe

Context: the B96 line is preferred for follow-up tuning because B48 runs are
slower. A two-epoch feasibility probe for `M2.5 B96 = 160/160/steps6/mlp2`
OOMed during the first training epoch, so no M2.5 B96 e400 config is prepared.
The B96 follow-up set therefore stays on known-feasible `M2-width B96 =
128/128/steps6/mlp2`.

| priority | config | model | batch | epochs | schedule | variant | purpose |
|---|---|---:|---:|---:|---|---|---|
| P0 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed1.yaml` | 128/128/s6/m2 | 96 | 400 | warmup_cosine | seed1 | seed stability for B96 M2-width |
| P0 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed2.yaml` | 128/128/s6/m2 | 96 | 400 | warmup_cosine | seed2 | seed stability for B96 M2-width |
| P1 | `frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml` | 128/128/s6/m2 | 96 | 400 | warmup_cosine | wd=1e-3 | conservative regularization for amplitude/variance overshoot |
| P1 | `frozen_v1_e400_adamw_m2width_B96_light_bg_bias_over_warmup_cosine_stratified_seed0.yaml` | 128/128/s6/m2 | 96 | 400 | warmup_cosine | bg bias=0.01, bg over=0.01 | light low-DeltaT/bin0 overprediction control |

Suggested SSH order:

1. Run M2-width B96 seed1 and seed2 first.
2. If seed stability is acceptable, run the `wd1e-3` variant.
3. Run the light background bias/over variant after comparing scalar,
   field-shape, bin0, and le0.05 diagnostics.
4. Do not run M2.5 B96 long training unless a future smaller-memory strategy
   makes the two-epoch probe pass without OOM.
