# Heat3D v2 Training Results Overview

Scope: research-stage diagnostic summary for `medium1024_gapA_full1024_v2`
with the stratified split. This is not a formal benchmark.

## Current Baseline

| role | run | status |
|---|---|---|
| scalar-loss baseline | `m1_B192_base_mse_lr3e4_e200_stratified_seed0` | keep as current baseline |
| replay audit | `m1_B192_base_mse_lr3e4_e200_stratified_replay_seed0` | old e200 reproduced |
| non-baseline extension | `m1_B192_base_mse_lr3e4_e300_stratified_seed0` | not treated as strict e200 continuation |

## Existing Conclusions

| finding | conclusion |
|---|---|
| M1 e200 replay | old M1 B192 e200 is reproducible and remains the scalar-loss baseline. |
| M1 B96 control | B96/update dynamics alone did not explain the M1.5 gain. |
| M1.5 B96 | capacity helps spatial correlation and top-k overlap. |
| M1.5 tradeoff | field variance and amplitude overshoot became worse. |
| low-DeltaT/bin0 | overprediction is reduced in some runs but remains unresolved. |
| M1 e300 | not adopted as a new baseline and should not be treated as strict old e200 extension. |
| M1.5 B192 | known OOM risk; do not prepare B192 M1.5 runs here. |

## Next Training Direction

Prepare, but do not run locally:

- M1.5 B96 e200 capacity continuation.
- M1.5 lower LR and schedule controls.
- M1.5 stronger clip / weight decay regularization controls.
- Light background anti-overshoot controls using existing runner loss modes only.
- M1.25 intermediate-capacity controls.
- M1 B192 seed robustness controls.

The next SSH runs should compare scalar loss, split-aware field-shape metrics,
and bin0/background diagnostics together. A run that improves scalar loss but
worsens variance, amplitude, or low-DeltaT overprediction should be treated as
a tradeoff, not a clean improvement.
