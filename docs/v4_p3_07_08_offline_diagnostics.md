# V4P3_07/08 Split-Aware Offline Diagnostics

Read this file only for V4P3_07/08 offline diagnostic decisions or follow-up
config selection.

## Scope

- Checkpoints: `output/heat3d_v4_runs/V4P3_07/params_best.pkl`,
  `output/heat3d_v4_runs/V4P3_08/params_best.pkl`.
- Split map:
  `configs/heat3d_v4/candidate1024_v0_train768_valid128_test128_stratified_seed0.json`.
- Prediction exports were written only under ignored
  `output/heat3d_v4_offline_diagnostics/<config>/best_<split>/`.
- No retraining, tmux launch, or tracked output artifact was used.

## Best-Checkpoint Training Context

| config | best epoch | best valid_base_mse | best raw DeltaT RMSE K | best rel_rmse_v4_pct | final valid_base_mse | all_groups |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V4P3_07 | 103 | 0.328 | 1.617 | 392.663 | 0.414 | skipped |
| V4P3_08 | 129 | 0.376 | 1.732 | 420.393 | 0.435 | skipped |

The original training selection metric still favors V4P3_07. Both 600-epoch
runs overfit after the early best checkpoint.

## Offline Split Metrics

| config | split | samples | RMSE K | MAE K | rel_rmse_v4_pct | corr | cosine | amp ratio | top5 overlap | p95 abs K | peak abs K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P3_07 | valid_iid | 128 | 0.652 | 0.291 | 275.347 | 0.756 | 0.817 | 2.233 | 0.531 | 1.055 | 4.366 |
| V4P3_07 | test_iid | 128 | 0.670 | 0.334 | 262.298 | 0.751 | 0.818 | 2.155 | 0.521 | 1.311 | 3.476 |
| V4P3_08 | valid_iid | 128 | 0.621 | 0.272 | 249.033 | 0.784 | 0.855 | 2.043 | 0.558 | 0.945 | 4.464 |
| V4P3_08 | test_iid | 128 | 0.603 | 0.300 | 238.435 | 0.773 | 0.850 | 1.971 | 0.575 | 1.161 | 3.068 |

On split-aware offline predictions, V4P3_08 is better on both valid_iid and
test_iid for recovered-temperature RMSE, shape correlation, cosine similarity,
amplitude ratio closeness to 1, top-k overlap, and test peak absolute error.
V4P3_07 remains better only by the original training selection metric.

## Failure Modes

| config | split | bin0 over | le0.05 over | strong-q RMSE K | top5 RMSE K | background RMSE K | q-bin weak point |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| V4P3_07 | valid_iid | 0.093 | 0.419 | 3.678 | 2.089 | 0.130 | q_power_bin_2 strong-q RMSE 6.682 |
| V4P3_07 | test_iid | 0.087 | 0.427 | 2.911 | 2.115 | 0.136 | q_power_bin_2 strong-q RMSE 4.798 |
| V4P3_08 | valid_iid | 0.485 | 0.702 | 3.844 | 2.122 | 0.122 | q_power_bin_2 strong-q RMSE 6.983 |
| V4P3_08 | test_iid | 0.510 | 0.705 | 2.563 | 1.916 | 0.136 | q_power_bin_2 strong-q RMSE 4.401 |

Main findings:

- V4P3_08 improves split-level shape and scalar accuracy but has much heavier
  low-DeltaT overprediction than V4P3_07.
- Both runs still underperform in strong-q and top-DeltaT regions, especially
  q_power_bin_2 strong-q samples.
- Hotspot location is not the dominant failure because top-k overlap improves
  in V4P3_08; the harder issue is amplitude calibration across background,
  strong-q, and high-DeltaT regions.
- The tracked metadata available to these diagnostics did not expose reliable
  `qc_class`, `DeltaT_bin`, `q_family`, `cooling_regime`, `diag3_policy`, or
  `high_deltaT_triage` groups. The reliable weak group here is the V3-style
  region/q_power decomposition.

## Next Configs

The latest follow-up instruction requested a message-passing-depth ablation on
the current V4P3_07 and V4P3_08 bases. Therefore:

- `V4P3_09` = `V4P3_07` with `processor_steps=8` instead of 6.
- `V4P3_10` = `V4P3_08` with `processor_steps=8` instead of 6.

Both inherit the 07/08 600-epoch schedule, formal split map, B32 training,
`prediction_split=valid_iid`, and memory-optimized non-all prediction behavior.
Feature-transform and background-loss ablations remain the next logical
direction after this message-passing pair, because the diagnostics show both
shape gains and low-DeltaT/background bias.
