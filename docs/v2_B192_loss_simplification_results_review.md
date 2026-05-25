# Heat3D v2 B192 Loss Simplification Results Review

本文整理 SSH WSL 上已有 B192 e50 三组结果。field-shape diagnostics 起初缺失，本轮仅基于已有 `predictions.npz` / `best_predictions.npz` 补跑只读 field-shape diagnostics。没有训练、没有改 output 结果。

## Runs

| label | run_dir | loss | lr | batch | updates/epoch | total updates |
|---|---|---|---:|---:|---:|---:|
| B192 full | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_seed0` | `background_pseudo_negative` | `3e-4` | 192 | 4 | 200 |
| B192 base_mse | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_base_mse_seed0` | `mse` | `3e-4` | 192 | 4 | 200 |
| B192 base_mse_hotspot | `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_base_mse_hotspot_seed0` | `background_hotspot` | `3e-4` | 192 | 4 | 200 |

## Best Checkpoint Comparison

| run | best_epoch | best_valid_loss | best_raw_deltaT_mse | hotspot_mae | bg_bias | pn_over_ratio | field_variance_ratio | centered_corr | amplitude_ratio | peak_abs_error | top_k_overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B192 full | 1 | 1.062913 | 0.00124697 | 0.051253 | 0.025413 | 1.000000 | 0.009926 | 0.456731 | 0.070747 | 0.277017 | 0.488867 |
| B192 base_mse | 1 | 0.651345 | 0.00125125 | 0.050710 | 0.025617 | 0.000000 | 0.015813 | 0.514332 | 0.091855 | 0.272198 | 0.542578 |
| B192 base_mse_hotspot | 1 | 0.732330 | 0.00125852 | 0.050106 | 0.025882 | 0.000000 | 0.025584 | 0.540445 | 0.115408 | 0.266927 | 0.563086 |

## Final Checkpoint Comparison

| run | final_valid_loss | final_raw_deltaT_mse | hotspot_mae | bg_bias | pn_over_ratio | field_variance_ratio | centered_corr | amplitude_ratio | peak_abs_error | top_k_overlap | final/best | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B192 full | 2.913905 | 0.00473993 | 0.049767 | 0.035068 | 1.000000 | 7.329983 | 0.768624 | 1.256147 | 0.087429 | 0.642773 | 2.74x | 565.26s |
| B192 base_mse | 1.641265 | 0.00315291 | 0.042265 | 0.031498 | 0.000000 | 4.627088 | 0.772610 | 1.167643 | 0.089821 | 0.652539 | 2.52x | 559.17s |
| B192 base_mse_hotspot | 2.638273 | 0.00496970 | 0.050312 | 0.037541 | 0.000000 | 6.972240 | 0.795993 | 1.231898 | 0.088944 | 0.634375 | 3.60x | 506.16s |

## Conclusions

1. B192 full composite is not better than B4 `lr=3e-4`. It is much faster, but best_valid_loss is worse and best_epoch is 1.
2. B192 base_mse improves best_valid_loss and final_valid_loss versus B192 full, and has better final raw_deltaT_mse. It still has best_epoch=1 and substantial final degradation.
3. B192 base_mse_hotspot is not better than base_mse overall. It has slightly better best hotspot/field-shape indicators, but final_valid_loss and final degradation are worse.
4. Three runs with best_epoch=1 indicate that simplification alone did not solve sustained optimization. The first epoch is already the best checkpoint under `valid_loss`.
5. Next step should lower LR and test optimizer/update-count hypotheses before further loss complexity. B192 is valuable as a fast ablation platform, but B192 e50 only has 200 optimizer updates.

## Required Next Comparisons

Future B192 pilots must compare:

- best/final valid_loss;
- best/final raw_deltaT_mse;
- hotspot_mae;
- bg_bias and pn_over_ratio;
- field_variance_ratio;
- centered_spatial_correlation;
- amplitude_ratio;
- peak_abs_error;
- top_k_overlap;
- final/best degradation;
- wall-clock;
- update count.
