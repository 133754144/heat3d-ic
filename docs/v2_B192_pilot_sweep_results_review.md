# Heat3D v2 B192 pilot sweep results review

Scope: research-stage diagnostic review only. These runs are single-seed controlled experiments on `medium1024_gapA_full1024_v2`, not formal benchmark results.

## Context

The previous B192 loss simplification runs showed fast wall-clock execution but all selected epoch 1 as best-valid. This suggested that B192 is useful as a rapid ablation platform, but lower validation loss after the first epoch was not sustained.

This round tested three lower-LR pilot runs on SSH WSL, sequentially and without parallel training:

- Pilot 1: B192 base MSE only, AdamW lr=1e-4.
- Pilot 2: B192 base MSE only, AdamW lr=3e-5.
- Pilot 3: B192 full composite `background_pseudo_negative`, AdamW lr=1e-4.

All three runs completed with `status_ok=True` and `grad_finite=True`. Best selection was rechecked from `epoch_history`; the summary `best_epoch` matches the minimum `valid_loss` epoch for every run listed below.

## Existing B192 baseline runs

| run | loss | lr | best_epoch | best_valid_loss | final_valid_loss | final/best | best_raw_deltaT_mse | final_raw_deltaT_mse | hotspot best/final | bg_bias best/final | pn_over best/final | elapsed_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B192 full | background_pseudo_negative | 3e-4 | 1 | 1.0629 | 2.9139 | 2.7414 | 0.00124697 | 0.00473993 | 0.051253/0.049767 | 0.025413/0.035068 | 1.0/1.0 | 565.26 |
| B192 base_mse | mse | 3e-4 | 1 | 0.651345 | 1.6413 | 2.5198 | 0.00125125 | 0.00315291 | 0.050710/0.042265 | 0.025617/0.031498 | 0.0/0.0 | 559.17 |
| B192 base_mse_hotspot | background_hotspot | 3e-4 | 1 | 0.732330 | 2.6383 | 3.6026 | 0.00125852 | 0.00496970 | 0.050106/0.050312 | 0.025882/0.037541 | 0.0/0.0 | 506.16 |

## Pilot runs

| run | loss | lr | best_epoch | best_valid_loss | final_valid_loss | final/best | best_raw_deltaT_mse | final_raw_deltaT_mse | hotspot best/final | bg_bias best/final | pn_over best/final | elapsed_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pilot 1 B192 base_mse | mse | 1e-4 | 1 | 0.662072 | 1.9644 | 2.9671 | 0.00127186 | 0.00377368 | 0.050552/0.041839 | 0.025834/0.038169 | 0.0/0.0 | 504.59 |
| Pilot 2 B192 base_mse | mse | 3e-5 | 1 | 0.693995 | 1.9817 | 2.8555 | 0.00133319 | 0.00380690 | 0.048815/0.040517 | 0.027077/0.043210 | 0.0/0.0 | 505.57 |
| Pilot 3 B192 full | background_pseudo_negative | 1e-4 | 1 | 1.0764 | 2.4498 | 2.2759 | 0.00126775 | 0.00383246 | 0.050765/0.043080 | 0.025729/0.036207 | 1.0/1.0 | 510.06 |

All B192 e50 runs use 4 updates/epoch and 200 total optimizer updates over 50 epochs. This is not update-count equivalent to B4 e50.

## Field-shape diagnostics

Read-only field-shape diagnostics were generated for the three pilot runs because predictions already existed and the JSON files were missing. Existing B192 diagnostics are included for comparison.

Best checkpoint:

| run | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap |
|---|---:|---:|---:|---:|---:|
| B192 full lr3e-4 | 0.009926 | 0.456731 | 0.070747 | 0.277017 | 0.488867 |
| B192 base_mse lr3e-4 | 0.015813 | 0.514332 | 0.091855 | 0.272198 | 0.542578 |
| B192 base_mse_hotspot lr3e-4 | 0.025584 | 0.540445 | 0.115408 | 0.266927 | 0.563086 |
| Pilot 1 base_mse lr1e-4 | 0.046305 | 0.252903 | 0.099976 | 0.274380 | 0.273438 |
| Pilot 2 base_mse lr3e-5 | 0.150412 | 0.100369 | 0.167077 | 0.262066 | 0.025000 |
| Pilot 3 full lr1e-4 | 0.043115 | 0.242076 | 0.095864 | 0.275072 | 0.268359 |

Final checkpoint:

| run | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap |
|---|---:|---:|---:|---:|---:|
| B192 full lr3e-4 | 7.3300 | 0.768624 | 1.2561 | 0.087429 | 0.642773 |
| B192 base_mse lr3e-4 | 4.6271 | 0.772610 | 1.1676 | 0.089821 | 0.652539 |
| B192 base_mse_hotspot lr3e-4 | 6.9722 | 0.795993 | 1.2319 | 0.088944 | 0.634375 |
| Pilot 1 base_mse lr1e-4 | 4.5759 | 0.741422 | 1.0757 | 0.099689 | 0.620117 |
| Pilot 2 base_mse lr3e-5 | 3.8867 | 0.682502 | 0.941235 | 0.117800 | 0.611523 |
| Pilot 3 full lr1e-4 | 5.2218 | 0.747381 | 1.1172 | 0.097708 | 0.621289 |

## Adaptive decision

The Phase A pilot outcome matches Case C from the sweep plan:

- Pilot 1, Pilot 2, and Pilot 3 all still select `best_epoch=1`.
- Lowering B192 lr from 3e-4 to 1e-4 or 3e-5 does not move best-valid later.
- Final/best degradation remains large: 2.97x, 2.86x, and 2.28x for the three pilots.
- B192 base_mse lr=3e-4 remains the best B192 run by best and final valid loss among the tested B192 configurations.

Therefore no additional Phase B candidate was launched in this round. Lower LR alone does not fix B192 optimization.

## Interpretation

The pilot results suggest the B192 instability is not mainly caused by the full composite loss being too strong or by lr=3e-4 alone. Even base MSE with lr=1e-4 and 3e-5 keeps the same epoch-1 best-valid behavior.

Possible explanations to validate next:

- B192 has only 4 updates/epoch and 200 updates total, so e50 may be too few updates for a fair large-batch run.
- Current Heat3D runner uses a Python loop around `jax.value_and_grad`, while upstream RIGNO uses a compiled `pmap` train step with optimizer update inside the compiled path.
- Upstream uses a more sophisticated LR schedule and much smaller AdamW weight decay; Heat3D B192 pilots used constant lr and weight decay 1e-4.
- The validation loss and field-shape metrics are not fully aligned. Some final checkpoints improve spatial correlation and top-k overlap while validation loss worsens.
- Low-temperature overprediction is still unresolved for full composite runs: `pn_over_ratio` remains 1.0 for both B192 full lr=3e-4 and full lr=1e-4.

## Recommendation

Do not continue blind lower-LR B192 sweeps. The next controlled work should prioritize:

1. Add better train-step monitoring: update count, per-epoch mean batch loss, lr, component means, reported grad norm, and final/best ratio.
2. Design update-count-equivalent tests, for example B192 with more epochs or B4/B192 matched by total optimizer updates.
3. Validate upstream training-gap hypotheses before larger sweeps: compiled train step feasibility, LR schedule, weight decay sensitivity, and selection metric alignment.
4. Consider a very small Phase B only after monitor improvements, such as B192 base_mse with weight_decay=0 or gradient_clip_norm=0.5, but not as an automatic continuation.

