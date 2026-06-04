# Heat3D v2 B192 train monitor and clip ablation results

Scope: research-stage diagnostic review only. This round kept B192, base MSE, lr=3e-4, AdamW, seed 0, and the M1 model fixed. It tested whether stricter gradient clipping reduces B192 late-epoch degradation and whether train loss rises with validation loss.

## Monitor additions

The runner now writes per-epoch lightweight monitor fields to `loss_summary.json`:

- `epoch_mean_train_batch_loss`
- `epoch_min_train_batch_loss`
- `epoch_max_train_batch_loss`
- `epoch_mean_grad_norm`
- `epoch_max_grad_norm`
- `epoch_mean_update_norm`
- `epoch_max_update_norm`
- `epoch_mean_param_norm`
- `epoch_update_to_param_norm_ratio`
- `epoch_max_update_to_param_norm_ratio`

The previous monitor fields remain:

- `updates_per_epoch`
- `total_update_count`
- `epoch_lrs`
- `initial_valid_loss`
- `initial_valid_raw_deltaT_mse`
- `final_best_ratio`

For B192, each epoch has 4 train batches and 200 total updates over e50.

## Runs

Baseline:

- Run: `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_base_mse_seed0`
- Setting: B192 + base MSE + AdamW lr=3e-4 + clip=1.0 + weight_decay=1e-4

New runs:

- `output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_clip05_seed0`
- `output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_clip01_seed0`

Both new runs used `weight_decay=0.0` and completed with `status_ok=True` and `grad_finite=True`. Best selection was rechecked from `epoch_history`; all runs correctly select epoch 1 as the minimum `valid_loss`.

## Results

| run | clip | best_epoch | best_valid_loss | final_valid_loss | final/best | final_raw_deltaT_mse | train batch loss e1/e25/e50 | valid loss e1/e25/e50 | grad norm e1/e25/e50 | update norm e1/e25/e50 | update/param e1/e25/e50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.0 | 1 | 0.65134460 | 1.641265 | 2.519811 | 0.00315291 | NA | 0.65134460 / 2.290206 / 1.641265 | NA | NA | NA |
| clip0.5 | 0.5 | 1 | 0.65134126 | 2.645290 | 4.061297 | 0.00508168 | 1.036174 / 0.391726 / 0.353292 | 0.651341 / 2.216492 / 2.645290 | 13.053556 / 1.464297 / 1.693108 | 0.113414 / 0.029727 / 0.027137 | 0.001624 / 0.000425 / 0.000388 |
| clip0.1 | 0.1 | 1 | 0.65133893 | 2.458120 | 3.773950 | 0.00472212 | 1.036178 / 0.391334 / 0.356007 | 0.651339 / 2.222625 / 2.458120 | 13.053639 / 1.495837 / 2.398528 | 0.112561 / 0.029004 / 0.022352 | 0.001612 / 0.000415 / 0.000320 |

## Train loss diagnosis

The new monitor shows train batch loss does not rise.

For both clip runs:

- epoch 1 mean train batch loss is about `1.036`;
- epoch 25 mean train batch loss is about `0.391`;
- epoch 50 mean train batch loss is about `0.35`;
- full train metrics at epoch 25/50 also decrease.

Validation loss rises while train loss decreases. This points away from "optimizer update is directly destroying the train objective" and toward overfitting, split mismatch, or metric alignment issues.

## Grad/update diagnosis

Gradient norm is largest at epoch 1 and then much smaller:

- clip0.5: mean grad norm `13.05 -> 1.46 -> 1.69`
- clip0.1: mean grad norm `13.05 -> 1.50 -> 2.40`

Update norm and update/param ratio also decrease:

- clip0.5 update/param ratio `0.00162 -> 0.00043 -> 0.00039`
- clip0.1 update/param ratio `0.00161 -> 0.00042 -> 0.00032`

There is no evidence from these monitor fields that late degradation is caused by exploding gradients or growing update-to-parameter scale.

## Clip effectiveness

The requested effectiveness criteria were:

- `best_epoch > 1`;
- final valid loss improves by at least 15%;
- final/best ratio improves by at least 20%;
- final raw DeltaT MSE clearly improves while best valid loss is not more than 10% worse.

Neither clip setting is effective:

- `best_epoch` remains 1.
- final valid loss is worse than baseline.
- final/best ratio is worse than baseline.
- final raw DeltaT MSE is worse than baseline.

## Train-valid split diagnostics

The read-only split report in `docs/v2_train_valid_split_diagnostics.md` shows large train/valid distribution differences:

- `low_power_near_zero_background_cases`: train `1`, valid `113`;
- `low_power`: train `1`, valid `113`;
- `high_top_h`: train `116`, valid `127`;
- `diag3`: train `72`, valid `127`;
- `low_k_barrier_or_TIM_variation`: train `2`, valid `67`;
- train raw DeltaT node mean `0.02929`, valid `0.01096`;
- low DeltaT fraction `<=0.01 K`: train `0.390`, valid `0.809`.

This supports the monitor-based diagnosis: the model keeps reducing the training objective while validation loss rises on a validation split that is dominated by low-power, low-DeltaT, high-top-h, diag3 and barrier/TIM-variation cases.

## Conclusion

Case B applies: train loss decreases while validation loss rises. The B192 late-epoch problem currently looks more like generalization failure, split mismatch, or selection/validation metric alignment than an optimizer update that is increasing the train objective.

Stricter gradient clipping does not fix B192 degradation under this setup.

## Next recommendation

Do not run more clip-only sweeps. The next targeted direction should be one of:

1. validation/split/metric alignment review using existing predictions;
2. update-count-equivalent comparison, because B192 e50 has only 200 updates;
3. compiled train-step feasibility, to reduce runner differences before more large sweeps.
