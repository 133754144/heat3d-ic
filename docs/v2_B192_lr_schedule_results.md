# Heat3D v2 B192 LR schedule results

Scope: research-stage diagnostic review only. This round kept B192, base MSE, AdamW, seed 0, and the M1 model fixed. It tested whether LR scheduling reduces `best_epoch=1` behavior or final degradation.

## Monitor additions

The runner now writes these lightweight fields to `run_config.json` and `loss_summary.json`:

- `initial_valid_loss`
- `initial_valid_raw_deltaT_mse`
- `updates_per_epoch`
- `total_update_count`
- `final_best_ratio`
- `epoch_lrs`

For the new B192 runs, `updates_per_epoch=4` and `total_update_count=200`. The older baseline was run before these monitor fields existed, so its initial valid values and total update count are not present in its JSON.

Epoch mean train mini-batch loss was not added in this round because doing it precisely would add per-batch host synchronization or require a larger train-loop refactor.

## Runs

Baseline:

- Run: `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_base_mse_seed0`
- Config: `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_seed0.yaml`
- Setting: B192 + base MSE + AdamW lr=3e-4 + constant LR

New schedule runs:

- `output/heat3d_v2_runs/m1_B192_base_mse_rapid_decay_seed0`
- `output/heat3d_v2_runs/m1_B192_base_mse_warmup_cosine_seed0`

Both SSH WSL runs completed with `status_ok=True` and `grad_finite=True`. Best selection was rechecked from `epoch_history`; all three summaries correctly select epoch 1 as the minimum `valid_loss`.

## Results

| run | schedule | best_epoch | initial_valid_loss | epoch1_valid_loss | best_valid_loss | final_valid_loss | final/best | best_raw_deltaT_mse | final_raw_deltaT_mse | total_updates | lr first/last | wall_clock_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | constant | 1 | NA | 0.65134460 | 0.65134460 | 1.641265 | 2.519811 | 0.00125125 | 0.00315291 | NA | 3e-4 / 3e-4 | 559.17 |
| rapid_decay | rapid_decay | 1 | 0.75285327 | 0.65134501 | 0.65134501 | 1.855216 | 2.848284 | 0.00125125 | 0.00356392 | 200 | 3e-4 / 1e-5 | 506.56 |
| warmup_cosine | warmup_cosine | 1 | 0.75285167 | 0.66193140 | 0.66193140 | 1.978332 | 2.988727 | 0.00127159 | 0.00380043 | 200 | 1.0067e-4 / 1e-6 | 511.28 |

## Epoch0 / epoch1 comparison

The new monitor shows the initialized model starts around `initial_valid_loss=0.75285`.

- `rapid_decay` reaches `epoch1_valid_loss=0.65135`, matching the constant-LR baseline, but then degrades to `final_valid_loss=1.85522`.
- `warmup_cosine` reaches only `epoch1_valid_loss=0.66193`, slightly worse than baseline, and degrades to `final_valid_loss=1.97833`.

Neither schedule avoids early overshoot. Both still select epoch 1.

## Effectiveness check

The requested effectiveness criteria were:

- `best_epoch > 1`;
- final valid loss improves by at least 15%;
- final/best ratio improves by at least 20%;
- final raw DeltaT MSE clearly improves while best valid loss is not more than 10% worse;
- epoch0/epoch1 comparison shows the schedule avoids early overshoot.

Neither schedule is effective:

- `best_epoch` remains 1 for both.
- Final valid loss is worse than the baseline for both schedules.
- Final/best ratio is worse than the baseline for both schedules.
- Final raw DeltaT MSE is worse than the baseline for both schedules.
- epoch0/epoch1 comparison does not show stabilization.

## Conclusion

B192 e50 with constant LR, rapid decay, or warmup cosine still fails to resolve the final degradation. Under these settings, LR scheduling alone does not fix the B192 optimization issue.

## Next recommendation

Do not run more B192 schedule variants blindly. The next targeted direction should be one of:

1. update-count-equivalent testing, because B192 e50 has only 200 optimizer updates;
2. compiled train-step feasibility, to reduce runner differences versus upstream RIGNO;
3. gradient clipping sensitivity, if kept as a single-variable test.

