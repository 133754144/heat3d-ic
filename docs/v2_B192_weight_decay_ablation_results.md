# Heat3D v2 B192 AdamW weight decay ablation results

Scope: research-stage diagnostic review only. These runs test whether AdamW `weight_decay` is the main cause of B192 late-epoch degradation. No model, loss, batch size, or learning rate change was made.

## Setup

Baseline:

- Run: `output/heat3d_v2_runs/m1_batch_e50_lr3e4_B192_base_mse_seed0`
- Config: `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_seed0.yaml`
- Setting: B192 + base MSE + AdamW lr=3e-4 + weight_decay=1e-4

New runs:

- `output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_wd1e8_seed0`
- `output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_wd0_seed0`

Both SSH WSL runs completed with `status_ok=True` and `grad_finite=True`. Best selection was rechecked from `epoch_history`; all three summaries correctly select epoch 1 as the minimum `valid_loss`.

## Results

| run | weight_decay | best_epoch | best_valid_loss | final_valid_loss | final/best | best_raw_deltaT_mse | final_raw_deltaT_mse | best_hotspot_mae | final_hotspot_mae | best_bg_bias | final_bg_bias | wall_clock_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1e-4 | 1 | 0.65134460 | 1.641265 | 2.519811 | 0.00125125 | 0.00315291 | 0.05070969 | 0.04226543 | 0.02561660 | 0.03149801 | 559.17 |
| wd1e-8 | 1e-8 | 1 | 0.65134567 | 1.628325 | 2.499941 | 0.00125125 | 0.00312806 | 0.05070952 | 0.04224536 | 0.02561668 | 0.03139802 | 506.98 |
| wd0 | 0 | 1 | 0.65134448 | 1.630802 | 2.503747 | 0.00125125 | 0.00313281 | 0.05070959 | 0.04225112 | 0.02561664 | 0.03141956 | 503.48 |

## Effectiveness check

The requested effectiveness criteria were:

- `best_epoch > 1`;
- final valid loss improves by at least 15%;
- final/best ratio improves by at least 20%;
- final raw DeltaT MSE is clearly lower while best valid loss is not more than 10% worse.

Neither new run satisfies the meaningful criteria:

- `best_epoch` stays at 1 for both runs.
- Final valid loss improves only about 0.8% for wd1e-8 and 0.6% for wd0.
- Final/best ratio improves only about 0.8% and 0.6%.
- Final raw DeltaT MSE is numerically lower by less than 1%, which is not a clear reduction.

## Conclusion

AdamW `weight_decay=1e-4` alone does not explain the B192 late-epoch degradation.

Reducing weight decay to `1e-8` or `0` leaves the same qualitative behavior:

- best-valid remains at epoch 1;
- final metrics still degrade by about 2.5x relative to best;
- best-valid and early validation metrics are effectively unchanged.

No follow-up runs were launched, because neither weight-decay ablation met the effectiveness threshold.

## Next recommendation

Do not continue weight-decay-only B192 sweeps. The next targeted diagnostic should test a different hypothesis, preferably one at a time:

- update-count equivalence for B192 versus B4;
- LR schedule rather than constant LR;
- gradient clipping sensitivity;
- runner monitoring of per-epoch batch loss and component trends.

