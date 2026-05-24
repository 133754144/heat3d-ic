# Heat3D v2 M1 vs V1 diagnostic reference and LR schedule plan

## Scope

This note consolidates existing M1 mini-batch e50 results and prepares the next LR schedule ablation. It does not introduce a formal benchmark claim. The V1 reference is a diagnostic-stage historical reference, and the V2 results are single-seed research-stage controlled runs.

## V2 M1 Best Checkpoints

| run_name | lr | best_epoch | valid_loss | raw_deltaT_mse | hotspot_mae | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap | bg_bias | pn_over_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m1_batch_e50_seed0` | 1e-3 | 2 | 9.27328050e-01 | 1.37366808e-03 | 6.37404770e-02 | 2.853343 | 0.758506 | 0.872976 | 0.106098 | 0.683008 | 1.21050868e-02 | 9.90977764e-01 |
| `m1_batch_e50_lr3e4_seed0` | 3e-4 | 2 | 8.20047617e-01 | 1.03067153e-03 | 4.21602204e-02 | 1.064125 | 0.756951 | 0.789703 | 0.098449 | 0.633398 | 1.83125418e-02 | 1.00000000e+00 |
| `m1_batch_e50_lr1e4_seed0` | 1e-4 | 20 | 1.03246975e+00 | 1.60945079e-03 | 6.16895109e-02 | 4.067817 | 0.856934 | 1.342208 | 0.092087 | 0.634961 | 1.13554373e-02 | 9.83587384e-01 |

## V2 M1 Final Checkpoints

| run_name | lr | final_valid_loss | final_raw_deltaT_mse | final_hotspot_mae | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap | final/best degradation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m1_batch_e50_seed0` | 1e-3 | 3.14366198e+00 | 5.30293537e-03 | 1.08158998e-01 | 10.153590 | 0.871330 | 1.532501 | 0.093611 | 0.644141 | 2.216334 |
| `m1_batch_e50_lr3e4_seed0` | 3e-4 | 3.16040659e+00 | 5.32081956e-03 | 1.09024994e-01 | 10.084601 | 0.877626 | 1.498525 | 0.083979 | 0.658984 | 2.340359 |
| `m1_batch_e50_lr1e4_seed0` | 1e-4 | 2.86302805e+00 | 4.82248934e-03 | 1.05052084e-01 | 9.641037 | 0.870452 | 1.528010 | 0.095782 | 0.640039 | 1.830558 |

## Frozen V1 Diagnostic Reference

Source: `configs/heat3d_v2/frozen_v1_reference.yaml`.

| reference | best_epoch | optimizer | lr | best_overall_rmse | best_overall_mae | best_valid_rmse | best_valid_mae | bin_0_bias | bin_0_over_ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_v1_best_diagnostic` | 33 | manual_full_batch_gradient_descent | 1.0e-2 | 3.94142446e-02 | 2.46786651e-02 | 2.73560372e-02 | 2.30636740e-02 | 1.89761732e-02 | 1.0 |

The V1 reference is diagnostic-stage historical evidence, not a formal benchmark. It does not include `field_variance_ratio`, `centered_spatial_correlation`, `amplitude_ratio`, or `top_k_overlap`. Therefore field-shape metrics can be compared across the three V2 M1 runs, but not as a complete field-shape comparison against V1.

## Research Interpretation

除 loss 外，整体热场复现最好的一组取决于指标。按 best checkpoint 的 overall DeltaT diagnostics，`lr=1e-4` previously showed the lowest DeltaT RMSE/MAE and the best centered spatial correlation, but it is worse on valid loss and raw DeltaT MSE. 按 best-valid 综合指标，`lr=3e-4` 更可靠。

对高温 hotspot / peak，`lr=3e-4` has the best best-valid hotspot MAE, while `lr=1e-4` has the lowest best-prediction peak_abs_error. `lr=1e-3` still has the highest best top_k_overlap. This is mixed evidence, not a stable winner.

对低温区域过预测，`lr=1e-4` has lower best bg_bias and pn_over_ratio than `lr=3e-4`, but low-temperature overprediction is not solved. `lr=3e-4` still has pn_over_ratio at 1.0 at best checkpoint.

valid_loss and field-shape metrics are not fully consistent. `lr=3e-4` is best by valid_loss/raw_deltaT_mse/hotspot_mae, while `lr=1e-4` is better on some spatial/peak/overall DeltaT diagnostics but overshoots variance and amplitude.

当前不能声称 V2 稳定优于 V1。V2 has stronger infrastructure and promising single-seed diagnostics, but there is no seed sensitivity for M1, final degradation remains large, and V1/V2 metrics are not one-to-one because V1 lacks field-shape diagnostics.

下一步应测试 LR schedule，而不是更低 constant lr. `lr=3e-4` learns best early but keeps an early best epoch; `lr=1e-4` delays the best epoch but weakens key valid metrics. A second-stage decay can test whether early learning at 3e-4 plus later stabilization at 1e-4 reduces final degradation without giving up the early best-valid quality.

## Planned Second-stage LR Ablation

New config:

`configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_decay_e5_to1e4_seed0.yaml`

Schedule:

- epochs 1-4: lr=3e-4
- epochs 5-50: lr=1e-4

Expected output directory:

`output/heat3d_v2_runs/m1_batch_e50_lr3e4_decay_e5_to1e4_seed0`

This remains a single-seed diagnostic ablation. It should not be treated as a benchmark or a stable model-performance claim.
