# Heat3D v2 M1 field-shape metrics review

## Scope

This review reads existing SSH WSL M1 mini-batch e50 outputs only. No training was run, and no model/loss/optimizer/batch-size semantics were changed.

Runs:

- `m1_batch_e50_seed0`: lr=1e-3
- `m1_batch_e50_lr3e4_seed0`: lr=3e-4
- `m1_batch_e50_lr1e4_seed0`: lr=1e-4

All three runs already had `loss_summary.json`, `run_config.json`, `predictions.npz`, `best_predictions.npz`, `field_shape_diagnostics_best.json`, `field_shape_diagnostics_final.json`, `run_analysis_best.json`, `run_analysis_final.json`, `baseline_comparison_best.json`, and `baseline_comparison_final.json`.

## Basic Run Metrics

| run_name | lr | best_epoch | best_valid_loss | best_valid_raw_deltaT_mse | best_valid_hotspot_mae | final_valid_loss | final degradation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m1_batch_e50_seed0` | 1e-3 | 2 | 9.273280e-01 | 1.373668e-03 | 6.374048e-02 | 3.143662e+00 | 2.216334e+00 |
| `m1_batch_e50_lr3e4_seed0` | 3e-4 | 2 | 8.200476e-01 | 1.030672e-03 | 4.216022e-02 | 3.160407e+00 | 2.340359e+00 |
| `m1_batch_e50_lr1e4_seed0` | 1e-4 | 20 | 1.032470e+00 | 1.609451e-03 | 6.168951e-02 | 2.863028e+00 | 1.830558e+00 |

## Best Prediction Field-Shape Metrics

| run_name | lr | best_epoch | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m1_batch_e50_seed0` | 1e-3 | 2 | 2.853343 | 0.758506 | 0.872976 | 0.106098 | 0.683008 |
| `m1_batch_e50_lr3e4_seed0` | 3e-4 | 2 | 1.064125 | 0.756951 | 0.789703 | 0.098449 | 0.633398 |
| `m1_batch_e50_lr1e4_seed0` | 1e-4 | 20 | 4.067817 | 0.856934 | 1.342208 | 0.092087 | 0.634961 |

## Final Prediction Field-Shape Metrics

| run_name | lr | best_epoch | field_variance_ratio | centered_spatial_correlation | amplitude_ratio | peak_abs_error | top_k_overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m1_batch_e50_seed0` | 1e-3 | 2 | 10.153590 | 0.871330 | 1.532501 | 0.093611 | 0.644141 |
| `m1_batch_e50_lr3e4_seed0` | 3e-4 | 2 | 10.084601 | 0.877626 | 1.498525 | 0.083979 | 0.658984 |
| `m1_batch_e50_lr1e4_seed0` | 1e-4 | 20 | 9.641037 | 0.870452 | 1.528010 | 0.095782 | 0.640039 |

## Metric Interpretation

- `field_variance_ratio`: predicted spatial variation relative to the true field. Values closer to 1 are better.
- `centered_spatial_correlation`: spatial shape correlation after mean-centering. Higher is better.
- `amplitude_ratio`: predicted field amplitude relative to true amplitude. Values closer to 1 are better.
- `peak_abs_error`: absolute peak error. Lower is better.
- `top_k_overlap`: overlap of top-k hotspot regions. Higher is better.

## Conclusion

`lr=3e-4` remains the best-valid overall M1 constant-lr run. It has the best `valid_loss`, `valid_raw_deltaT_mse`, and `valid_hotspot_mae`, and its best prediction has the field variance ratio closest to 1. However, its best-prediction top-k overlap is lower than the lr=1e-3 baseline.

`lr=1e-4` moves best_epoch from 2 to 20, but it does not cleanly improve the run. It is worse than lr=3e-4 on best-valid loss, raw DeltaT MSE, and hotspot MAE. Its field-shape diagnostics are mixed: centered correlation and peak error improve, but field variance and amplitude overshoot the true field, and top-k overlap does not beat lr=1e-3.

There is a real mismatch between valid-loss selection and field-shape metrics. The lr=1e-4 best checkpoint looks better on some spatial-shape diagnostics while worse on valid loss and raw DeltaT MSE. Future reviews should report valid loss, raw DeltaT MSE, and field-shape metrics together rather than treating one scalar as sufficient.

The next LR step should be schedule work, not lower constant-lr sweeping. A reasonable next controlled direction is M1 lr=3e-4 with warmup+cosine or a second-stage decay, followed by seed sensitivity once final-vs-best behavior is better controlled.
