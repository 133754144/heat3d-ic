# Heat3D v3 Prediction Mechanism Diagnostics

## Purpose

`scripts/analyze_heat3d_v3_prediction_mechanisms.py` is a read-only diagnostic
for existing `predictions.npz` or `best_predictions.npz` files. It helps answer
whether a trained run is mainly failing by amplitude, shape, hotspot placement,
or specific data conditions.

This is diagnostic tooling only. It does not train, build graphs, execute the
model, or change decoder/loss/objective behavior.

## Metrics

- `pred_mean / target_mean`: mean predicted and true `DeltaT` per sample.
- `pred_std / target_std`: predicted and true spatial standard deviation.
- `amplitude_ratio`: predicted `DeltaT` range divided by true `DeltaT` range.
- `mean_bias`: mean `pred_deltaT - true_deltaT`.
- `centered_corr`: centered spatial correlation between predicted and true
  `DeltaT`.
- `zscore_rmse`: RMSE after per-sample z-score normalization; this isolates
  shape mismatch from amplitude scale.
- `top_k_overlap`: overlap between true and predicted top hotspot nodes.
- `hotspot_centroid_distance`: distance between true and predicted top-k
  hotspot centroids in coordinate space.
- `peak_abs_error / peak_rel_error`: peak `DeltaT` mismatch.

## Grouping

The script reports overall metrics, per-sample metrics, and grouped metrics for:

- `split`
- `source_category`
- `q_power_range`
- `k_mode`
- `k_region_mode`
- `bc_category`

It also emits top weak amplitude, shape, and hotspot groups.

## Usage

Initial targets are W1, L2, S1, and B6 final/best predictions. S2/S3 can be
added after their runs complete. Outputs should stay under ignored `output/`
paths and should not be committed.

Example:

```bash
python3 scripts/analyze_heat3d_v3_prediction_mechanisms.py \
  --run-dir output/heat3d_v2_runs/<run_name> \
  --prediction-name best_predictions.npz \
  --prediction-label best \
  --output-json output/heat3d_v3_prediction_mechanisms/<run_name>_best.json \
  --output-md output/heat3d_v3_prediction_mechanisms/<run_name>_best.md
```

## Interpretation

Use this alongside the long-run audit summary:

- low `amplitude_ratio` with good `centered_corr` suggests amplitude collapse or
  scale underfit;
- poor `centered_corr` or high `zscore_rmse` suggests shape-path failure;
- low `top_k_overlap` or high `hotspot_centroid_distance` suggests local
  hotspot routing/recovery weakness;
- condition-wise weak groups can prioritize P3 decoder/model-path audit without
  changing the model in this step.
