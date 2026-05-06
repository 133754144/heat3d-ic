# Heat3D v1 Physics-Label Medium Baseline Tooling Report

## Purpose

This report records the first reproducible baseline comparison tooling for the
`v1_multilayer_bc_eq_physics_label_medium_v2` subset. The tooling currently
computes zero-delta baseline diagnostics across per-sample, split-wise, and
condition-wise groups.

This is baseline comparison tooling / zero-delta diagnostic summary only. It is
not a formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Script

Script:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py
```

Default subset:

```text
data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

Supported arguments:

- `--subset`: alternate subset path
- `--trained-predictions`: optional trained prediction path
- `--output-json`: optional JSON report path, intended for ignored `output/`

The trained prediction interface is intentionally explicit. Supported formats
are a `.npz` with arrays named by `sample_id`, or a directory containing
`<sample_id>.npy`, `<sample_id>/temperature.npy`, or
`<sample_id>/pred_temperature.npy` recovered-temperature predictions.

If trained predictions are not provided, the script computes only the
`zero_delta` baseline and prints `pending_no_trained_predictions`.

## Zero-Delta Definition

The zero-delta baseline predicts:

```text
DeltaT_pred = 0
T_pred = T_ref
```

`T_ref` is resolved from metadata using the existing non-leaking policy:

1. bottom Dirichlet fixed temperature
2. top Robin ambient temperature
3. fallback 300 K

## Metrics

Per sample, the script reports:

- recovered temperature MSE/RMSE/MAE
- raw DeltaT MSE/RMSE/MAE
- max absolute error
- p95 absolute error
- true and predicted peak temperature
- peak temperature error
- peak DeltaT error
- true and predicted hotspot index
- hotspot coordinate error
- top-k hotspot overlap
- split and condition metadata

Condition-wise summaries are grouped by:

- `source_pattern_tag`
- `k_region_mode`
- `k_field_mode`
- `stack_template`
- `bc_category`

## Command Result

Command:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py
```

Summary:

- per-sample rows: 64
- trained comparison status: `pending_no_trained_predictions`
- JSON written: false

Overall zero-delta summary:

| metric | value |
|---|---:|
| sample count | `64` |
| mean recovered T RMSE | `6.51538666e-02` |
| mean recovered T MAE | `3.82510808e-02` |
| mean DeltaT RMSE | `6.51538666e-02` |
| mean max abs error | `3.92192729e-01` |
| mean p95 abs error | `1.33909164e-01` |
| mean peak T error | `3.92192729e-01` |
| mean hotspot coordinate error | `7.37971023e-03` |

## Split-Wise Zero-Delta Summary

| split | n | mean T RMSE | mean T MAE | mean max abs | mean peak T error | mean hotspot distance |
|---|---:|---:|---:|---:|---:|---:|
| train | `48` | `6.55463677e-02` | `3.84876624e-02` | `3.99655061e-01` | `3.99655061e-01` | `7.33466603e-03` |
| valid | `8` | `6.01870855e-02` | `3.50358703e-02` | `3.60984203e-01` | `3.60984203e-01` | `7.61922496e-03` |
| test_id | `4` | `5.02665907e-02` | `2.90849079e-02` | `3.10464144e-01` | `3.10464144e-01` | `7.63574735e-03` |
| test_ood_bc_candidate | `2` | `1.18932494e-01` | `6.78608298e-02` | `6.81300273e-01` | `6.81300273e-01` | `7.75627370e-03` |
| test_ood_stack_candidate | `2` | `5.15968897e-02` | `3.41565616e-02` | `2.12280493e-01` | `2.12280493e-01` | `6.61407431e-03` |

The `test_ood_*` rows remain diagnostic candidates only and do not support OOD
claims.

## Condition-Wise Coverage

The script generated condition-wise zero-delta summaries for:

- heat-source pattern groups:
  `broad_block_power`, `centered_single_hotspot`,
  `dual_active_layer_hotspots`, `edge_or_corner_hotspot`,
  `multi_block_power`, `shifted_single_hotspot`, `two_hotspots_same_layer`
- k-region groups:
  `layerwise_isotropic_k`, `blockwise_isotropic_k`,
  `interposer_equivalent_k`, `diagonal_anisotropic_k`
- k-field modes: `iso1`, `diag3`
- stack templates:
  `baseline_4_layer`, `compact_3_layer`, `dual_active_4_layer`,
  `interposer_like_4_layer`, `held_out_interposer_like_candidate`
- BC categories:
  `nominal_top_h`, `low_top_h`, `high_top_h`,
  `held_out_top_h_candidate`

This confirms that the baseline tooling can summarize the same condition
metadata used by the medium generation manifest.

## Trained Comparison Status

Detailed trained-model comparison is pending. The previous 30-epoch controlled
training smoke did not save per-sample predictions or a checkpoint, so the
script cannot reconstruct condition-wise trained metrics without rerunning the
trained predictor under a fixed comparison protocol.

The script deliberately does not fabricate trained metrics from recorded scalar
summaries.

## Next Step

For the next controlled run, write predictions or metrics to ignored `output/`
using a fixed format. Then run:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --trained-predictions output/<ignored-trained-predictions-path>
```

The recommended next implementation step is to add an in-memory trained
prediction export path to the existing validation metrics smoke or to add a
dedicated comparison runner that trains, evaluates, aggregates condition-wise
metrics, and writes only ignored diagnostics artifacts.

## Non-Claims

This report does not claim:

- formal benchmark status
- model performance
- OOD generalization
- high-fidelity thermal labels
- industrial 3D IC simulation validity
