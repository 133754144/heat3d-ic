# Heat3D v1 Medium Condition Bias Diagnostics

This note documents diagnostic tooling for the current `medium1024_gapA`
training stage. It is not a formal benchmark, not an OOD generalization claim,
and not a model-performance conclusion.

## Motivation

`medium1024_gapA_full1024_v2` has passed generation checks, label diagnostics,
diversity diagnostics, and short e50/e100 controlled training probes. The main
remaining diagnostic bottleneck is persistent low-DeltaT `bin_0` background
overprediction. The next useful step is not longer e200/e300 training, but more
structured analysis of where the background bias appears.

## Run Summary Inputs

`scripts/analyze_heat3d_v1_medium_run_summary.py` now supports explicit
comparison and error-bin inputs:

```bash
python scripts/analyze_heat3d_v1_medium_run_summary.py \
  --run-dir output/heat3d_v1_medium_runs/<run> \
  --baseline-comparison-json output/heat3d_v1_medium_runs/<run>/baseline_comparison_final.json \
  --error-bins-json output/heat3d_v1_medium_runs/<run>/error_bins_final.json \
  --prediction-label final \
  --output-json output/heat3d_v1_medium_runs/<run>/run_analysis_final.json \
  --output-md output/heat3d_v1_medium_runs/<run>/run_analysis_final.md
```

The same interface can be used for `best_predictions.npz` by passing the
corresponding best comparison/error-bin JSON and `--prediction-label best`.

## Condition-Wise Diagnostics

`scripts/analyze_heat3d_v1_medium_condition_diagnostics.py` recomputes
prediction errors from a subset and a recovered-temperature prediction archive.
It groups results by:

- split
- source category (`source_pattern_tag`)
- k region mode
- BC category
- k mode (`k_field_mode`)
- q power range

For each group it reports overall RMSE, MAE, signed bias, overprediction ratio,
and underprediction ratio. It also reports global DeltaT-bin metrics for
`bin_0` through `bin_4`, including relative RMSE/MAE change, signed bias,
overprediction ratio, and underprediction ratio.

The q-power range is a diagnostic quantile grouping based on per-sample
integrated source power. It is used to find whether low-power cases contribute
disproportionately to the low-DeltaT background bias.

## Final vs Best Predictions

`predictions.npz` remains the final-epoch prediction archive. When the training
runner was invoked with `--save-best-predictions`, `best_predictions.npz` is an
additional archive selected by the configured validation metric. The analysis
tools do not assume either archive is better; they only label and summarize the
chosen input.

Recommended diagnostic pattern:

1. Run comparison, error bins, run summary, and condition diagnostics for final
   predictions.
2. Run the same tools for best predictions.
3. Compare `bin_0` signed bias and overprediction ratio before considering
   longer training.

Example final/best analysis layout:

```bash
python scripts/compare_heat3d_v1_medium_baselines.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 \
  --trained-predictions output/heat3d_v1_medium_runs/e100_twostage_best_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/baseline_comparison_final.json \
  --stdout-mode compact

python scripts/analyze_heat3d_v1_medium_error_bins.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 \
  --trained-predictions output/heat3d_v1_medium_runs/e100_twostage_best_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/error_bins_final.json \
  --output-md output/heat3d_v1_medium_runs/e100_twostage_best_seed0/error_bins_final.md \
  --stdout-mode compact

python scripts/analyze_heat3d_v1_medium_run_summary.py \
  --run-dir output/heat3d_v1_medium_runs/e100_twostage_best_seed0 \
  --baseline-comparison-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/baseline_comparison_final.json \
  --error-bins-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/error_bins_final.json \
  --prediction-label final \
  --output-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/run_analysis_final.json \
  --output-md output/heat3d_v1_medium_runs/e100_twostage_best_seed0/run_analysis_final.md \
  --stdout-mode compact

python scripts/analyze_heat3d_v1_medium_condition_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2 \
  --trained-predictions output/heat3d_v1_medium_runs/e100_twostage_best_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/e100_twostage_best_seed0/condition_diagnostics_final.json \
  --output-md output/heat3d_v1_medium_runs/e100_twostage_best_seed0/condition_diagnostics_final.md \
  --prediction-label final \
  --stdout-mode compact
```

Repeat the same commands with `best_predictions.npz` and filenames ending in
`_best` when the run contains a best-valid prediction archive.

## Reporting Boundary

These outputs should be described as condition-wise background-bias diagnostics
for a controlled research-stage training run. They should not be used to claim a
formal benchmark result, publication-ready performance, or solved OOD
generalization.
