# Heat3D v1 Medium Error-Binning Diagnostics Tooling

## Purpose

`scripts/analyze_heat3d_v1_medium_error_bins.py` diagnoses where a trained
Heat3D v1 medium-style run differs from the zero-delta baseline. The tool is
intended for cases where mean RMSE/MAE can look worse than zero_delta while
peak and hotspot diagnostics look better.

This is error-binning / background-bias diagnostics tooling only. It is not a
formal benchmark, not a model-performance conclusion, not OOD generalization
evidence, and not high-fidelity solver validation.

## Inputs

Required:

```text
--subset
--trained-predictions
```

Optional:

```text
--output-json
--output-md
--bins
--group-by
```

`--trained-predictions` is normally the `predictions.npz` written by
`scripts/run_heat3d_v1_medium_controlled_training_export.py`. The NPZ stores
recovered-temperature arrays keyed by `sample_id`, matching the loader contract
used by `scripts/compare_heat3d_v1_medium_baselines.py`.

## Outputs

Default output names are:

```text
error_bins.json
error_bins.md
```

The JSON keeps structured fields for:

- global DeltaT bin edges
- overall bin metrics
- split-wise bins
- condition-wise bins
- background-bias interpretation flags
- recommended next actions

The Markdown report includes:

- Run inputs
- Global DeltaT bin edges
- Overall bin table
- Split-wise bin summary
- Condition-wise key findings
- Background-bias interpretation
- Recommended next actions

## DeltaT Bins

Bins are defined from global true `DeltaT = T_true - T_ref` percentiles across
all generated points in the subset.

Default:

```text
p50,p75,p90,p95
```

This creates:

```text
bin_0: [min, p50]
bin_1: (p50, p75]
bin_2: (p75, p90]
bin_3: (p90, p95]
bin_4: (p95, max]
```

The same global edges are reused for overall, split-wise, and condition-wise
summaries.

## Metrics

For each bin the tool reports:

- point count and sample count
- DeltaT min / max / mean
- zero_delta RMSE / MAE
- trained RMSE / MAE
- trained signed bias: `mean(T_pred_trained - T_true)`
- zero signed bias
- trained overprediction ratio: `mean(error_trained > 0)`
- trained underprediction ratio: `mean(error_trained < 0)`
- relative RMSE / MAE change

Relative change is:

```text
(trained_error_metric - zero_delta_error_metric) / abs(zero_delta_error_metric)
```

Negative means trained is lower-error than zero_delta for that metric. Positive
means trained is higher-error. Hotspot and background interpretations remain
diagnostic only.

## Background-Bias Interpretation

The tool marks:

```text
likely_background_overprediction = true
```

when low-DeltaT bins have worse trained RMSE/MAE than zero_delta and positive
trained signed bias.

It marks:

```text
likely_hotspot_region_improvement = true
```

when high-DeltaT bins improve against zero_delta.

If both are true, it marks:

```text
likely_hotspot_learning_with_background_bias = true
```

This is a diagnosis of error distribution, not a model-performance conclusion.

## SSH Git-Only Usage

Run true medium256_e200 error-binning analysis on the server:

```bash
cd ~/myCodeGitOnly/heat3d-ic
conda activate rigno
git fetch origin
git switch research/v1-medium256-dataset
git pull --ff-only

python scripts/analyze_heat3d_v1_medium_error_bins.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2 \
  --trained-predictions output/heat3d_v1_medium_runs/medium256_e200_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/medium256_e200_seed0/error_bins.json \
  --output-md output/heat3d_v1_medium_runs/medium256_e200_seed0/error_bins.md
```

Inspect the report:

```bash
sed -n '1,220p' output/heat3d_v1_medium_runs/medium256_e200_seed0/error_bins.md
```

Generated analysis files remain under ignored `output/`.

## Recommended Next Actions

- background penalty
- hotspot + background combined loss
- conservative / residual learning
- optional e100/e200 comparison
