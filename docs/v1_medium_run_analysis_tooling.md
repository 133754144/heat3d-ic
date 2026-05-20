# Heat3D v1 Medium Run Analysis Tooling

## Purpose

`scripts/analyze_heat3d_v1_medium_run_summary.py` summarizes an existing
Heat3D v1 medium-style training/export run. It reads the run's loss summary and
baseline comparison JSON, compares `trained_prediction` against `zero_delta`,
and writes structured JSON plus a Markdown report.

This is run analysis tooling only. It is diagnostic support, not a formal
benchmark, not a model-performance conclusion, not OOD generalization evidence,
and not high-fidelity solver validation.

## Inputs

Required:

```text
--run-dir
```

Defaults derived from `--run-dir`:

```text
<run-dir>/loss_summary.json
<run-dir>/baseline_comparison.json
<run-dir>/run_analysis.json
<run-dir>/run_analysis.md
```

Optional overrides:

```text
--loss-summary
--baseline-comparison
--output-json
--output-md
--metric-set
```

Default metric set:

```text
mean_T_rmse
mean_T_mae
mean_DeltaT_rmse
mean_max_abs
mean_p95_abs
mean_peak_T_err
mean_hotspot_dist
```

The comparison JSON stores these using the actual schema fields from
`compare_heat3d_v1_medium_baselines.py`, for example
`mean_recovered_T_rmse`, `mean_recovered_T_mae`,
`mean_peak_T_error`, and `mean_hotspot_coord_error`. The analysis script maps
the report-facing metric names to those JSON fields.

## Outputs

`run_analysis.json` keeps structured fields for later multi-run aggregation:

- loss trend
- overall trained vs zero_delta changes
- split-wise changes
- condition-wise changes
- top improved condition groups
- top degraded condition groups
- `likely_hotspot_learning_with_background_bias`

`run_analysis.md` provides a human-readable report with:

- Run summary
- Loss trend
- Overall trained vs zero_delta table
- Overall relative changes
- Split-wise summary
- Top improved condition groups
- Top degraded condition groups
- Interpretation of mean-field vs hotspot conflicts
- Recommended next experiments

## Relative Change

For each error metric:

```text
relative_change = (trained_prediction - zero_delta) / abs(zero_delta)
```

Negative means the trained prediction has a lower error than zero_delta for
that metric. Positive means the trained prediction has a higher error than
zero_delta. Hotspot distance is treated as an error metric, so smaller is
better.

If overall `mean_T_rmse` and `mean_T_mae` degrade while `mean_peak_T_err` and
`mean_hotspot_dist` improve, the analysis marks:

```text
likely_hotspot_learning_with_background_bias = true
```

This is an interpretation aid only. It does not establish model performance.

## SSH Git-Only Usage

Analyze the 50-epoch medium256 run:

```bash
cd ~/myCodeGitOnly/heat3d-ic
python scripts/analyze_heat3d_v1_medium_run_summary.py \
  --run-dir output/heat3d_v1_medium_runs/medium256_e050_seed0
```

Analyze the 200-epoch medium256 run:

```bash
cd ~/myCodeGitOnly/heat3d-ic
python scripts/analyze_heat3d_v1_medium_run_summary.py \
  --run-dir output/heat3d_v1_medium_runs/medium256_e200_seed0
```

The generated analysis files remain under ignored `output/` unless an explicit
artifact-publishing policy is approved.

## Recommended Follow-Up Diagnostics

- error-binning / background-bias analysis
- optional e100/e200 comparison
- weighted loss / background penalty / residual learning
