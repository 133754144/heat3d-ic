# heat3d_v6_p1i_continuous_physics1024_v0 distribution audit

Status: **failed**.

No training or model inference was run. This report audits the frozen 1024-sample artifact without filtering or replacement.

## Temperature coverage

- Primary 30--150 K: 973/1024 (95.02%).
- Outer-safety violations: 0.
- Twelve-bin counts: `[40, 87, 81, 90, 82, 90, 80, 87, 91, 84, 92, 69]`.
- Empty bins: 0; nonzero max/min ratio: 2.300.
- Largest adjacent sorted peak gap: 5.743 K.
- Significant KDE modes: 2 at [81.996, 123.64].

## DeltaT summaries

| metric | mean | std | q05 | q50 | q95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| deltaT_max_K | 95.3561 | 34.8536 | 41.1993 | 95.5809 | 149.8188 | 33.1395 | 174.4833 |
| deltaT_mean_K | 76.1064 | 28.7004 | 32.9260 | 75.0063 | 122.2212 | 25.5250 | 142.9152 |
| deltaT_cv_rms_K | 76.6749 | 28.8272 | 33.1477 | 75.8707 | 122.9359 | 25.9541 | 143.2098 |

## Gate details

- `primary_fraction`: PASS
- `outer_safety`: PASS
- `empty_bins`: PASS
- `bin_ratio`: PASS
- `sorted_gap`: FAIL
- `kde_not_four`: PASS
- `kde_mode_limit`: PASS
- `energy_balance`: PASS
- `linear_residual`: PASS
- `support_coverage`: PASS
- `finite`: PASS
- `split_pre_solve_parameter_ks`: FAIL
- `split_temperature_metric_ks`: PASS
- `split_peak_median`: PASS

The distribution was not binned or replaced after solving. Correlations are descriptive only and cannot be used to remove samples.
