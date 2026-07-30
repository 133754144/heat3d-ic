# V6-P1i pilot128 distribution audit

Status: **failed**.

No training or model inference was run. The 1024-sample expansion remains blocked pending an explicit decision after this report.

## Temperature coverage

- Primary 30--150 K: 108/128 (84.38%).
- Outer-safety violations: 4.
- Twelve-bin counts: `[20, 25, 11, 16, 8, 11, 10, 2, 2, 1, 1, 1]`.
- Empty bins: 0; nonzero max/min ratio: 25.000.
- Largest adjacent sorted peak gap: 10.736 K.
- Significant KDE modes: 1 at [40.891].

## DeltaT summaries

| metric | mean | std | q05 | q50 | q95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| deltaT_max_K | 56.8888 | 26.7541 | 22.0688 | 49.5340 | 102.1918 | 15.0699 | 142.3986 |
| deltaT_mean_K | 43.7731 | 22.9027 | 15.8110 | 36.5297 | 83.2081 | 10.3091 | 121.3436 |
| deltaT_cv_rms_K | 44.1200 | 23.0763 | 15.8512 | 36.9901 | 84.6628 | 10.4745 | 121.9266 |

## Gate details

- `primary_fraction`: FAIL
- `outer_safety`: FAIL
- `empty_bins`: PASS
- `bin_ratio`: FAIL
- `sorted_gap`: PASS
- `kde_not_four`: PASS
- `kde_mode_limit`: PASS
- `energy_balance`: PASS
- `linear_residual`: PASS
- `support_coverage`: PASS
- `finite`: PASS

The distribution was not binned or replaced after solving. Correlations are descriptive only and cannot be used to remove samples.
