# heat3d_v6_p1i_continuous_physics128_v2 distribution audit

Status: **passed**.

No training or model inference was run. The 1024-sample expansion remains blocked pending an explicit decision after this report.

## Temperature coverage

- Primary 30--150 K: 124/128 (96.88%).
- Outer-safety violations: 0.
- Twelve-bin counts: `[5, 8, 12, 11, 13, 10, 9, 10, 13, 11, 12, 10]`.
- Empty bins: 0; nonzero max/min ratio: 2.600.
- Largest adjacent sorted peak gap: 8.673 K.
- Significant KDE modes: 2 at [69.159, 116.751].

## DeltaT summaries

| metric | mean | std | q05 | q50 | q95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| deltaT_max_K | 94.8161 | 34.2296 | 41.9305 | 93.6721 | 146.4225 | 35.3892 | 166.7650 |
| deltaT_mean_K | 75.6812 | 27.5765 | 33.7110 | 74.2691 | 118.7924 | 26.5208 | 132.7566 |
| deltaT_cv_rms_K | 76.2821 | 27.7931 | 34.2169 | 75.1303 | 119.3405 | 26.5948 | 133.0485 |

## Gate details

- `primary_fraction`: PASS
- `outer_safety`: PASS
- `empty_bins`: PASS
- `bin_ratio`: PASS
- `sorted_gap`: PASS
- `kde_not_four`: PASS
- `kde_mode_limit`: PASS
- `energy_balance`: PASS
- `linear_residual`: PASS
- `support_coverage`: PASS
- `finite`: PASS

The distribution was not binned or replaced after solving. Correlations are descriptive only and cannot be used to remove samples.
