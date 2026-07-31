# V6-P1i formal1024_v0 zero-solve postmortem

This analysis reuses the frozen formal1024_v0 outputs and performs no new PDE solve, training, or model inference. Split candidates are scored only with reconstructed pre-solve inputs.

## Temperature gap

- Largest gap: 5.743380 K, from `v6p1if_0923` to `v6p1if_0650`.
- Empirical location: q=0.998047; classification: `extreme_upper_tail`.
- KDE density at the midpoint is 12.058% of the global maximum. The failed gap is therefore a sparse extreme-tail interval, not a core/modal-valley discontinuity.

## Proxy and response audit

- Dominant single latents at |Spearman| >= 0.9: `['continuous_severity', 'mean_source_area_fraction']`.
- Severity/source-size Spearman coupling: 0.945382.
- `Reff = peak DeltaT / package power`. Partial correlations regress both response and each transformed feature on log(power); they are descriptive diagnostics, not a power backsolve.

## Target-independent split comparison

| dataset | method | max continuous KS | max discrete TV | max joint | score |
|---|---|---:|---:|---:|---:|
| 1024 | balanced_pre_solve_assignment | 0.140625 | 0.117188 | 0.035042 | 0.292855 |
| 1024 | independent_sobol_dimension | 0.132812 | 0.156250 | 0.021430 | 0.310492 |
| 1024 | global_hash | 0.195312 | 0.148438 | 0.052466 | 0.396216 |
| 1024 | octet_hash | 0.177083 | 0.187500 | 0.051422 | 0.416006 |
| 128 | balanced_pre_solve_assignment | 0.375000 | 0.322917 | 0.114229 | 0.812145 |
| 128 | global_hash | 0.375000 | 0.562500 | 0.110976 | 1.048476 |
| 128 | octet_hash | 0.562500 | 0.375000 | 0.136660 | 1.074160 |
| 128 | independent_sobol_dimension | 0.666667 | 0.666667 | 0.134557 | 1.467890 |

Selected method for preregistration: `balanced_pre_solve_assignment`.

formal1024_v0 remains permanently qualification-failed. No sample, split, threshold, or frozen artifact was repaired.
