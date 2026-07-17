# Gate 6J valid-only causal diagnostic

Scope: existing V13 e318 and V32 e474 checkpoints, train-only normalization/context reconstruction, and `valid_iid` evaluation. No training, checkpoint/model mutation, checkpoint selection, or test/hard/sealed access occurred.

## Frozen V5 metrics

| model | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |
|---|---:|---:|---:|---:|---:|
| V13 e318 | 23.700678 | 20.316459 | 0.167982 | 0.150284 | 0.165381 |
| V32 e474 | 22.408387 | 21.034804 | 0.160067 | 0.145697 | 0.197482 |

## Paired bootstrap

All differences are V32 minus V13; negative is better.

| metric | observed Δ | 95% CI | win rate | median per-sample Δ |
|---|---:|---:|---:|---:|
| point_global_relative_rmse_pct | -1.292291 | [-3.451426, 0.814868] | 0.5547 | -0.250998 |
| sample_first_cv_relative_rmse_pct | 0.718345 | [-0.558399, 2.199583] | 0.5469 | -0.410647 |
| raw_cv_weighted_rmse_K | -0.007915 | [-0.024100, 0.006760] | 0.5469 | -0.000865 |
| shape_cv_rmse | -0.004587 | [-0.009299, 0.000196] | 0.5859 | -0.004340 |
| scale_log_rmse | 0.032101 | [-0.001094, 0.071599] | 0.4688 | 0.003320 |

The point-global gain is tail-sensitive: its CI crosses zero. Sample-first has a positive mean difference despite a negative median and a win rate above 50%, showing that a minority of regressions dominates the unweighted sample mean.

## Stratified result

Quartile axes are fixed from the 128 valid samples. The condition category is the pre-solve generator metadata `q_block_metadata.DeltaT_target_bin`; it is not derived from solved temperature labels.

### true_cv_rms_deltaT_K

| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |
|---|---:|---:|---:|---:|
| Q1 | 32 | 0.275753 | 0.6562 | 0.257611 |
| Q2 | 32 | 3.675671 | 0.4375 | 77.324689 |
| Q3 | 32 | 0.090499 | 0.5312 | -7.414112 |
| Q4 | 32 | -1.168544 | 0.5625 | -392.735235 |

### total_power_W

| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |
|---|---:|---:|---:|---:|
| Q1 | 32 | 1.536635 | 0.5625 | 16.473307 |
| Q2 | 32 | -0.061155 | 0.5312 | -73.875941 |
| Q3 | 32 | -0.138387 | 0.6875 | -408.023041 |
| Q4 | 32 | 1.536287 | 0.4062 | 142.858628 |

### source_occupancy_fraction

| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |
|---|---:|---:|---:|---:|
| Q1 | 32 | 2.669312 | 0.5625 | -56.199456 |
| Q2 | 32 | -0.261687 | 0.5625 | -316.446580 |
| Q3 | 32 | 0.679599 | 0.5000 | -44.953206 |
| Q4 | 32 | -0.213844 | 0.5625 | 95.032196 |

### q_weighted_inverse_kz_mK_W

| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |
|---|---:|---:|---:|---:|
| Q1 | 32 | 1.892863 | 0.5312 | 12.023794 |
| Q2 | 32 | -0.377192 | 0.5000 | -245.750257 |
| Q3 | 32 | 2.546100 | 0.5938 | 82.301803 |
| Q4 | 32 | -1.188392 | 0.5625 | -171.142387 |

### generator_condition_category

| bin | n | sample-first Δ pp | win rate | point SSE Δ K² |
|---|---:|---:|---:|---:|
| low | 23 | -0.673737 | 0.6087 | 6.813925 |
| low_to_nominal | 24 | -1.882192 | 0.7917 | -384.643012 |
| nominal_to_hard | 81 | 1.884157 | 0.4568 | 55.262041 |

## Attention residual

Residual/mean-pool cosine: mean `0.131536`, median `0.116231`. Norm ratio: mean `0.755709`, median `0.739505`.

Residual cosine/norm ratio has weak correlation with the V32−V13 shape, scale, and relative-error changes, while cosine is strongly correlated with true ΔT and q-weighted inverse-kz. The path is physics-responsive, but residual magnitude alone does not explain which samples regress.

## Inference-only α sweep

| α | point-global % | sample-first % | raw CV RMSE K | Q1 sample-first Δ pp |
|---:|---:|---:|---:|---:|
| 0.00 | 45.661461 | 32.422858 | 0.335536 | 8.465128 |
| 0.25 | 40.058294 | 28.803808 | 0.293554 | 5.476690 |
| 0.50 | 33.631269 | 25.167694 | 0.244995 | 2.826473 |
| 0.75 | 26.925166 | 22.069643 | 0.193955 | 0.923332 |
| 1.00 | 22.408410 | 21.034925 | 0.160067 | 0.275462 |

Optimal discrete sensitivity interval: `[1.00, 1.00]`. Only α=1 preserves the point-global gain; no α restores sample-first, and low-ΔT Q1 regresses for every α.

## Decision

Unique recommendation: **objective_alignment**.

Reason: alpha scaling does not restore the paired low-DeltaT Q1 sample-first error.
