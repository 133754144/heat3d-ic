# V4P3_19 Relative Error Decomposition

Read this file only for V4P3 relative-error source analysis or selection-metric
decisions.

## Scope

- Target run: `V4P3_19`.
- Checkpoints: `best` and `final`.
- Splits: `valid_iid` and `test_iid`.
- This is read-only diagnostics on existing predictions and labels. No
  training, tmux launch, or artifact sync was started.
- The decomposition uses runner-compatible raw DeltaT error:
  `100 * raw_deltaT_RMSE_K / mean_abs_true_deltaT_K`.
- `P02_like_proxy=True` means the sample `qc_physical_keep_reasons` contains
  `low_k_trapped_hotspot` or `multi_source_or_high_power_bottleneck`. It is a
  candidate1024 proxy for disconnected-path or thermal-bottleneck behavior, not
  an exact final-probe P02 label.

## Metric Recheck

| entry | raw DeltaT RMSE K | mean abs true DeltaT K | rel RMSE % | loss_summary rel % |
| --- | ---: | ---: | ---: | ---: |
| best valid_iid | 1.483 | 0.412 | 359.94 | 359.94 |
| final valid_iid | 1.509 | 0.412 | 366.33 | 366.32 |
| best test_iid | 0.810 | 0.468 | 173.21 | n/a |
| final test_iid | 0.805 | 0.468 | 172.13 | n/a |

The `rel_rmse_v4_pct` around 360% is therefore reproducible from existing
`V4P3_19` valid_iid predictions. The valid split contains a very high-DeltaT
tail: max true DeltaT is `277 K`, while median true DeltaT is only `0.053 K`.

## Per-Bin Relative RMSE

Primary entry: `final_valid_iid`.

| true DeltaT bin K | point frac | MSE contribution | RMSE K | local rel RMSE % | over ratio | bias K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <=0.05 | 49.4% | 0.05% | 0.047 | 554.1 | 0.652 | 0.015 |
| 0.05-0.10 | 9.1% | 0.02% | 0.069 | 95.2 | 0.445 | 0.008 |
| 0.10-0.25 | 15.0% | 0.1% | 0.110 | 67.4 | 0.360 | -0.009 |
| 0.25-0.50 | 10.5% | 0.1% | 0.179 | 50.1 | 0.331 | -0.035 |
| 0.50-1.00 | 7.7% | 0.3% | 0.316 | 44.5 | 0.332 | -0.057 |
| 1.00-2.00 | 4.9% | 0.8% | 0.593 | 42.9 | 0.345 | -0.104 |
| >2.00 | 3.4% | 98.6% | 8.120 | 127.7 | 0.296 | -1.456 |

The low-DeltaT bins have large local relative RMSE because their denominators
are tiny, but they do not drive the global MSE numerator. The `>2 K` bin is only
3.4% of points and contributes 98.6% of global squared error.

## Low-DeltaT And Overprediction

| entry | region | point frac | MSE contribution | RMSE K | local rel RMSE % | over ratio | bias K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best valid_iid | DeltaT <= 0.05 | 49.4% | 0.11% | 0.069 | 806.7 | 0.620 | 0.020 |
| best valid_iid | low-DeltaT background | 48.9% | 0.11% | 0.069 | 836.2 | 0.617 | 0.020 |
| final valid_iid | DeltaT <= 0.05 | 49.4% | 0.05% | 0.047 | 554.1 | 0.652 | 0.015 |
| final valid_iid | low-DeltaT background | 48.9% | 0.05% | 0.047 | 573.9 | 0.649 | 0.015 |
| final test_iid | DeltaT <= 0.05 | 49.6% | 0.17% | 0.047 | 508.5 | 0.659 | 0.014 |
| final test_iid | low-DeltaT background | 48.2% | 0.16% | 0.047 | 540.1 | 0.650 | 0.014 |

`le0.05` overprediction is real: around 62-66% of points in this bin are
overpredicted. But its direct squared-error contribution is below 0.2% on both
valid and test. It is a calibration problem, not the main source of the 360%
global relative RMSE.

## Strong-Q And Hotspot Contribution

Primary entry: `final_valid_iid`.

| region | point frac | MSE contribution | RMSE K | local rel RMSE % | over ratio | bias K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 DeltaT | 10.1% | 96.4% | 4.670 | 206.3 | 0.318 | -0.469 |
| top5 DeltaT | 5.1% | 91.5% | 6.406 | 198.5 | 0.303 | -0.723 |
| q_positive | 11.3% | 85.5% | 4.148 | 265.2 | 0.388 | -0.280 |
| strong_q | 6.0% | 55.8% | 4.603 | 406.6 | 0.389 | -0.182 |

The dominant error is hotspot/strong-q underprediction in high-DeltaT regions,
not background overprediction. Top-DeltaT and q-positive masks overlap strongly
with the P02-like samples below.

## P02-Like And Structural Contribution

Primary entry: `final_valid_iid`.

| group | samples | point frac | MSE contribution | RMSE K | median sample rel % |
| --- | ---: | ---: | ---: | ---: | ---: |
| P02_like_proxy=True | 12 | 9.4% | 98.4% | 4.889 | 97.3 |
| P02_like_proxy=False | 116 | 90.6% | 1.6% | 0.198 | 67.7 |
| qc_class=physical_hard_keep | 12 | 9.4% | 98.4% | 4.889 | 97.3 |
| qc_class=review_hold | 13 | 10.2% | 0.8% | 0.431 | 92.2 |
| qc_class=clean_keep | 103 | 80.5% | 0.7% | 0.145 | 63.6 |

On `final_test_iid`, the same pattern holds but is less extreme:
`P02_like_proxy=True` has 12 samples and contributes 87.0% of MSE.

## Top Bad Samples

Primary entry: `final_valid_iid`, ranked by MSE contribution.

| sample | MSE contribution | sample rel % | RMSE K | mean abs DeltaT K | peak DeltaT K | qc class | reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| sample_0174 | 78.5% | 414.1 | 15.128 | 3.653 | 277.000 | physical_hard_keep | multi_source_or_high_power_bottleneck |
| sample_0731 | 11.9% | 98.2 | 5.887 | 5.994 | 78.133 | physical_hard_keep | low_k_trapped_hotspot; multi_source_or_high_power_bottleneck |
| sample_0204 | 2.6% | 143.0 | 2.777 | 1.942 | 120.044 | physical_hard_keep | low_k_trapped_hotspot; weak_cooling |
| sample_0339 | 2.1% | 243.8 | 2.477 | 1.016 | 40.077 | physical_hard_keep | low_k_trapped_hotspot |
| sample_0274 | 1.1% | 243.3 | 1.782 | 0.732 | 47.812 | physical_hard_keep | low_k_trapped_hotspot |
| sample_0993 | 0.7% | 74.9 | 1.438 | 1.921 | 16.386 | physical_hard_keep | multi_source_or_high_power_bottleneck |

The top five samples contribute about 96% of valid_iid MSE. The single worst
sample, `sample_0174`, contributes 78.5% by itself.

Per-sample relative RMSE can be misleading when sorted alone. The highest
relative sample is `sample_0334` at 1323.5%, but its RMSE is only 0.073 K and
its MSE contribution is effectively zero because the true DeltaT denominator is
near zero.

## Answer

The `rel_rmse_v4_pct` around 360% is not mainly caused by low-DeltaT background
errors. Low-DeltaT background has high local relative error and persistent
overprediction, but contributes less than 0.2% of squared error.

The main source is a small number of P02-like structural samples with
high-DeltaT hotspot/strong-q regions. On `final_valid_iid`, 12 P02-like /
`physical_hard_keep` samples contribute 98.4% of MSE, top5 DeltaT points
contribute 91.5%, and strong-q points contribute 55.8%. The global percentage
looks especially large because that high-DeltaT error is divided by the split's
small average true DeltaT denominator (`0.412 K`), which is pulled down by many
near-background points.
