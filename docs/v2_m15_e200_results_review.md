# Heat3D v2 M1.5 B96 e200 Results Review

Scope: research-stage diagnostics on existing `medium1024_gapA_full1024_v2`
stratified runs. Post-hoc split-aware diagnostics only; no new training or data.

## Loss Comparison

| run | model/batch/epochs | lr/schedule | best_epoch | best_valid_iid_loss | final_valid_iid_loss | final/best | final_valid_stress_loss | final_valid_bg_bias | final_valid_hotspot_mae |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M1 e200 | M1/B192/e200 | 3e-4/constant | 200 | 0.3020 | 0.3020 | 1.000 | 0.4554 | 0.01351 | 0.03931 |
| M1 e200 replay | M1/B192/e200 | 3e-4/constant | 200 | 0.3047 | 0.3047 | 1.000 | 0.4535 | 0.01280 | 0.04140 |
| M1.5 e100 | M1.5/B96/e100 | 3e-4/constant | 100 | 0.3160 | 0.3160 | 1.000 | 0.4812 | 0.01210 | 0.04217 |
| M1.5 e200 3e-4 | M1.5/B96/e200 | 3e-4/constant | 196 | 0.2658 | 0.2972 | 1.118 | 0.4407 | 0.00834 | 0.04376 |
| M1.5 e200 1e-4 | M1.5/B96/e200 | 1e-4/constant | 182 | 0.3003 | 0.3148 | 1.048 | 0.4234 | 0.01186 | 0.04587 |
| M1.5 e200 warmup | M1.5/B96/e200 | 3e-4/warmup_cosine | 199 | 0.2739 | 0.2740 | 1.000 | 0.3969 | 0.00986 | 0.04113 |

## Field Diagnostics, Best Checkpoint

### valid_iid

| run | raw_deltaT_mse | field_variance_ratio | spatial_corr | amplitude_ratio | p95_abs_error | p99_abs_error | peak_abs_error | top_k_overlap | hotspot_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 e200 | 0.02148 | 1.866 | 0.8961 | 1.030 | 0.04104 | 0.07390 | 0.06012 | 0.7096 | 0.04686 |
| M1 e200 replay | 0.02148 | 1.576 | 0.8906 | 1.024 | 0.04118 | 0.07322 | 0.05688 | 0.7327 | 0.04686 |
| M1.5 e100 | 0.02170 | 2.399 | 0.9008 | 1.117 | 0.04282 | 0.07501 | 0.05902 | 0.7423 | 0.04894 |
| M1.5 e200 3e-4 | 0.01939 | 1.870 | 0.9189 | 1.097 | 0.03971 | 0.07131 | 0.05296 | 0.7481 | 0.04561 |
| M1.5 e200 1e-4 | 0.02136 | 1.923 | 0.9028 | 1.099 | 0.04099 | 0.07409 | 0.05693 | 0.7404 | 0.04552 |
| M1.5 e200 warmup | 0.01996 | 1.863 | 0.9108 | 1.099 | 0.04005 | 0.07044 | 0.05505 | 0.7481 | 0.04443 |

### valid_stress

| run | raw_deltaT_mse | field_variance_ratio | spatial_corr | amplitude_ratio | p95_abs_error | p99_abs_error | peak_abs_error | top_k_overlap | hotspot_mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 e200 | 0.02760 | 5.727 | 0.8541 | 1.339 | 0.05557 | 0.09482 | 0.09473 | 0.6364 | 0.06428 |
| M1 e200 replay | 0.02751 | 4.918 | 0.8457 | 1.306 | 0.05459 | 0.09666 | 0.09210 | 0.6523 | 0.06547 |
| M1.5 e100 | 0.02854 | 6.612 | 0.8714 | 1.485 | 0.05663 | 0.10390 | 0.10750 | 0.6864 | 0.07237 |
| M1.5 e200 3e-4 | 0.02555 | 5.254 | 0.8862 | 1.460 | 0.05174 | 0.09416 | 0.09785 | 0.6841 | 0.06623 |
| M1.5 e200 1e-4 | 0.02709 | 5.349 | 0.8728 | 1.468 | 0.05396 | 0.09832 | 0.10050 | 0.6795 | 0.06713 |
| M1.5 e200 warmup | 0.02576 | 5.252 | 0.8817 | 1.487 | 0.05228 | 0.09259 | 0.10200 | 0.6864 | 0.06463 |

## Low-DeltaT / Bin0 Diagnostics, Best Checkpoint

### valid_iid

| run | le_0p05_mae | le_0p05_bias | le_0p05_over | bin0_mae | bin0_bias | bin0_over |
|---|---:|---:|---:|---:|---:|---:|
| M1 e200 | 0.01237 | 0.00898 | 0.8057 | 0.01351 | 0.01351 | 1.000 |
| M1 e200 replay | 0.01187 | 0.00816 | 0.7939 | 0.01280 | 0.01280 | 1.000 |
| M1.5 e100 | 0.01153 | 0.00749 | 0.7794 | 0.01213 | 0.01213 | 0.9995 |
| M1.5 e200 3e-4 | 0.00877 | 0.00391 | 0.7223 | 0.00751 | 0.00733 | 0.9403 |
| M1.5 e200 1e-4 | 0.01194 | 0.00844 | 0.8007 | 0.01322 | 0.01322 | 1.000 |
| M1.5 e200 warmup | 0.00997 | 0.00556 | 0.7549 | 0.00992 | 0.00990 | 0.9901 |

### valid_stress

| run | le_0p05_mae | le_0p05_bias | le_0p05_over | bin0_mae | bin0_bias | bin0_over |
|---|---:|---:|---:|---:|---:|---:|
| M1 e200 | 0.01569 | 0.01121 | 0.7720 | 0.01731 | 0.01731 | 0.9973 |
| M1 e200 replay | 0.01478 | 0.00946 | 0.7453 | 0.01601 | 0.01600 | 0.9934 |
| M1.5 e100 | 0.01460 | 0.01014 | 0.7655 | 0.01569 | 0.01567 | 0.9844 |
| M1.5 e200 3e-4 | 0.01151 | 0.00598 | 0.6973 | 0.01069 | 0.01021 | 0.8987 |
| M1.5 e200 1e-4 | 0.01466 | 0.01071 | 0.7895 | 0.01589 | 0.01589 | 0.9999 |
| M1.5 e200 warmup | 0.01253 | 0.00749 | 0.7371 | 0.01255 | 0.01244 | 0.9591 |

## Judgment

- M1.5 B96 e200 warmup_cosine exceeds the M1 B192 e200 baseline on primary
  valid_iid loss and on valid_stress loss. It also improves valid_iid/stress
  raw DeltaT MSE, spatial correlation, top-k overlap, and most low-DeltaT
  diagnostics.
- It does improve both valid_iid and valid_stress, but not all field-shape
  metrics are clean wins: stress amplitude remains high.
- Compared with old M1.5 e100, warmup e200 reduces field_variance_ratio on both
  splits and reduces valid_iid amplitude ratio. Stress amplitude is roughly
  unchanged/slightly worse: 1.487 vs 1.485.
- Bin0 and low-DeltaT overprediction are mitigated but not solved. The e200
  3e-4 best checkpoint gives the strongest bin0 improvement, but its final/best
  ratio is 1.118, so it needs best-checkpoint discipline.
- Current practical candidate: M1.5 B96 e200 warmup_cosine because it is stable
  final-vs-best and improves both splits. Current best-checkpoint candidate:
  M1.5 B96 e200 3e-4, with late degradation risk.

## Next Training Order

1. Run M1.25 B96 `lr=3e-4` and M1.25 B128 `lr=3e-4` to test whether an
   intermediate capacity keeps shape/top-k gains with less amplitude overshoot.
2. Run M1.5 B96 `clip0p5` and `wd1e3` to test regularization against stress
   amplitude overshoot.
3. Run light background over/bias only after the M1.25 and regularization
   results clarify whether bin0 can improve without harming field shape.
4. Do not prioritize M1.5 lower LR 1e-4 or very-low LR 3e-5 next; current 1e-4
   is not competitive enough.
