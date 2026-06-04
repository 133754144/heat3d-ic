# Heat3D v2 e200 and Memory-Safe M1.5 Results

This is a research-stage diagnostic review on the existing
`medium1024_gapA_full1024_v2` stratified split. No new data or labels were
generated.

## Runs

| run | model | batch | epochs | updates | best_epoch | best_valid_iid | final_valid_iid | final/best | final_valid_stress | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m1_B192_base_mse_lr3e4_e100_stratified_seed0 | M1 | 192 | 100 | 400 | 100 | 0.4154 | 0.4154 | 1.000 | 0.5930 | 1214.9s |
| m1_B192_base_mse_lr3e4_e200_stratified_seed0 | M1 | 192 | 200 | 800 | 200 | 0.3020 | 0.3020 | 1.000 | 0.4554 | 2167.5s |
| m15_B96_base_mse_lr3e4_e100_stratified_seed0 | M1.5 | 96 | 100 | 800 | 100 | 0.3160 | 0.3160 | 1.000 | 0.4812 | 2552.5s |

## Field Diagnostics

Final and best are identical for these runs because best_epoch is the last
epoch. Diagnostics are post-hoc split-aware metrics on existing predictions.

| split | run | raw_deltaT_mse | field_variance_ratio | spatial_corr | amplitude_ratio | p95_abs_error | p99_abs_error | peak_abs_error | top_k_overlap | hotspot_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_iid | M1 e100 | 0.02592 | 1.325 | 0.797 | 1.002 | 0.04950 | 0.09352 | 0.06322 | 0.681 | 0.06248 |
| valid_iid | M1 e200 | 0.02148 | 1.866 | 0.896 | 1.030 | 0.04104 | 0.07390 | 0.06012 | 0.710 | 0.04686 |
| valid_iid | M1.5 B96 e100 | 0.02170 | 2.399 | 0.901 | 1.117 | 0.04282 | 0.07501 | 0.05902 | 0.742 | 0.04894 |
| valid_stress | M1 e100 | 0.03202 | 3.952 | 0.789 | 1.262 | 0.06276 | 0.10900 | 0.08892 | 0.648 | 0.07387 |
| valid_stress | M1 e200 | 0.02760 | 5.727 | 0.854 | 1.339 | 0.05557 | 0.09482 | 0.09473 | 0.636 | 0.06428 |
| valid_stress | M1.5 B96 e100 | 0.02854 | 6.612 | 0.871 | 1.485 | 0.05663 | 0.10395 | 0.10751 | 0.686 | 0.07237 |

## Low-DeltaT Diagnostics

`le_0p05` is the existing low raw DeltaT threshold diagnostic. `bin_0` is the
lowest true raw DeltaT percentile bin `[min, p50]`; it is also used here as the
background diagnostic region.

| split | run | checkpoint | le_0p05_mae | le_0p05_bias | le_0p05_over | le_0p05_under | bin0_mae | bin0_bias | bin0_over | bin0_under |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_iid | M1 e100 | final/best | 0.01345 | 0.00872 | 0.773 | 0.227 | 0.01595 | 0.01595 | 1.000 | 0.000 |
| valid_iid | M1 e200 | final/best | 0.01237 | 0.00898 | 0.806 | 0.194 | 0.01351 | 0.01351 | 1.000 | 0.000 |
| valid_iid | M1.5 B96 e100 | final/best | 0.01153 | 0.00749 | 0.779 | 0.221 | 0.01213 | 0.01213 | 1.000 | 0.000 |
| valid_stress | M1 e100 | final/best | 0.01646 | 0.01023 | 0.744 | 0.256 | 0.01864 | 0.01864 | 1.000 | 0.000 |
| valid_stress | M1 e200 | final/best | 0.01569 | 0.01121 | 0.772 | 0.228 | 0.01731 | 0.01731 | 0.997 | 0.003 |
| valid_stress | M1.5 B96 e100 | final/best | 0.01460 | 0.01014 | 0.765 | 0.235 | 0.01569 | 0.01567 | 0.984 | 0.016 |

## Conclusions

1. e200 clearly improves over e100 for scalar loss: valid_iid improves from
   0.4154 to 0.3020 and valid_stress improves from 0.5930 to 0.4554.
2. e200 does not show late degradation in this run. The best checkpoint is
   epoch 200 and final/best is 1.0.
3. M1.5 B96 is trainable where M1.5 B192 was not. It reaches best_epoch 100,
   final/best 1.0, and 800 updates.
4. M1.5 B96 does not beat M1 e200 on valid_iid loss, valid_stress loss, or raw
   DeltaT MSE, but it improves spatial correlation and top-k overlap on both
   splits.
5. Stress field variance remains too high. e200 improves scalar error and
   correlation but worsens stress field_variance_ratio versus M1 e100; M1.5
   B96 further increases variance/amplitude overshoot.
6. Low-DeltaT overprediction is reduced but not solved. On valid_iid, M1.5 B96
   has the lowest bin0/background MAE and bias. On valid_stress, M1.5 B96 also
   lowers bin0 MAE and bias, but bin0 remains mostly overpredicted.
7. Because field_variance_ratio can worsen while scalar loss and bin0 metrics
   improve, this is a tradeoff. Do not claim stable model superiority from a
   single seed.

## Next Recommendation

Keep M1 B192 e200 as the current scalar-loss baseline. Use M1.5 B96 as evidence
that capacity helps spatial correlation/top-k but needs better amplitude
control. The next controlled step should be either M1.5-deeper or gradient
accumulation only if memory allows, while keeping background/bin0 as post-hoc
diagnostics. Do not reintroduce background/bias loss yet; observe whether the
same low-DeltaT overprediction persists across one more capacity/update-count
test before changing loss semantics.

Low-resolution data remains enough for pipeline, split, and baseline work, but
not for final 3D IC local-gradient claims. A small high-res pilot can wait until
capacity/update-count behavior is clearer.
