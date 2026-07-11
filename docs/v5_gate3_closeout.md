# V5 Gate 3 Boundary-General Shape–Scale Oracle Closeout

## Scope And Frozen Semantics

- Frozen V5 semantics: bottom Dirichlet, top Robin, sides adiabatic; `DeltaT = T - T_bottom`.
- `scale` is CV-weighted RMS(`DeltaT`); `shape = DeltaT / (scale + eps)`.
- The implementation reads region type, masks, and prescribed values from metadata plus BC features. It does not infer a Dirichlet location from z coordinates in this V5 audit.
- This is a read-only diagnostic of frozen V4P5_02 best/final raw-temperature predictions. No model, loss, configuration, data, or training changed.

## Boundary And Reconstruction Invariants

- Samples decomposed: `1073`; nonzero target scales: `1073`; all invariant passes: `1073`.
- Target reconstruction max abs error: `0.000000000000` K; Dirichlet projection max error: `0.000000000000` K.
- Coordinate fallback used: `0`. Current boundary signature(s): `{'bottom:dirichlet:256;side:adiabatic:120;top:robin:256': 1073}`.
- Projection occurs in raw physical temperature space: it sets only Dirichlet nodes to their prescribed values and leaves non-Dirichlet values unchanged.

## Sample-First CV-Weighted Oracle Metrics

`predicted_shape_true_scale` is the shape-only oracle; `true_shape_predicted_scale` is the scale-only oracle. Amplitude ratio is the weighted projection onto the target DeltaT, while CV-RMS ratio is the uncentered field RMS ratio.

### clean

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 0.0680 | 0.0544 | 0.0345 | 0.0679 | 0.9799 | 1.0408 | 1.0561 | 0.1787 |
| final | 0.0634 | 0.0498 | 0.0343 | 0.0634 | 0.9817 | 1.0365 | 1.0499 | 0.1640 |

### hard

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 4.4561 | 2.0175 | 4.2707 | 4.4561 | 0.9457 | 0.4294 | 0.4481 | 17.5980 |
| final | 4.4452 | 2.0449 | 4.2471 | 4.4452 | 0.9440 | 0.4345 | 0.4543 | 17.6008 |

### train

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 0.0376 | 0.0340 | 0.0125 | 0.0376 | 0.9863 | 1.0404 | 1.0512 | 0.0823 |
| final | 0.0312 | 0.0272 | 0.0129 | 0.0312 | 0.9888 | 1.0324 | 1.0402 | 0.0638 |

### valid_iid

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 0.1294 | 0.0981 | 0.0754 | 0.1294 | 0.9617 | 1.0391 | 1.0666 | 0.3738 |
| final | 0.1318 | 0.0989 | 0.0765 | 0.1318 | 0.9626 | 1.0456 | 1.0725 | 0.3757 |

### test_iid

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 0.1656 | 0.1179 | 0.1091 | 0.1656 | 0.9645 | 1.0447 | 1.0713 | 0.4895 |
| final | 0.1641 | 0.1197 | 0.1040 | 0.1641 | 0.9636 | 1.0493 | 1.0776 | 0.4784 |

### hard_train_holdout

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 4.3410 | 1.9783 | 4.1526 | 4.3410 | 0.9464 | 0.4374 | 0.4563 | 17.0033 |
| final | 4.3297 | 2.0051 | 4.1282 | 4.3297 | 0.9447 | 0.4428 | 0.4629 | 17.0061 |

### hard_challenge_valid

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 5.7113 | 2.2869 | 5.5503 | 5.7113 | 0.9484 | 0.4028 | 0.4198 | 23.6549 |
| final | 5.7078 | 2.3367 | 5.5337 | 5.7078 | 0.9455 | 0.4056 | 0.4240 | 23.6672 |

### hard_challenge_test

| checkpoint | original RMSE K | shape-only RMSE K | scale-only RMSE K | projected RMSE K | original corr | original amp | original CV-RMS ratio | hotspot RMSE K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 4.3621 | 2.1435 | 4.1819 | 4.3621 | 0.9361 | 0.3752 | 0.3937 | 17.5374 |
| final | 4.3464 | 2.1535 | 4.1587 | 4.3464 | 0.9351 | 0.3793 | 0.3985 | 17.5309 |

## Hard-Failure Decomposition And Gate 4 Direction

- `best`: original / shape-only / scale-only / boundary-projected hard RMSE = `4.4561` / `2.0175` / `4.2707` / `4.4561` K.
  Gate 4 direction: `prioritize_scale_path_diagnostics_before_any_Gate_4_model_change`.
- `final`: original / shape-only / scale-only / boundary-projected hard RMSE = `4.4452` / `2.0449` / `4.2471` / `4.4452` K.
  Gate 4 direction: `prioritize_scale_path_diagnostics_before_any_Gate_4_model_change`.

These counterfactual components are non-additive diagnostic evidence; they do not themselves establish a causal mechanism.

## Lateral-Spreading Mechanism Evidence

The following relations use the frozen Gate 1 corrected `z_collapsed_1d_operator` scale log residual. They are descriptive associations, not causal claims.

| group | feature | n | Spearman rho | slope | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| all_samples | q_weighted_local_kz_W_mK | 1073 | 0.5124 | 0.0035 | 0.1161 |
| all_samples | q_weighted_inverse_kz_mK_W | 1073 | -0.7855 | -1.9614 | 0.5245 |
| all_samples | q_low_k_overlap_fraction | 1073 | -0.4646 | -1.0813 | 0.1935 |
| all_samples | source_layer_kz_heterogeneity_cv | 1073 | -0.5459 | -0.8128 | 0.2466 |
| all_samples | source_concentration | 1073 | -0.1522 | -0.0551 | 0.0214 |
| all_samples | source_z_centroid_m | 1073 | 0.0490 | 287.4490 | 0.0064 |
| all_samples | source_z_centroid_normalized | 1073 | 0.0490 | 0.5749 | 0.0064 |
| clean | q_weighted_local_kz_W_mK | 928 | 0.4378 | 0.0023 | 0.0856 |
| clean | q_weighted_inverse_kz_mK_W | 928 | -0.7080 | -2.1117 | 0.3526 |
| clean | q_low_k_overlap_fraction | 928 | -0.4209 | -0.7369 | 0.1425 |
| clean | source_layer_kz_heterogeneity_cv | 928 | -0.4748 | -0.6281 | 0.1974 |
| clean | source_concentration | 928 | -0.1292 | -0.0339 | 0.0133 |
| clean | source_z_centroid_m | 928 | 0.0271 | 133.8190 | 0.0022 |
| clean | source_z_centroid_normalized | 928 | 0.0271 | 0.2676 | 0.0022 |
| hard | q_weighted_local_kz_W_mK | 145 | 0.2379 | -0.0013 | 0.0027 |
| hard | q_weighted_inverse_kz_mK_W | 145 | -0.7299 | -1.1259 | 0.4603 |
| hard | q_low_k_overlap_fraction | 145 | -0.4406 | -1.2588 | 0.1959 |
| hard | source_layer_kz_heterogeneity_cv | 145 | -0.1235 | -0.1778 | 0.0143 |
| hard | source_concentration | 145 | -0.2599 | -0.0874 | 0.0481 |
| hard | source_z_centroid_m | 145 | 0.2657 | 871.2261 | 0.0612 |
| hard | source_z_centroid_normalized | 145 | 0.2657 | 1.7425 | 0.0612 |
| train | q_weighted_local_kz_W_mK | 672 | 0.4367 | 0.0024 | 0.0869 |
| train | q_weighted_inverse_kz_mK_W | 672 | -0.7214 | -2.0175 | 0.3514 |
| train | q_low_k_overlap_fraction | 672 | -0.4469 | -0.8035 | 0.1556 |
| train | source_layer_kz_heterogeneity_cv | 672 | -0.5084 | -0.7232 | 0.2395 |
| train | source_concentration | 672 | -0.0857 | -0.0226 | 0.0057 |
| train | source_z_centroid_m | 672 | 0.0346 | 156.0951 | 0.0029 |
| train | source_z_centroid_normalized | 672 | 0.0346 | 0.3122 | 0.0029 |
| valid_iid | q_weighted_local_kz_W_mK | 128 | 0.4002 | 0.0016 | 0.0769 |
| valid_iid | q_weighted_inverse_kz_mK_W | 128 | -0.6215 | -2.7355 | 0.3493 |
| valid_iid | q_low_k_overlap_fraction | 128 | -0.4559 | -0.7274 | 0.1835 |
| valid_iid | source_layer_kz_heterogeneity_cv | 128 | -0.3288 | -0.3052 | 0.0585 |
| valid_iid | source_concentration | 128 | -0.2444 | -0.0600 | 0.0591 |
| valid_iid | source_z_centroid_m | 128 | 0.2042 | 479.5172 | 0.0406 |
| valid_iid | source_z_centroid_normalized | 128 | 0.2042 | 0.9590 | 0.0406 |
| test_iid | q_weighted_local_kz_W_mK | 128 | 0.4801 | 0.0026 | 0.0963 |
| test_iid | q_weighted_inverse_kz_mK_W | 128 | -0.7165 | -2.4978 | 0.3788 |
| test_iid | q_low_k_overlap_fraction | 128 | -0.2294 | -0.4409 | 0.0644 |
| test_iid | source_layer_kz_heterogeneity_cv | 128 | -0.4395 | -0.4756 | 0.1474 |
| test_iid | source_concentration | 128 | -0.2364 | -0.0714 | 0.0523 |
| test_iid | source_z_centroid_m | 128 | -0.1685 | -299.7079 | 0.0118 |
| test_iid | source_z_centroid_normalized | 128 | -0.1685 | -0.5994 | 0.0118 |
| hard_train_holdout | q_weighted_local_kz_W_mK | 121 | 0.2631 | -0.0016 | 0.0037 |
| hard_train_holdout | q_weighted_inverse_kz_mK_W | 121 | -0.7244 | -1.1581 | 0.4422 |
| hard_train_holdout | q_low_k_overlap_fraction | 121 | -0.4180 | -1.2476 | 0.1819 |
| hard_train_holdout | source_layer_kz_heterogeneity_cv | 121 | -0.1714 | -0.2285 | 0.0231 |
| hard_train_holdout | source_concentration | 121 | -0.2246 | -0.0699 | 0.0284 |
| hard_train_holdout | source_z_centroid_m | 121 | 0.2436 | 823.1626 | 0.0525 |
| hard_train_holdout | source_z_centroid_normalized | 121 | 0.2436 | 1.6463 | 0.0525 |
| hard_challenge_valid | q_weighted_local_kz_W_mK | 12 | -0.2308 | -0.0083 | 0.1186 |
| hard_challenge_valid | q_weighted_inverse_kz_mK_W | 12 | -0.7832 | -0.8700 | 0.6244 |
| hard_challenge_valid | q_low_k_overlap_fraction | 12 | -0.4659 | -1.1077 | 0.1988 |
| hard_challenge_valid | source_layer_kz_heterogeneity_cv | 12 | 0.2168 | 0.3677 | 0.0302 |
| hard_challenge_valid | source_concentration | 12 | -0.7762 | -0.2914 | 0.4910 |
| hard_challenge_valid | source_z_centroid_m | 12 | 0.5141 | 1057.8162 | 0.1031 |
| hard_challenge_valid | source_z_centroid_normalized | 12 | 0.5141 | 2.1156 | 0.1031 |
| hard_challenge_test | q_weighted_local_kz_W_mK | 12 | 0.3636 | 0.0106 | 0.1943 |
| hard_challenge_test | q_weighted_inverse_kz_mK_W | 12 | -0.8112 | -1.7520 | 0.6918 |
| hard_challenge_test | q_low_k_overlap_fraction | 12 | -0.7321 | -1.6379 | 0.5013 |
| hard_challenge_test | source_layer_kz_heterogeneity_cv | 12 | 0.2587 | 0.0338 | 0.0008 |
| hard_challenge_test | source_concentration | 12 | -0.2727 | -0.1113 | 0.1749 |
| hard_challenge_test | source_z_centroid_m | 12 | 0.2448 | 1063.0112 | 0.1055 |
| hard_challenge_test | source_z_centroid_normalized | 12 | 0.2448 | 2.1260 | 0.1055 |

## Integrity And Reproducibility

- Per-sample CSV: `1073` rows; SHA256 `ff5ab6eaf49401079d0ddd034601ad470f22b158e85ada651d24d013344d5bd8`.
- Cross-role input/full/provenance duplicate groups: `0` / `0` / `0`.
- `--verify-summary` independently reconstructs all target, oracle, hard-failure, lateral, and leakage summaries from the CSV only.
- Test roles are frozen descriptive reports only; no Gate 3 threshold, formula, or method was selected from them.
