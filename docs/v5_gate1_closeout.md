# V5 Gate 1 Closeout: Operator-Consistent Physics Scale

## Scope

- Dataset: `heat3d_v4_p5_clean_nohard_v0`; samples: `1073`.
- Read-only audit: no model/loss/config/data change, no training, and no reference-label solver call.
- P0 remains intact; its historical effective source power is preserved here as `P_array`.

## Frozen Operator Semantics

`P_array = sum_all(q*CV)`, `P_bottom = sum_bottom(q*CV)`, and `P_operator = sum_non_bottom(q*CV)`. The latter matches V4 bottom-row replacement: bottom Dirichlet rows receive `T_bottom`, not `q*CV`.

| role | n | P_array mean W | P_bottom mean W | P_operator mean W | BC offset range K | bottom label max error K | driver categories |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| train | 672 | 1.1941 | 0.0000 | 1.1941 | [0.0000, 0.0000] | 0.0000 | source_driven:672 |
| valid_iid | 128 | 1.1160 | 0.0000 | 1.1160 | [0.0000, 0.0000] | 0.0000 | source_driven:128 |
| test_iid | 128 | 1.1203 | 0.0000 | 1.1203 | [0.0000, 0.0000] | 0.0000 | source_driven:128 |
| hard_train_holdout | 121 | 2.3030 | 0.0000 | 2.3030 | [0.0000, 0.0000] | 0.0000 | source_driven:121 |
| hard_challenge_valid | 12 | 1.9637 | 0.0000 | 1.9637 | [0.0000, 0.0000] | 0.0000 | source_driven:12 |
| hard_challenge_test | 12 | 2.2714 | 0.0000 | 2.2714 | [0.0000, 0.0000] | 0.0000 | source_driven:12 |
| all_samples | 1073 | 1.3217 | 0.0000 | 1.3217 | [0.0000, 0.0000] | 0.0000 | source_driven:1073 |

## Calibration And Selection

All calibrations fit only `train`; candidate selection uses only `valid_iid`; `hard_challenge_valid` is OOD inspection; test roles are report-only after selection.
- Best physical valid candidate: `z_collapsed_1d`.
- Decision: `select_deterministic_physics_base`.
- Selected deterministic base: `z_collapsed_1d`.

| candidate | valid log-RMSE | valid log-MAE | valid Spearman | valid factor-2 | paired delta CI95 vs constant | hard-challenge-valid log-RMSE | test_iid log-RMSE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| constant | 1.0819 | 0.8831 | n/a | 0.4922 | reference | 2.8732 | 1.1577 |
| power_only | 0.8976 | 0.7232 | 0.4859 | 0.5234 | [-0.2801, -0.0822] | 2.3254 | 1.0028 |
| q_rms_lz2_over_kz | 0.8905 | 0.6862 | 0.5854 | 0.6250 | [-0.3373, -0.0459] | 1.0242 | 0.7316 |
| legacy_p_array_r_series | 0.9093 | 0.7300 | 0.5026 | 0.5703 | [-0.2645, -0.0717] | 2.3157 | 0.9485 |
| source_centroid_two_path | 0.8076 | 0.5927 | 0.6683 | 0.6953 | [-0.4347, -0.1197] | 1.0453 | 0.6509 |
| z_collapsed_1d | 0.7964 | 0.5829 | 0.6782 | 0.6875 | [-0.4317, -0.1284] | 1.0778 | 0.6467 |

### Valid-IID Calibration Detail

| candidate | valid log-R2 | valid log slope | ratio q05 | ratio q50 | ratio q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | -0.0025 | n/a | 0.2345 | 0.9797 | 6.9492 |
| power_only | 0.3099 | 1.0020 | 0.2727 | 0.9627 | 4.5873 |
| q_rms_lz2_over_kz | 0.3208 | 0.7401 | 0.3651 | 0.8939 | 6.3861 |
| legacy_p_array_r_series | 0.2918 | 0.9663 | 0.2881 | 0.9396 | 6.1067 |
| source_centroid_two_path | 0.4414 | 0.7863 | 0.3679 | 1.0082 | 5.3927 |
| z_collapsed_1d | 0.4568 | 0.7973 | 0.3977 | 0.9934 | 5.4988 |

## Residual Analysis

Residuals are `log(pred/s_y)` for `z_collapsed_1d`. Test-role rows appear only in this post-selection descriptive report.

| feature | n | Spearman rho | slope | R2 |
| --- | ---: | ---: | ---: | ---: |
| T_inf_minus_T_bottom_K | 1073 | n/a | n/a | n/a |
| anisotropy_xy_over_z | 1073 | 0.0885 | 0.0991 | 0.0043 |
| harmonic_kz_W_mK | 1073 | 0.1939 | 0.0045 | 0.0120 |
| q_active_cv_fraction | 1073 | 0.2457 | 0.8303 | 0.0379 |
| q_rms_to_mean_concentration | 1073 | -0.2388 | -0.0537 | 0.0289 |
| source_z_centroid_normalized | 1073 | 0.1307 | 0.7990 | 0.0176 |
| top_h_W_m2K | 1073 | 0.0020 | -2.2910e-05 | 6.2214e-04 |

| role family | n | residual mean | residual median | residual std |
| --- | ---: | ---: | ---: | ---: |
| clean | 928 | 0.0187 | -0.0652 | 0.7107 |
| hard | 145 | -0.8914 | -0.8965 | 0.5399 |

## Integrity And Reproducibility

- Per-sample table rows: `1073`; SHA256: `7d17589b909157670b5c052ae9ff901e1cff39bbff77e0e7cf7e60055b89e8c7`.
- Cross-role model-input/full-sample/provenance duplicate groups: `0` / `0` / `0`.
- `--verify-summary` regenerates split summaries, calibration, predictions, selection, and residual analysis from the CSV only.
