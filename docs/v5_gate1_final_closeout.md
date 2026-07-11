# V5 Gate 1 Final Correction And Closeout

## Scope

- Dataset: `heat3d_v4_p5_clean_nohard_v0`; samples: `1073`.
- Read-only audit: no model/loss/config/data change, no training, and no reference-label solver call.
- P0 and the previous Gate 1 layer-averaged result remain intact for backward traceability; this closeout adds corrected per-column z conductance.

## Frozen Operator Semantics

`P_array = sum_all(q*CV)`, `P_bottom = sum_bottom(q*CV)`, and `P_operator = sum_non_bottom(q*CV)`. The latter matches V4 bottom-row replacement: bottom Dirichlet rows receive `T_bottom`, not `q*CV`. `z_collapsed_1d_operator` sums every x-y column's local harmonic-kz face conductance at its actual z spacing.

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
- Best physical valid candidate: `z_collapsed_1d_operator`.
- Runner-up physical valid candidate: `legacy_z_collapsed_1d`.
- Decision: `select_deterministic_physics_base`.
- Selected deterministic base: `z_collapsed_1d_operator`.
- Significant improvement versus constant / power-only: `True` / `True`.

| candidate | valid log-RMSE | valid log-MAE | valid Spearman | valid factor-2 | paired delta CI95 vs constant | hard-challenge-valid log-RMSE | test_iid log-RMSE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| constant | 1.0819 | 0.8831 | n/a | 0.4922 | reference | 2.8732 | 1.1577 |
| power_only | 0.8976 | 0.7232 | 0.4859 | 0.5234 | [-0.2814, -0.0820] | 2.3254 | 1.0028 |
| q_rms_lz2_over_kz | 0.8905 | 0.6862 | 0.5854 | 0.6250 | [-0.3305, -0.0424] | 1.0242 | 0.7316 |
| legacy_p_array_r_series | 0.9093 | 0.7300 | 0.5026 | 0.5703 | [-0.2784, -0.0671] | 2.3157 | 0.9485 |
| source_centroid_two_path | 0.8076 | 0.5927 | 0.6683 | 0.6953 | [-0.4208, -0.1197] | 1.0453 | 0.6509 |
| legacy_z_collapsed_1d | 0.7964 | 0.5829 | 0.6782 | 0.6875 | [-0.4446, -0.1318] | 1.0778 | 0.6467 |
| z_collapsed_1d_operator | 0.6076 | 0.4898 | 0.8068 | 0.7188 | [-0.6091, -0.3284] | 1.8700 | 0.6890 |

### Winner Versus Runner-Up

- Direct paired valid_iid log-RMSE delta (`z_collapsed_1d_operator` minus `legacy_z_collapsed_1d`): `[-0.3338, -0.0460]`.
- Tie-break applied: `False`.

### Valid-IID Calibration Detail

| candidate | valid log-R2 | valid log slope | ratio q05 | ratio q50 | ratio q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | -0.0025 | n/a | 0.2345 | 0.9797 | 6.9492 |
| power_only | 0.3099 | 1.0020 | 0.2727 | 0.9627 | 4.5873 |
| q_rms_lz2_over_kz | 0.3208 | 0.7401 | 0.3651 | 0.8939 | 6.3861 |
| legacy_p_array_r_series | 0.2918 | 0.9663 | 0.2881 | 0.9396 | 6.1067 |
| source_centroid_two_path | 0.4414 | 0.7863 | 0.3679 | 1.0082 | 5.3927 |
| legacy_z_collapsed_1d | 0.4568 | 0.7973 | 0.3977 | 0.9934 | 5.4988 |
| z_collapsed_1d_operator | 0.6838 | 1.0030 | 0.3829 | 1.2343 | 2.4055 |

## Residual Analysis

Residuals are `log(pred/s_y)` for `z_collapsed_1d_operator`. Test-role rows appear only in this post-selection descriptive report.

| feature | n | Spearman rho | slope | R2 |
| --- | ---: | ---: | ---: | ---: |
| q_active_cv_fraction | 1073 | 0.1498 | 0.9350 | 0.0337 |
| q_rms_to_mean_concentration | 1073 | -0.1522 | -0.0551 | 0.0214 |
| source_z_centroid_normalized | 1073 | 0.0494 | 0.5749 | 0.0064 |
| harmonic_kz_W_mK | 1073 | 0.6123 | 0.0218 | 0.2003 |
| anisotropy_xy_over_z | 1073 | -0.0421 | -0.0834 | 0.0021 |
| top_h_W_m2K | 1073 | 0.0396 | 2.5767e-05 | 5.5279e-04 |
| T_inf_minus_T_bottom_K | 1073 | n/a | n/a | n/a |

| role family | n | residual mean | residual median | residual std |
| --- | ---: | ---: | ---: | ---: |
| clean | 928 | 0.0221 | 0.1651 | 0.7094 |
| hard | 145 | -1.5061 | -1.4997 | 0.8923 |

### Hard OOD Systematic Bias

- Role: `hard_challenge_valid`; candidate: `z_collapsed_1d_operator`.
- Direction: `systematic_underprediction`; mean/median log residual: `-1.6919` / `-1.5875`.
- Hard OOD log-RMSE: `1.8700`; prediction/target median ratio: `0.2046`.

## Integrity And Reproducibility

- Per-sample table rows: `1073`; SHA256: `79b7f79c32ac5c3da100e27ebafeeea25cb185088687785c6140f0359bde7de9`.
- Cross-role model-input/full-sample/provenance duplicate groups: `0` / `0` / `0`.
- `--verify-summary` regenerates split summaries, calibration, predictions, selection, and residual analysis from the CSV only.
