# V5-P0-1 P5 Physics-Scale Read-Only Audit

## Scope

- Dataset: `heat3d_v4_p5_clean_nohard_v0` (`data/heat3d_v4_p5_clean_nohard_v0`).
- Split map: `configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`.
- Samples audited: `1073` across `train, valid_iid, test_iid, hard_train_holdout, hard_challenge_valid, hard_challenge_test`.
- The audit read arrays and metadata only; it did not write samples, call a solver, modify a model, or train/evaluate.

DeltaT is `temperature.npy - sample_meta.boundary_params.bottom.T_fixed_K`. CV RMS and CV mean use rectilinear control-volume weights inferred from `coords.npy`; target max is the nodewise DeltaT maximum.

## Per-Role Target And Power Summary

| role | n | CV volume mean m3 | effective power mean W | target CV-RMS mean K | target CV-mean mean K | target max max K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 672 | 2.0000e-07 | 1.1941 | 0.6509 | 0.3550 | 14.8861 |
| valid_iid | 128 | 2.0000e-07 | 1.1160 | 0.5344 | 0.3000 | 13.6209 |
| test_iid | 128 | 2.0000e-07 | 1.1203 | 0.6281 | 0.3549 | 13.7722 |
| hard_train_holdout | 121 | 2.0000e-07 | 2.3030 | 6.4905 | 2.7140 | 200.0041 |
| hard_challenge_valid | 12 | 2.0000e-07 | 1.9637 | 7.6217 | 2.6134 | 277.0003 |
| hard_challenge_test | 12 | 2.0000e-07 | 2.2714 | 6.3757 | 2.5173 | 118.3263 |
| all_samples | 1073 | 2.0000e-07 | 1.3217 | 1.4348 | 0.6639 | 277.0003 |

## Physics-Scale Proxies

`R_top = 1 / (top_h * top area)` and `R_z = depth / (top area * CV-harmonic kz)`. `R_series = R_top + R_z`; its DeltaT proxy is effective source power times `R_series`.

| role | harmonic kz median W/m/K | top_h median W/m2/K | R_series median K/W | target mean / series proxy median |
| --- | ---: | ---: | ---: | ---: |
| train | 13.5411 | 857.7570 | 14.6927 | 0.0171 |
| valid_iid | 12.4554 | 871.2487 | 14.3520 | 0.0153 |
| test_iid | 12.9824 | 864.6153 | 13.3308 | 0.0174 |
| hard_train_holdout | 3.0519 | 874.1036 | 23.3041 | 0.0623 |
| hard_challenge_valid | 2.5643 | 1443.6343 | 16.3557 | 0.0660 |
| hard_challenge_test | 3.0269 | 1205.8813 | 15.5850 | 0.0783 |
| all_samples | 11.3107 | 874.1036 | 15.7819 | 0.0199 |

## q/BC Linear Relations

The table reports descriptive Pearson r or R2. The combined predictor is the two-column least-squares fit using effective source power and the top-Robin DeltaT proxy; it is not a causal law because conductivity and source geometry also vary.

| role | power vs top_h r | target mean vs power R2 | target mean vs top-Robin proxy R2 | combined q/BC R2 |
| --- | ---: | ---: | ---: | ---: |
| train | -0.0177 | 0.2156 | 0.1456 | 0.2207 |
| valid_iid | 0.0319 | 0.1646 | 0.1396 | 0.1927 |
| test_iid | 0.0437 | 0.2965 | 0.1354 | 0.2965 |
| hard_train_holdout | 0.0618 | 0.0959 | 0.0137 | 0.1043 |
| hard_challenge_valid | 0.4215 | 0.1533 | 0.0114 | 0.1894 |
| hard_challenge_test | -0.1002 | 0.0060 | 0.1824 | 0.2044 |
| all_samples | 0.0108 | 0.1573 | 0.0625 | 0.1582 |

## Split Duplicate Leakage

- Cross-role model-input duplicate groups: `0`.
- Cross-role full-sample duplicate groups: `0`.
- Cross-role P5 provenance duplicate groups: `0`.
- Audit pass: `True`.
- Shared fixed-grid coordinates alone are expected and are not considered leakage; fingerprints include q/k/BC, and full fingerprints also include temperature.

## Interpretation Limit

All q/BC linear relations are descriptive statistics across heterogeneous P5 scenes, not causal response laws or model metrics.
