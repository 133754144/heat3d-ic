# V4P5_01 Split-Aware Diagnostics

Read this file only for the V4P5 clean-IID baseline, hard-challenge, or P5
follow-up configuration decisions.

## Scope

- Checkpoints: `V4P5_01` best (epoch 198) and final (epoch 200), read-only.
- Dataset/split: `heat3d_v4_p5_clean_nohard_v0` and
  `candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0`.
- Missing test/hard predictions were exported by read-only checkpoint inference
  to ignored `output/heat3d_v4_offline_diagnostics/V4P5_01/`; no training ran.
- `all_iid_*` is a reporting union of the corresponding clean and hard split;
  it is not a training split.

`RMSE`, `MAE`, `corr`, `cosine`, `amp`, `top5`, and `strong-q` are computed
per sample and then averaged. `rel%` is point-global raw DeltaT RMSE divided by
point-global mean absolute true DeltaT. `low` reports
`point fraction / RMSE K / over-prediction fraction` for true DeltaT <= 0.05 K.
Top5 is the per-sample true-DeltaT p95 tail; strong-q is positive q at or above
the per-sample positive-q p90.

## Training Scalars

| checkpoint | epoch | valid_base_mse | raw RMSE K | rel_rmse_v4_pct |
| --- | ---: | ---: | ---: | ---: |
| best | 198 | 0.04967 | 0.178 | 70.1 |
| final | 200 | 0.05064 | 0.179 | 70.8 |

The best checkpoint remains the selection checkpoint because it has the lower
`valid_base_mse` and slightly better clean tail metrics.

## Best Checkpoint

| cohort | n | RMSE K | MAE K | rel% | corr | cosine | amp | top5 K | strong-q K | low: fraction / RMSE / over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clean valid_iid | 128 | 0.124 | 0.062 | 70.1 | 0.953 | 0.968 | 1.074 | 0.369 | 0.416 | 50.7% / 0.019 / 0.589 |
| clean test_iid | 128 | 0.151 | 0.076 | 75.1 | 0.952 | 0.968 | 1.084 | 0.442 | 0.448 | 51.0% / 0.017 / 0.600 |
| hard challenge valid | 12 | 4.527 | 1.194 | 370.1 | 0.944 | 0.949 | 0.385 | 18.946 | 42.458 | 38.2% / 0.018 / 0.498 |
| hard challenge test | 12 | 3.494 | 1.057 | 208.1 | 0.931 | 0.938 | 0.340 | 14.442 | 27.981 | 35.5% / 0.014 / 0.402 |
| all_iid valid | 140 | 0.502 | 0.159 | 534.9 | 0.952 | 0.966 | 1.015 | 1.962 | 4.020 | 49.6% / 0.019 / 0.583 |
| all_iid test | 140 | 0.438 | 0.160 | 275.9 | 0.950 | 0.966 | 1.020 | 1.642 | 2.808 | 49.7% / 0.017 / 0.588 |
| original clean valid | 116 | 0.121 | 0.059 | 66.6 | 0.954 | 0.968 | 1.076 | 0.358 | 0.408 | 50.6% / 0.019 / 0.587 |
| generated replacement valid | 12 | 0.159 | 0.085 | 92.5 | 0.946 | 0.965 | 1.052 | 0.483 | 0.500 | 51.9% / 0.013 / 0.608 |
| original clean test | 116 | 0.159 | 0.079 | 74.9 | 0.951 | 0.968 | 1.078 | 0.468 | 0.478 | 51.1% / 0.017 / 0.599 |
| generated replacement test | 12 | 0.079 | 0.047 | 55.8 | 0.958 | 0.972 | 1.138 | 0.183 | 0.159 | 50.1% / 0.013 / 0.606 |

## Final Checkpoint

| cohort | n | RMSE K | MAE K | rel% | corr | cosine | amp | top5 K | strong-q K | low: fraction / RMSE / over |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clean valid_iid | 128 | 0.124 | 0.061 | 70.7 | 0.952 | 0.967 | 1.047 | 0.374 | 0.419 | 50.7% / 0.018 / 0.577 |
| clean test_iid | 128 | 0.151 | 0.075 | 75.9 | 0.952 | 0.968 | 1.059 | 0.446 | 0.450 | 51.0% / 0.016 / 0.588 |
| hard challenge valid | 12 | 4.537 | 1.200 | 370.3 | 0.945 | 0.950 | 0.383 | 18.975 | 42.475 | 38.2% / 0.017 / 0.488 |
| hard challenge test | 12 | 3.503 | 1.066 | 208.5 | 0.933 | 0.940 | 0.339 | 14.465 | 27.990 | 35.5% / 0.014 / 0.394 |
| all_iid valid | 140 | 0.502 | 0.159 | 535.3 | 0.951 | 0.966 | 0.990 | 1.968 | 4.023 | 49.6% / 0.018 / 0.571 |
| all_iid test | 140 | 0.438 | 0.160 | 276.6 | 0.950 | 0.966 | 0.997 | 1.648 | 2.811 | 49.7% / 0.016 / 0.576 |
| original clean valid | 116 | 0.120 | 0.058 | 66.4 | 0.953 | 0.968 | 1.050 | 0.361 | 0.408 | 50.6% / 0.019 / 0.576 |
| generated replacement valid | 12 | 0.165 | 0.089 | 97.8 | 0.942 | 0.963 | 1.016 | 0.501 | 0.519 | 51.9% / 0.013 / 0.588 |
| original clean test | 116 | 0.159 | 0.079 | 75.7 | 0.951 | 0.968 | 1.055 | 0.473 | 0.480 | 51.1% / 0.016 / 0.587 |
| generated replacement test | 12 | 0.078 | 0.046 | 54.7 | 0.958 | 0.972 | 1.095 | 0.193 | 0.168 | 50.1% / 0.012 / 0.598 |

## Hard-Tail And Provenance Decomposition

| checkpoint | hard contribution to all-IID point MSE, valid | test |
| --- | ---: | ---: |
| best | 99.37% | 96.91% |
| final | 99.36% | 96.86% |

- The hard tail, not low-DeltaT background points, dominates all-IID error:
  low-DeltaT RMSE stays `0.014-0.019 K`, while hard `top5` and `strong-q`
  errors reach `14.4-19.0 K` and `28.0-42.5 K` respectively.
- Hard samples retain relatively high average correlation (`0.931-0.945`) but
  have an amplitude ratio of only `0.339-0.385`. This is primarily a
  high-amplitude/strong-q scale failure, rather than a uniformly failed field
  shape.
- The generated-replacement subgroups are small (`n=12` each): validation is
  modestly worse than original clean data, while test is better. This is useful
  monitoring evidence, not a distribution-quality conclusion.

## Final-Probe Summary

`clean probe average` means P01/P03/P04/P05/P07/P08/P10; it excludes P02
(disconnected path), P06 (anisotropic power), and P09 (diag3 mismatch).

| checkpoint | all probe mean RMSE / corr | clean probe mean RMSE / corr | P02 RMSE / corr / amp | P06 RMSE / corr / amp | P09 RMSE / corr / amp |
| --- | --- | --- | --- | --- | --- |
| best | 0.745 / 0.856 | 0.108 / 0.963 | 3.502 / 0.571 / 1.467 | 2.704 / 0.330 / 3.475 | 0.486 / 0.914 / 0.206 |
| final | 0.745 / 0.854 | 0.106 / 0.963 | 3.510 / 0.568 / 1.465 | 2.705 / 0.317 / 3.473 | 0.487 / 0.915 / 0.207 |

P02 and P06 are the principal final-probe error sources: both have multi-K
RMSE and poor shape correlation. P09 has moderate RMSE and preserved shape but
a severe amplitude collapse. The hard tail remains an independent
curriculum/fine-tune/feature-redesign research line: this baseline does not
restore `hard_weight=2.0` and does not perform hard fine-tuning.

## Planned Controls

- `V4P5_02_clean_baseline_raw_B28_e600` is the raw-coordinate long-schedule
  control. Relative to V4P5_01, only epochs change from 200 to 600, together
  with identity and ignored output paths.
- `V4P5_03_clean_fourier_freq4_B_safe` is the coordinate-encoding ablation.
  Relative to V4P5_01, it changes only to `raw_plus_fourier` with four
  frequencies, together with identity and ignored output paths.
- Both are `planned` and `explicit_user_instruction_only`; this diagnostics
  task did not launch either configuration.
