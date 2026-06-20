# V4 P1 Target Normalization Scale Audit

Read this file only for V4 P1 target/normalization scale, final-probe
amplitude, or model-lab merge review questions.

## Scope

This is a read-only diagnostic audit. It did not train, start tmux, create a
new run, or modify model, solver, loss, loader, registry CSV, data, checkpoints,
logs, or prediction artifacts.

Remote audit source:

- host: devbox
- branch during artifact read: `research/v4`
- audit code: temporary `/tmp/heat3d_v4_p1_scale_audit.py`
- full temporary JSON: `/tmp/heat3d_v4_p1_target_normalization_scale_audit.json`
- checkpoint: `output/heat3d_v4_runs/V4Test00_baseline_seed_0/params_best.pkl`
- checkpoint kind/epoch: `best`, epoch `596`
- run config: `output/heat3d_v4_runs/V4Test00_baseline_seed_0/run_config.json`

Final-probe BC masks used `coords_extrema_reconstructed`, matching the P1.0b
compatibility policy. The resulting final-probe BC fractions were top 0.25,
bottom 0.25, side 0.234375, and interior 0.3828125.

## Target Scale

`valid` below is the runner primary validation split, `valid_iid`. The V4Test00
run split map resolved 704 train samples, 104 `valid_iid` samples, and 88
`valid_stress` samples.

Checkpoint train target normalization:

- `target_delta_mean = 0.0262432918 K`
- `target_delta_std = 0.0431969985 K`
- recovery formula: `DeltaT_pred = pred_norm * train_deltaT_std + train_deltaT_mean`
- recovery formula max absolute check error: `4.997e-08 K`

| split | samples | raw DeltaT p99 | raw DeltaT max | normalized DeltaT p99 | normalized DeltaT max |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 704 | 0.202 K | 0.998 K | 4.069 | 22.505 |
| valid_iid | 104 | 0.210 K | 0.980 K | 4.252 | 22.077 |
| final_probe | 10 | 2.522 K | 7.709 K | 57.783 | 177.848 |

Final-probe target amplitudes are therefore extreme under train normalization:
the largest final-probe DeltaT is about `178 sigma` in normalized target space.
This is not caused by raw recovery; the recovery is a verified linear inverse.

## Condition Scale

Raw ranges and train z-score ranges:

| split | k raw range | k z max | q max | q z max | q z p99 | top_h max | top_h z max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4.402 to 290.989 | 3.996 | 1.509e8 | 34.647 | 3.956 | 1714.586 | 1.968 |
| valid_iid | 4.420 to 271.579 | 3.637 | 1.585e8 | 36.389 | 3.827 | 1706.809 | 1.948 |
| final_probe | 0.668 to 423.366 | 6.443 | 1.940e8 | 44.573 | 18.882 | 3400.000 | 6.335 |

Final-probe is outside the runner train distribution in target DeltaT, k,
q, and top_h. The q z-score is especially sparse-tail sensitive: max z is high
even in train, but final-probe p99 and max are much larger.

## Model Output Scale

V4Test00 seed0 best checkpoint, final-probe inference:

| space | pred min | pred max | label min | label max | peak ratio | range ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized DeltaT | -0.608 | 17.725 | -0.608 | 177.848 | 0.0997 | 0.1027 |
| raw DeltaT K | -0.00001 | 0.792 | 0.000 | 7.709 | 0.1027 | 0.1027 |

Per-sample final-probe summary:

| metric | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: |
| RMSE_K | 0.216 | 0.351 | 0.114 | 0.966 |
| relRMSE_DeltaT | 0.711 | 0.740 | 0.607 | 0.910 |
| peak_error_K | -2.193 | -3.089 | -6.989 | -0.901 |
| mean_bias_K | -0.116 | -0.123 | -0.285 | -0.018 |
| scale_ratio | 0.245 | 0.240 | 0.093 | 0.458 |
| range_ratio | 0.245 | 0.240 | 0.093 | 0.458 |
| centered_corr | 0.860 | 0.857 | 0.805 | 0.912 |

Conclusion: amplitude compression is already present in normalized model
output. Raw recovery does not introduce it; it preserves the same scale/range
ratio up to the linear train `target_delta_std`.

## Post-Hoc Calibration

Calibration was fit only on final-probe raw DeltaT as a diagnostic, not as a
model improvement or deployable evaluation protocol.

| variant | coefficients | RMSE_K | relRMSE_DeltaT | peak_error_K | scale_ratio | centered_corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uncalibrated | `a=1, b=0` | 0.437 | 0.842 | -6.917 | 0.103 | 0.742 |
| scale-only | `a=3.541, b=0` | 0.328 | 0.631 | -4.904 | 0.364 | 0.742 |
| linear | `a=3.792, b=-0.0517` | 0.325 | 0.626 | -4.758 | 0.383 | 0.742 |

Simple global calibration improves RMSE and relative RMSE substantially while
leaving centered correlation unchanged, so scale/amplitude failure is a major
component. The improvement is incomplete, and peak error remains large, so
shape/model extrapolation and condition OOD still matter.

## Decision

For V4 P1, the final-probe scale_ratio near 0.2 is not a target recovery bug.
It reflects model output amplitude compression under extreme normalized target
and condition extrapolation. The next research question should be whether to
change the target/condition scaling policy or the training distribution before
changing model architecture.
