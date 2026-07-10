# Heat3D-IC V4 Closeout

Read this file for the frozen V4 baseline, P5 evaluation protocol, or V5
handoff. It supersedes the earlier P5_01 planning status; historical audit
documents remain evidence for their own runs.

## Frozen Decision

- Frozen clean baseline: `V4P5_02_clean_baseline_raw_B28_e600`.
- Formal model checkpoint: `params_best.pkl` at epoch 405, selected only by
  normalized `valid_base_mse`.
- Training population: `heat3d_v4_p5_clean_nohard_v0`, with plain MSE, raw
  coordinates, B28, seed 0, and the P5 formal split map.
- The epoch-600 final checkpoint is retained as a trajectory control; it is
  not the selected baseline checkpoint.
- `V4P5_03_clean_fourier_freq4_B_safe` is a completed negative ablation. It
  is not a V5 default.

This closeout reports sample-first statistics unless a cell explicitly says
`point-global`: each sample metric is calculated first, then averaged within
the cohort. Point-global raw DeltaT RMSE and relative RMSE are shown separately
for physical-scale cross-checks and are not substitutes for the sample-first
metric contract.

## Dataset And Protocol

| item | value |
| --- | --- |
| dataset | `data/heat3d_v4_p5_clean_nohard_v0` |
| split map | `configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json` |
| clean train / valid / test | 672 / 128 / 128 |
| hard train holdout / challenge valid / challenge test | 121 / 12 / 12 |
| clean `physical_hard_keep` count | 0 |
| all-IID | reporting union of the matching clean and hard split; never a training split |

The 49 generated clean replacements are unique, solver-accepted additions to
the clean pool. The original `physical_hard_keep` samples remain unmodified in
the hard holdout/challenge roles.

## Artifact Audit

| run | source | required run files | final probe | closeout split exports |
| --- | --- | --- | --- | --- |
| V4P5_02 | wsl2 | best/final params, run config, loss summary, best/final clean-valid predictions | best/final complete | best/final clean-test and hard valid/test complete |
| V4P5_03 | devbox | best/final params, run config, loss summary, best/final clean-valid predictions | best/final complete | best/final clean-test and hard valid/test complete |

Both runs recorded `status_ok=true`, finite gradients, CPU-resident best
parameters, and `all_groups_status=skipped`. The latter is expected for
`prediction_split=valid_iid`. Post-training all-sample diagnostics were
therefore skipped; this closeout uses dedicated split-aware checkpoint
inference written only under ignored `output/heat3d_v4_offline_diagnostics/`.

## V4P5_02 Training Cross-Check

| checkpoint | epoch | normalized valid_base_mse | point-global raw DeltaT RMSE K | point-global rel_rmse_v4_pct |
| --- | ---: | ---: | ---: | ---: |
| best | 405 | 0.04565 | 0.170 | 67.17 |
| final | 600 | 0.04619 | 0.171 | 67.57 |

The e600 run slightly regressed after the selected epoch: final/best normalized
validation ratio is 1.0118. The final train loss was 0.00161 versus final
validation base MSE 0.04619, so longer optimization is not justification for
replacing the selected best checkpoint.

### Best Checkpoint: Three Reporting Views

`SF RMSE/MAE`, corr, cosine, amplitude, top5, and strong-q are sample-first
means. `global RMSE` and `rel%` are point-global raw DeltaT values. `low` is
true DeltaT <= 0.05 K as `point-global RMSE K / overprediction fraction`.

| cohort | n | SF RMSE K | SF MAE K | global RMSE K | rel% | corr | cosine | amp | top5 K | strong-q K | low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clean valid_iid | 128 | 0.119 | 0.058 | 0.170 | 67.2 | 0.962 | 0.974 | 1.095 | 0.355 | 0.397 | 0.025 / 0.642 |
| clean test_iid | 128 | 0.153 | 0.074 | 0.236 | 79.0 | 0.964 | 0.976 | 1.076 | 0.467 | 0.505 | 0.020 / 0.647 |
| hard_challenge_valid | 12 | 4.709 | 1.251 | 7.450 | 377.8 | 0.946 | 0.952 | 0.345 | 19.728 | 43.530 | 0.011 / 0.558 |
| hard_challenge_test | 12 | 3.683 | 1.130 | 4.286 | 217.4 | 0.934 | 0.941 | 0.320 | 15.179 | 29.023 | 0.010 / 0.523 |
| all_iid_valid | 140 | 0.512 | 0.161 | 2.187 | 545.9 | 0.960 | 0.972 | 1.031 | 2.015 | 4.094 | 0.024 / 0.637 |
| all_iid_test | 140 | 0.455 | 0.165 | 1.275 | 288.3 | 0.961 | 0.973 | 1.011 | 1.728 | 2.949 | 0.020 / 0.640 |

### Final Checkpoint: Three Reporting Views

| cohort | n | SF RMSE K | SF MAE K | global RMSE K | rel% | corr | cosine | amp | top5 K | strong-q K | low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clean valid_iid | 128 | 0.121 | 0.059 | 0.171 | 67.6 | 0.962 | 0.974 | 1.083 | 0.357 | 0.404 | 0.023 / 0.594 |
| clean test_iid | 128 | 0.152 | 0.073 | 0.229 | 76.8 | 0.963 | 0.975 | 1.079 | 0.456 | 0.501 | 0.021 / 0.609 |
| hard_challenge_valid | 12 | 4.704 | 1.244 | 7.448 | 377.7 | 0.943 | 0.949 | 0.346 | 19.729 | 43.590 | 0.011 / 0.526 |
| hard_challenge_test | 12 | 3.667 | 1.110 | 4.275 | 216.8 | 0.933 | 0.939 | 0.318 | 15.152 | 29.103 | 0.010 / 0.470 |
| all_iid_valid | 140 | 0.514 | 0.160 | 2.187 | 545.8 | 0.961 | 0.972 | 1.020 | 2.017 | 4.105 | 0.023 / 0.590 |
| all_iid_test | 140 | 0.453 | 0.162 | 1.271 | 287.3 | 0.961 | 0.972 | 1.014 | 1.716 | 2.953 | 0.020 / 0.601 |

Hard samples account for 99.45%/96.87% of best all-IID point MSE on
valid/test, and 99.44%/97.02% for final. The hard tail has high average
correlation but a 0.318-0.346 amplitude ratio and very large strong-q/top5
errors. It is a scale-and-tail failure, not an argument to blend hard samples
back into clean-IID training or evaluation.

### Clean Provenance Decomposition

The replacement subgroups are only 12 samples each and are monitoring evidence,
not a separate benchmark. Best sample-first RMSE is 0.116/0.160 K for original
clean valid/test and 0.148/0.081 K for generated replacements. Final values are
0.118/0.158 K and 0.145/0.087 K respectively.

## Final-Probe Summary

All entries are per-probe means over the ten fixed probes. `clean probe` means
P01/P03/P04/P05/P07/P08/P10; P02, P06, and P09 remain explicit challenge
probes. Values are RMSE K unless otherwise noted.

| run/checkpoint | all probe RMSE / corr | clean probe RMSE / corr | P02 RMSE / corr / amp | P06 RMSE / corr / amp | P09 RMSE / corr / amp |
| --- | --- | --- | --- | --- | --- |
| P5_02 best | 0.530 / 0.892 | 0.106 / 0.965 | 2.326 / 0.660 / 1.256 | 1.762 / 0.569 / 2.467 | 0.469 / 0.934 / 0.272 |
| P5_02 final | 0.539 / 0.894 | 0.107 / 0.965 | 2.390 / 0.682 / 1.348 | 1.779 / 0.578 / 2.459 | 0.477 / 0.924 / 0.249 |
| P5_03 best | 0.903 / 0.726 | 0.149 / 0.942 | 3.862 / 0.014 / 1.440 | 3.732 / -0.285 / 5.505 | 0.397 / 0.939 / 0.436 |
| P5_03 final | 0.843 / 0.731 | 0.135 / 0.951 | 3.617 / 0.011 / 1.434 | 3.450 / -0.292 / 5.459 | 0.415 / 0.933 / 0.439 |

## V4P5_03 Fourier Negative Ablation

- It completed at B28 without OOM and with finite gradients.
- Best selection occurred at epoch 124. Its clean-valid sample-first RMSE was
  0.148 K versus 0.119 K for P5_02 best; clean-test was 0.175 K versus
  0.153 K. Point-global clean-valid raw RMSE was 0.207 K versus 0.170 K.
- By epoch 200, normalized validation base MSE worsened from 0.06763 to
  0.06904 (final/best 1.0209), while final train loss was 0.00560. This is a
  train-validation gap and a post-best validation regression, not a reason to
  extend the Fourier run.
- P02 and P06 regress sharply on both checkpoints. P09 has a local RMSE gain
  (best 0.397 K versus P5_02 best 0.469 K), but that isolated gain does not
  offset the clean-IID and hard/final-probe regressions.
- Conclusion: `raw_plus_fourier`, frequency 4 is a valid negative ablation;
  do not carry it into the V5 default path.

## V5 Handoff

The clean baseline is frozen. The hard tail remains a separate research line,
not a clean-IID relabeling or weighting change. Planned V5 questions are:

1. q/target decomposition and a global physical-scale branch;
2. shape-scale decomposition for amplitude failures;
3. a bottom-Dirichlet hard constraint and discrete physical residual metrics;
4. controlled hard-tail curriculum or fine-tune studies after a clean baseline
   is held fixed; and
5. multi-seed evaluation before any broader performance claim.

None of these V5 directions is implemented by this closeout.
