# V4P4_01-04 Split-Aware Diagnostics

Read this file only for V4P4_01-04 result comparison or clean/hard evaluation
decisions.

## Scope And Artifacts

- This is read-only evaluation on completed runs. No training or tmux session
  was started.
- Source host: `wsl2`; source revision: `research/v4@88e3e1a`.
- Existing `valid_iid` predictions and final-probe metrics were reused.
- Missing `test_iid` predictions were exported from all eight best/final
  checkpoints to ignored
  `output/heat3d_v4_offline_diagnostics/V4P4_0*/<checkpoint>_test_iid/`.
- All four runs have `loss_summary.json`, `run_config.json`, best/final
  checkpoints, 128-sample valid predictions, and 10-sample best/final
  final-probe metrics.

Metrics marked RMSE, MAE, corr, cosine, amplitude ratio, top5, and strong-q are
computed per sample and then averaged. `rel_rmse_v4_pct` retains the runner
definition: point-global raw DeltaT RMSE divided by point-global mean absolute
true DeltaT. Top5 is sample-wise true DeltaT at or above p95; strong-q is
sample-wise positive q at or above its p90.

## All-IID Results

| config | ckpt | split | n | RMSE K | MAE K | rel % | corr | cosine | amp | top5 RMSE K | strong-q RMSE K | hard MSE % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P4_01 | best | valid_iid | 128 | 0.435 | 0.176 | 373.5 | 0.848 | 0.896 | 1.519 | 1.453 | 2.668 | 98.0 |
| V4P4_01 | best | test_iid | 128 | 0.405 | 0.192 | 171.8 | 0.857 | 0.905 | 1.391 | 1.260 | 1.521 | 83.8 |
| V4P4_01 | final | valid_iid | 128 | 0.425 | 0.169 | 378.1 | 0.860 | 0.904 | 1.481 | 1.437 | 2.637 | 98.2 |
| V4P4_01 | final | test_iid | 128 | 0.396 | 0.185 | 174.9 | 0.866 | 0.911 | 1.336 | 1.245 | 1.581 | 86.3 |
| V4P4_02 | best | valid_iid | 128 | 0.774 | 0.394 | 409.5 | 0.614 | 0.734 | 3.057 | 2.073 | 2.950 | 90.5 |
| V4P4_02 | best | test_iid | 128 | 0.748 | 0.423 | 243.1 | 0.614 | 0.739 | 2.196 | 1.935 | 1.565 | 70.8 |
| V4P4_02 | final | valid_iid | 128 | 0.719 | 0.336 | 432.9 | 0.627 | 0.738 | 2.925 | 1.781 | 2.843 | 92.6 |
| V4P4_02 | final | test_iid | 128 | 0.697 | 0.360 | 235.3 | 0.629 | 0.751 | 2.107 | 1.709 | 1.659 | 74.1 |
| V4P4_03 | best | valid_iid | 128 | 0.803 | 0.407 | 416.3 | 0.650 | 0.754 | 3.332 | 2.128 | 3.229 | 89.2 |
| V4P4_03 | best | test_iid | 128 | 0.786 | 0.434 | 242.3 | 0.654 | 0.763 | 2.549 | 2.081 | 1.962 | 66.3 |
| V4P4_03 | final | valid_iid | 128 | 0.745 | 0.353 | 424.8 | 0.608 | 0.725 | 3.131 | 1.736 | 2.699 | 91.0 |
| V4P4_03 | final | test_iid | 128 | 0.727 | 0.377 | 236.5 | 0.610 | 0.736 | 2.230 | 1.723 | 1.650 | 71.0 |
| V4P4_04 | best | valid_iid | 128 | 0.882 | 0.470 | 422.1 | 0.603 | 0.728 | 3.456 | 2.287 | 3.557 | 87.2 |
| V4P4_04 | best | test_iid | 128 | 0.837 | 0.486 | 251.6 | 0.612 | 0.739 | 2.590 | 2.155 | 2.089 | 64.4 |
| V4P4_04 | final | valid_iid | 128 | 0.866 | 0.442 | 432.3 | 0.573 | 0.701 | 3.865 | 1.949 | 3.014 | 87.2 |
| V4P4_04 | final | test_iid | 128 | 0.831 | 0.460 | 254.2 | 0.592 | 0.720 | 2.849 | 1.911 | 2.093 | 64.8 |

## Clean And Hard Cohorts

| config | ckpt | split | cohort | n | RMSE K | MAE K | rel % | corr | cosine | amp | top5 K | strong-q K |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P4_01 | best | valid_iid | clean_nohard | 116 | 0.188 | 0.108 | 91.4 | 0.837 | 0.890 | 1.576 | 0.472 | 0.415 |
| V4P4_01 | best | valid_iid | hard_challenge | 12 | 2.820 | 0.831 | 252.3 | 0.948 | 0.954 | 0.963 | 10.934 | 24.454 |
| V4P4_01 | best | test_iid | clean_nohard | 116 | 0.243 | 0.134 | 108.9 | 0.847 | 0.899 | 1.434 | 0.628 | 0.482 |
| V4P4_01 | best | test_iid | hard_challenge | 12 | 1.970 | 0.750 | 121.8 | 0.952 | 0.957 | 0.971 | 7.361 | 11.567 |
| V4P4_01 | final | valid_iid | clean_nohard | 116 | 0.178 | 0.103 | 86.9 | 0.850 | 0.899 | 1.540 | 0.437 | 0.370 |
| V4P4_01 | final | valid_iid | hard_challenge | 12 | 2.814 | 0.815 | 255.7 | 0.951 | 0.956 | 0.909 | 11.101 | 24.550 |
| V4P4_01 | final | test_iid | clean_nohard | 116 | 0.230 | 0.128 | 101.9 | 0.857 | 0.906 | 1.380 | 0.582 | 0.470 |
| V4P4_01 | final | test_iid | hard_challenge | 12 | 2.008 | 0.744 | 125.9 | 0.952 | 0.958 | 0.913 | 7.654 | 12.322 |
| V4P4_02 | best | valid_iid | clean_nohard | 116 | 0.478 | 0.297 | 218.0 | 0.588 | 0.719 | 3.255 | 0.922 | 0.785 |
| V4P4_02 | best | valid_iid | hard_challenge | 12 | 3.634 | 1.323 | 265.8 | 0.867 | 0.878 | 1.138 | 13.199 | 23.879 |
| V4P4_02 | best | test_iid | clean_nohard | 116 | 0.538 | 0.344 | 206.6 | 0.589 | 0.725 | 2.325 | 1.111 | 0.865 |
| V4P4_02 | best | test_iid | hard_challenge | 12 | 2.777 | 1.187 | 158.5 | 0.858 | 0.872 | 0.954 | 9.899 | 8.324 |
| V4P4_02 | final | valid_iid | clean_nohard | 116 | 0.433 | 0.247 | 203.7 | 0.600 | 0.721 | 3.134 | 0.664 | 0.412 |
| V4P4_02 | final | valid_iid | hard_challenge | 12 | 3.483 | 1.192 | 284.1 | 0.888 | 0.900 | 0.904 | 12.576 | 26.344 |
| V4P4_02 | final | test_iid | clean_nohard | 116 | 0.488 | 0.286 | 188.4 | 0.603 | 0.736 | 2.240 | 0.875 | 0.519 |
| V4P4_02 | final | test_iid | hard_challenge | 12 | 2.724 | 1.078 | 157.0 | 0.875 | 0.895 | 0.816 | 9.773 | 12.682 |
| V4P4_03 | best | valid_iid | clean_nohard | 116 | 0.521 | 0.317 | 236.2 | 0.627 | 0.741 | 3.571 | 1.041 | 1.016 |
| V4P4_03 | best | valid_iid | hard_challenge | 12 | 3.528 | 1.280 | 268.2 | 0.874 | 0.887 | 1.026 | 12.638 | 24.617 |
| V4P4_03 | best | test_iid | clean_nohard | 116 | 0.588 | 0.361 | 221.3 | 0.633 | 0.752 | 2.714 | 1.313 | 1.177 |
| V4P4_03 | best | test_iid | hard_challenge | 12 | 2.707 | 1.143 | 152.9 | 0.853 | 0.872 | 0.953 | 9.501 | 9.550 |
| V4P4_03 | final | valid_iid | clean_nohard | 116 | 0.468 | 0.265 | 219.7 | 0.579 | 0.707 | 3.353 | 0.661 | 0.420 |
| V4P4_03 | final | valid_iid | hard_challenge | 12 | 3.417 | 1.199 | 276.5 | 0.885 | 0.897 | 0.983 | 12.130 | 24.731 |
| V4P4_03 | final | test_iid | clean_nohard | 116 | 0.524 | 0.305 | 200.6 | 0.582 | 0.720 | 2.367 | 0.897 | 0.547 |
| V4P4_03 | final | test_iid | hard_challenge | 12 | 2.696 | 1.072 | 154.4 | 0.875 | 0.894 | 0.907 | 9.710 | 12.306 |
| V4P4_04 | best | valid_iid | clean_nohard | 116 | 0.587 | 0.377 | 260.8 | 0.577 | 0.713 | 3.698 | 1.106 | 1.172 |
| V4P4_04 | best | valid_iid | hard_challenge | 12 | 3.735 | 1.375 | 268.9 | 0.856 | 0.871 | 1.124 | 13.709 | 26.616 |
| V4P4_04 | best | test_iid | clean_nohard | 116 | 0.634 | 0.413 | 236.2 | 0.588 | 0.726 | 2.753 | 1.348 | 1.322 |
| V4P4_04 | best | test_iid | hard_challenge | 12 | 2.800 | 1.192 | 156.4 | 0.850 | 0.865 | 1.012 | 9.949 | 9.504 |
| V4P4_04 | final | valid_iid | clean_nohard | 116 | 0.579 | 0.348 | 266.8 | 0.544 | 0.683 | 4.160 | 0.841 | 0.604 |
| V4P4_04 | final | valid_iid | hard_challenge | 12 | 3.640 | 1.344 | 275.4 | 0.857 | 0.874 | 1.017 | 12.661 | 26.313 |
| V4P4_04 | final | test_iid | clean_nohard | 116 | 0.632 | 0.389 | 237.3 | 0.564 | 0.703 | 3.054 | 1.091 | 0.870 |
| V4P4_04 | final | test_iid | hard_challenge | 12 | 2.759 | 1.149 | 158.6 | 0.861 | 0.883 | 0.866 | 9.839 | 13.912 |

## Final-Probe Summary

| config | ckpt | mean RMSE K | mean corr | P02 RMSE K | P03 RMSE K | P09 RMSE K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V4P4_01 | best | 0.421 | 0.827 | 1.519 | 0.592 | 0.474 |
| V4P4_01 | final | 0.409 | 0.831 | 1.472 | 0.610 | 0.476 |
| V4P4_02 | best | 0.418 | 0.725 | 0.911 | 0.796 | 0.418 |
| V4P4_02 | final | 0.485 | 0.703 | 1.749 | 0.731 | 0.430 |
| V4P4_03 | best | 0.761 | 0.676 | 2.857 | 0.841 | 0.392 |
| V4P4_03 | final | 0.599 | 0.629 | 2.342 | 0.760 | 0.444 |
| V4P4_04 | best | 0.478 | 0.755 | 0.635 | 1.057 | 0.377 |
| V4P4_04 | final | 0.512 | 0.642 | 1.536 | 0.870 | 0.366 |

## Assessment

- `V4P4_01-final` is the strongest P4 entry on formal splits and mean
  final-probe RMSE. It does not beat `V4P3_19-final`: its sample-first
  valid/test RMSE is `0.425/0.396 K` versus `0.410/0.372 K`, and its valid/test
  rel RMSE is `378.1%/174.9%` versus `366.3%/172.1%`.
- The hard cohorts retain high shape correlation, especially for V4P4_01
  (`0.951-0.952` final), but have very large top5 and strong-q RMSE. The
  remaining failure is concentrated high-amplitude hotspot error rather than
  loss of field shape.
- `V4P4_04` does not validate hard-sample weighting. Its hard RMSE is worse
  than V4P4_01 on both splits, and its clean cohort degrades sharply. Its lower
  all-IID hard MSE share is caused by increased clean error, not by solving the
  hard samples.
- V4P4_04-best improves P02 to `0.635 K`, but worsens P03 to `1.057 K` and does
  not improve mean probe RMSE. This is a local tradeoff, not an overall gain.
- Current formal split ranking remains `V4P3_19-final`; among V4P4_01-04,
  retain V4P4_01-final only as the least-regressive P4 control.
