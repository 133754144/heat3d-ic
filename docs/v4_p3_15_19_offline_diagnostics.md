# V4P3_15-19 Split-Aware Offline Diagnostics

Read this file only for V4P3_15-19 result ranking, CSV backfill, or next-step training decisions.

## Scope

- No new training or tmux launch was started.
- V4P3_15/16/18 were audited on `wsl2`; V4P3_17/19 were audited on `devbox`.
- Existing `valid_iid` predictions were reused; `test_iid` predictions were
  exported from existing best/final checkpoints under ignored
  `output/heat3d_v4_offline_diagnostics/<config>/<checkpoint>_test_iid/`.
- `post_training_diagnostics` remains skipped by design for
  `prediction_split=valid_iid`; split-aware offline diagnostics replace the old
  all-sample post-diagnostics requirement for this review.

## Artifact Check

| config | host | loss | run_config | best ckpt | final ckpt | valid preds | final-probe json | split offline complete | CSV status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V4P3_15 | wsl2 | yes | yes | yes | yes | yes | yes | yes | reviewed_completed |
| V4P3_16 | wsl2 | yes | yes | yes | yes | yes | yes | yes | reviewed_completed |
| V4P3_17 | devbox | yes | yes | yes | yes | yes | yes | yes | reviewed_completed |
| V4P3_18 | wsl2 | yes | yes | yes | yes | yes | yes | yes | reviewed_completed |
| V4P3_19 | devbox | yes | yes | yes | yes | yes | yes | yes | reviewed_completed |

## Training Scalars

| config | best epoch | best valid_base_mse | best raw RMSE K | best rel % | final valid_base_mse | final raw RMSE K | final rel % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P3_15 | 131 | 0.384 | 1.749 | 424.6 | 0.417 | 1.823 | 442.6 |
| V4P3_16 | 108 | 0.390 | 1.763 | 427.9 | 0.477 | 1.951 | 473.7 |
| V4P3_17 | 100 | 0.364 | 1.704 | 413.7 | 0.364 | 1.704 | 413.7 |
| V4P3_18 | 114 | 0.309 | 1.571 | 381.3 | 0.362 | 1.699 | 412.4 |
| V4P3_19 | 21 | 0.276 | 1.483 | 359.9 | 0.285 | 1.509 | 366.3 |

## Split-Aware Offline Metrics

RMSE/MAE are recovered-temperature K metrics on the prediction npz keys
actually present in each split. `le0.05 over` is the overprediction ratio where
true DeltaT <= 0.05 K.

| config | ckpt | split | RMSE K | MAE K | corr | cosine | amp | top-k | le0.05 over | strong-q RMSE | p95 abs K |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P3_15 | best | valid_iid | 0.571 | 0.242 | 0.794 | 0.861 | 1.690 | 0.472 | 0.740 | 3.632 | 0.840 |
| V4P3_15 | best | test_iid | 0.578 | 0.275 | 0.786 | 0.858 | 1.736 | 0.478 | 0.742 | 2.763 | 1.073 |
| V4P3_15 | final | valid_iid | 0.522 | 0.192 | 0.844 | 0.895 | 1.549 | 0.452 | 0.658 | 3.577 | 0.723 |
| V4P3_15 | final | test_iid | 0.470 | 0.208 | 0.827 | 0.890 | 1.465 | 0.469 | 0.678 | 2.366 | 0.876 |
| V4P3_16 | best | valid_iid | 0.618 | 0.270 | 0.781 | 0.850 | 2.022 | 0.436 | 0.591 | 3.831 | 0.927 |
| V4P3_16 | best | test_iid | 0.569 | 0.288 | 0.773 | 0.846 | 1.992 | 0.484 | 0.574 | 2.418 | 1.093 |
| V4P3_16 | final | valid_iid | 0.539 | 0.191 | 0.852 | 0.900 | 1.588 | 0.430 | 0.691 | 3.882 | 0.721 |
| V4P3_16 | final | test_iid | 0.457 | 0.202 | 0.838 | 0.897 | 1.447 | 0.483 | 0.694 | 2.327 | 0.875 |
| V4P3_17 | best | valid_iid | 0.501 | 0.193 | 0.833 | 0.889 | 1.542 | 0.448 | 0.678 | 3.343 | 0.711 |
| V4P3_17 | best | test_iid | 0.464 | 0.213 | 0.821 | 0.885 | 1.504 | 0.455 | 0.666 | 2.256 | 0.869 |
| V4P3_17 | final | valid_iid | 0.501 | 0.193 | 0.833 | 0.889 | 1.542 | 0.448 | 0.678 | 3.343 | 0.711 |
| V4P3_17 | final | test_iid | 0.464 | 0.213 | 0.821 | 0.885 | 1.504 | 0.455 | 0.666 | 2.257 | 0.869 |
| V4P3_18 | best | valid_iid | 0.685 | 0.330 | 0.666 | 0.768 | 2.668 | 0.403 | 0.617 | 2.726 | 1.300 |
| V4P3_18 | best | test_iid | 0.689 | 0.364 | 0.668 | 0.775 | 1.946 | 0.389 | 0.593 | 1.687 | 1.452 |
| V4P3_18 | final | valid_iid | 0.644 | 0.284 | 0.694 | 0.783 | 2.722 | 0.412 | 0.767 | 2.821 | 1.190 |
| V4P3_18 | final | test_iid | 0.623 | 0.298 | 0.698 | 0.798 | 1.848 | 0.437 | 0.768 | 1.705 | 1.294 |
| V4P3_19 | best | valid_iid | 0.423 | 0.165 | 0.874 | 0.913 | 1.494 | 0.434 | 0.564 | 2.525 | 0.635 |
| V4P3_19 | best | test_iid | 0.387 | 0.178 | 0.879 | 0.920 | 1.449 | 0.497 | 0.583 | 1.627 | 0.741 |
| V4P3_19 | final | valid_iid | 0.410 | 0.154 | 0.891 | 0.924 | 1.273 | 0.486 | 0.614 | 2.525 | 0.616 |
| V4P3_19 | final | test_iid | 0.372 | 0.165 | 0.891 | 0.928 | 1.271 | 0.519 | 0.621 | 1.658 | 0.702 |

## Final-Probe Summary

| config | ckpt | RMSE K | relRMSE | Tmax err K | shape corr | scale ratio | P02 RMSE | P03 RMSE | P09 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4P3_15 | best | 0.280 | 0.692 | 1.868 | 0.839 | 1.549 | 0.432 | 0.672 | 0.263 |
| V4P3_15 | final | 0.226 | 0.514 | 0.890 | 0.901 | 1.211 | 0.403 | 0.645 | 0.228 |
| V4P3_16 | best | 0.387 | 0.952 | 3.019 | 0.833 | 1.925 | 0.662 | 0.763 | 0.289 |
| V4P3_16 | final | 0.216 | 0.488 | 1.020 | 0.908 | 1.229 | 0.337 | 0.647 | 0.263 |
| V4P3_17 | best | 0.220 | 0.510 | 0.284 | 0.889 | 1.113 | 0.326 | 0.582 | 0.330 |
| V4P3_17 | final | 0.220 | 0.510 | 0.284 | 0.889 | 1.113 | 0.326 | 0.582 | 0.330 |
| V4P3_18 | best | 0.404 | 0.938 | 0.667 | 0.762 | 1.319 | 1.153 | 0.699 | 0.389 |
| V4P3_18 | final | 0.419 | 0.929 | 0.521 | 0.798 | 1.202 | 1.465 | 0.634 | 0.418 |
| V4P3_19 | best | 0.432 | 0.896 | 0.167 | 0.788 | 1.161 | 1.850 | 0.467 | 0.502 |
| V4P3_19 | final | 0.410 | 0.840 | 0.032 | 0.791 | 1.141 | 1.739 | 0.467 | 0.511 |

## Conclusions

- Same-split offline ranking favors V4P3_19: final checkpoint has the lowest
  test_iid RMSE (`0.372 K`), highest test_iid corr (`0.891`), best cosine
  (`0.928`), closest amplitude ratio (`1.271`), and lowest low-DeltaT
  overprediction among these runs (`le0.05 over=0.621`, still high).
- Final-probe ranking does not favor V4P3_19. V4P3_17 is the best final-probe
  compromise among 15-19 (`RMSE=0.220 K`, `relRMSE=0.510`,
  `shape_corr=0.889`), while V4P3_19 has worse probe RMSE/relRMSE and a severe
  P02 failure (`P02 RMSE=1.739 K` final).
- V4P3_15 and V4P3_16 overfit by training scalar, but their final checkpoints
  improve split-aware RMSE and final-probe versus best checkpoints. This means
  `valid_base_mse` alone is still not a reliable final ranking metric for P3
  candidate behavior.
- V4P3_18 has strong training scalar (`best valid_base_mse=0.309`) but poor
  split-aware shape (`corr around 0.67-0.70`) and poor final-probe
  (`relRMSE around 0.93`), so the logk/signedlog transform plus extra loss
  weights is not validated here.
- All split diagnostics show persistent low-DeltaT overprediction and
  strong-q/hotspot weakness. V4P3_19 reduces background/low-DeltaT error most,
  but final-probe P02 indicates generalization risk under disconnected
  conduction paths.

## CSV Backfill

- `configs/heat3d_v4/run_registry.csv` was backfilled only after loss summary,
  best/final checkpoints, valid_iid predictions, test_iid offline predictions,
  and final-probe JSON were present.
- Rows V4P3_15-19 are marked `reviewed_completed`, not scalar-only
  `completed`.
- `result_post_training_diagnostics_status=non_all_prediction_split` records
  that the old all-sample diagnostics path was intentionally skipped.

## Next Direction

Prioritize one research question: preserve V4P3_19 split-level gains while
fixing final-probe P02 generalization. The next change should target
evaluation/selection or feature/loss behavior for disconnected conduction paths
and low-DeltaT overprediction; do not choose by `valid_base_mse` alone.
