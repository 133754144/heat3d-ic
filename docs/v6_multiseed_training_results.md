# V6_03 P1h seed0/1/2 latest training results

Scope: frozen saved `valid_iid` predictions only. No checkpoint inference, training, checkpoint mutation, test, hard, or sealed access.

| seed | host | point-global % | sample-first % | raw RMSE K | shape CV-RMSE | scale log-RMSE | best epoch | final point-global % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | wsl2 | 0.912029 | 0.750343 | 0.390107 | 0.007421 | 0.001608 | 111 | 1.047920 |
| 1 | wsl2 | 0.962142 | 0.710118 | 0.411542 | 0.007043 | 0.001137 | 254 | 1.050489 |
| 2 | devbox | 0.898834 | 0.758224 | 0.384463 | 0.007486 | 0.001311 | 139 | 0.997425 |

## Assessment

- Point-global mean ± sample std: 0.924335% ± 0.033400%; range 0.063308 percentage points.
- Sample-first mean ± sample std: 0.739562% ± 0.025802%.
- Raw RMSE mean ± sample std: 0.395371 ± 0.014286 K.
- Best point-global seed: seed2 at 0.898834%.
- At the point-global-selected checkpoints, seed1 has the best sample-first value (0.710118%) while seed2 has the best point-global/raw values; this is a small shape-scale/aggregation trade-off rather than a uniform winner.
- At the separately frozen sample-first checkpoints, seed0/1/2 are 0.723250%/0.698306%/0.680736%, with seed2 best.
- All three seeds are below the 20% point-global threshold. Seed-to-seed spread is small relative to the threshold and the P1h gain is reproducible on the frozen 1024 shared support.
- Every seed degrades from its point-global checkpoint to e600; checkpoint selection remains necessary.
- The separate volume-representative ladder failure is not overturned: this multi-seed result establishes repeatability only on the canonical P1h operator support.

Remote note: both repositories were clean on `research/v6-p1h-shared-support` at training HEAD `7d30b78`; the configured standalone seed log files were absent, but complete `loss_summary.json`, run config, checkpoints, predictions, and reload audits were present.
