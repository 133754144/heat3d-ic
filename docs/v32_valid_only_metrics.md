# V32 valid-only metrics

- Scope: `valid_iid` only (128 samples, 1024 nodes/sample).
- Training commit: `fcdb01d`.
- Evaluator commit: `370ed5cb661da5809d7d34e40cbf49011592d023`.
- Frozen formula source: `9c2c6f04f87a0c958c50d9ac9947dcbc655d0a51`.
- Log integrity: declared log is absent; e600 completion is supported by contiguous loss history and the final checkpoint, but log completeness cannot be verified.
- No test, hard, or sealed-IID role was accessed.

| checkpoint | epoch | point-global % | sample-first % | raw CV RMSE K | amplitude | correlation | legacy MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| point_global_best | 474 | 22.408387 | 21.034804 | 0.160067 | 1.001418 | 0.981542 | 0.03268148 |
| legacy_best | 474 | 22.408368 | 21.034539 | 0.160067 | 1.001416 | 0.981543 | 0.03268142 |
| sample_first_best | 366 | 22.978049 | 20.662153 | 0.163956 | 0.986231 | 0.981625 | 0.03436424 |
| final | 600 | 22.627739 | 21.024261 | 0.161803 | 0.991993 | 0.981296 | 0.03332444 |

## Remaining frozen V5 metrics

| checkpoint | hotspot K | top-5 K | strong-q K | low-ΔT bias K | low-ΔT RMSE K | low-ΔT over-ratio | shape CV-RMSE | scale log-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| point_global_best | 0.302944 | 0.467499 | 0.366795 | 0.004501 | 0.018058 | 0.412114 | 0.145697 | 0.197482 |
| legacy_best | 0.302947 | 0.467526 | 0.366808 | 0.004501 | 0.018058 | 0.412248 | 0.145696 | 0.197483 |
| sample_first_best | 0.305601 | 0.478933 | 0.371650 | 0.004946 | 0.018605 | 0.422124 | 0.145141 | 0.193614 |
| final | 0.305893 | 0.476248 | 0.373203 | 0.004812 | 0.017847 | 0.415793 | 0.146520 | 0.197681 |

The saved sample-first checkpoint used `valid_native_joint_relative_rmse` with ordinary raw RMSE as a tie-break only on exact equality. The correct CV metric above is post-hoc diagnostic evidence; no checkpoint was reselected.
