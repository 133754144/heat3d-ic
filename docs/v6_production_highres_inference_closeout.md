# V6 production high-resolution inference closeout

No training, checkpoint mutation, test, or hard-role access occurred.

## Frozen workflow

`1024 anchor forward -> anchor-derived Global Context/scale -> N-node source-aware forward -> anchor-scale reconstruction`.

The production default is 4096 nodes and the frozen production maximum is 16384. 32768 passed as an experimental extension.

## Resolution results

| Nodes | CPU point-global | CPU full RMSE K | CPU e2e s | GPU e2e s |
|---:|---:|---:|---:|---:|
| 1024 | 0.9025% | 1.8564 | 37.22 | 69.94 |
| 2048 | 1.1917% | 1.5623 | 68.32 | 132.38 |
| 4096 | 1.3359% | 1.4290 | 74.16 | 135.31 |
| 8192 | 1.4189% | 1.2252 | 77.32 | 135.42 |
| 16384 | 1.6443% | 1.1346 | 96.74 | 138.54 |
| 32768 | 1.8863% | 1.0811 | 161.29 | 137.67 |

## Graph optimization

The exact sparse KD-tree backend plus query-only regional-node reduction is accepted. At 4096, point-global is 1.3359% and full-field RMSE is 1.4290 K (0.9902x the unoptimized baseline). CPU cached and uncached predictions are exactly identical. GPU graph hashes are exact and repeat predictions stay within 0.005 K maximum / 0.001 K RMSE.

At 4096, batch 8 reduced CPU end-to-end time to 45.18 s and GPU end-to-end time to 51.02 s for 128 samples.

The prior 32768 timeout was caused by dense N-by-R graph-distance materialization. Sparse search completed the full 128-sample evaluation without changing the checkpoint.

## Solver comparison

FVM cold mean is 2.6865 s/sample and warm mean is 1.4573 s/sample. The comparison is nonmatched-DOF; speedups are workflow timing comparisons, not equal-system-size algorithmic complexity claims.
