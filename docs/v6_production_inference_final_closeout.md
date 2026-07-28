# V6 production inference final closeout

Only `valid_iid` was evaluated. No training, checkpoint mutation, test, or hard-role access occurred.

## Three-seed accuracy

| Nodes | Support point-global | Full-field RMSE K | Full-field relative | Peak RMSE K | Source RMSE K | Layer/interface RMSE K |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1.4240±0.1251% | 1.4966±0.0593 | 3.6908±0.1462% | 1.4743±0.0950 | 0.9053±0.0627 | 0.8321/0.6454 |
| 8192 | 1.4867±0.1473% | 1.3001±0.0890 | 3.2062±0.2196% | 1.6560±0.2349 | 0.9709±0.0724 | 0.7835/0.6392 |
| 16384 | 1.6354±0.0705% | 1.2236±0.0934 | 3.0174±0.2304% | 1.5742±0.1928 | 1.1207±0.0518 | 0.7993/0.6931 |

The support and full-field CSV/JSON also retain sample-first, shape/scale, top/bottom, layer-drop, and sampling-floor fields.

## Sparse graph equivalence

| Nodes | Chunked build s | Sparse KD-tree build s | Speedup | Hash |
|---:|---:|---:|---:|:---:|
| 8192 | 1.9169 | 0.1343 | 14.27× | metadata+graph exact |
| 16384 | 5.1759 | 0.2739 | 18.89× | metadata+graph exact |

The training runner smoke reused one shared-support metadata/graph build instead of eight per-sample builds, with 0 K forward difference and no optimizer update.

## Production timing and solver comparison

| Nodes | CPU forward/e2e s | GPU forward/e2e s | CPU/GPU sample/s | Cold solver speedup CPU/GPU | Warm solver speedup CPU/GPU |
|---:|---:|---:|---:|---:|---:|
| 1024 | 9.73/20.23 | 16.74/29.21 | 6.33/4.38 | 17.00×/11.77× | 9.22×/6.38× |
| 2048 | 17.69/30.70 | 29.08/48.27 | 4.17/2.65 | 11.20×/7.12× | 6.08×/3.86× |
| 4096 | 27.06/37.75 | 32.01/53.89 | 3.39/2.38 | 9.11×/6.38× | 4.94×/3.46× |
| 8192 | 38.98/50.48 | 30.64/51.08 | 2.54/2.51 | 6.81×/6.73× | 3.70×/3.65× |
| 16384 | 66.78/86.79 | 31.79/53.43 | 1.47/2.40 | 3.96×/6.44× | 2.15×/3.49× |
| 32768 | 121.20/149.11 | 32.90/55.97 | 0.86/2.29 | 2.31×/6.14× | 1.25×/3.33× |

FVM cold mesh+assembly+solve: mean 2.686 s, median 2.694 s, P95 2.774 s per sample. Warm solve: mean 1.457 s, median 1.473 s, P95 1.580 s per sample.

At 4096, the selected CPU B8 persistent warm workflow is 18.201 s/128 samples; GPU B16 is 5.361 s/128 samples. Stage-separated input, graph cache, anchor/query forward, scale, label read, reconstruction, metric, and serialization timings are frozen in `v6_production_stage_timing.csv`.

## Decision

4096 remains the general default. 8192 is frozen as the optional full-field mode because it materially lowers reconstructed full-field RMSE. 16384 remains the high-accuracy production limit; 32768 remains experimental seed0-only.

All solver comparisons are explicitly nonmatched-DOF: FVM uses 240825 nodes.
