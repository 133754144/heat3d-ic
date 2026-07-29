# V6 final performance closeout

The model, checkpoint, sampling, graph parameters, and reconstruction method remained frozen. The corrected confirmatory holdout was opened after preregistration; hard remained sealed.

The first test command used the legacy ladder for temporary 4096/8192 outputs. They are explicitly excluded; the 16384 attempt stopped before label loading. The formal table below is the corrected, frozen source-aware workflow and did not change any selection.

## Direct timing

| Platform | Mode | B | Nodes | Model-core s | Production s | Evaluation s |
|---|---|---:|---:|---:|---:|---:|
| cpu | cold | 8 | 4096 | 25.992 | 34.450 | 36.924 |
| gpu | cold | 8 | 4096 | 31.552 | 47.492 | 49.524 |
| cpu | cached | 8 | 4096 | 25.956 | 32.537 | 34.917 |
| gpu | cached | 8 | 4096 | 31.669 | 42.689 | 44.489 |
| cpu | persistent | 8 | 4096 | 17.873 | 21.321 | 23.823 |
| gpu | persistent | 8 | 4096 | 10.990 | 16.028 | 17.859 |
| cpu | cold | 8 | 8192 | 37.561 | 45.984 | 48.380 |
| gpu | cold | 8 | 8192 | 30.036 | 46.095 | 47.904 |
| cpu | cached | 8 | 8192 | 37.469 | 43.967 | 46.454 |
| gpu | cached | 8 | 8192 | 30.073 | 41.309 | 43.178 |
| cpu | persistent | 8 | 8192 | 30.048 | 33.356 | 35.864 |
| gpu | persistent | 8 | 8192 | 10.812 | 16.044 | 17.882 |
| cpu | cold | 8 | 16384 | 65.026 | 73.759 | 76.253 |
| gpu | cold | 8 | 16384 | 31.805 | 49.094 | 51.029 |
| cpu | cached | 8 | 16384 | 63.527 | 70.776 | 73.255 |
| gpu | cached | 8 | 16384 | 31.475 | 43.300 | 45.157 |
| cpu | persistent | 8 | 16384 | 55.626 | 59.568 | 62.103 |
| gpu | persistent | 8 | 16384 | 10.891 | 16.768 | 18.650 |
| cpu | cold | 8 | 32768 | 112.161 | 122.675 | 124.916 |
| gpu | cold | 8 | 32768 | 32.662 | 50.975 | 52.979 |
| cpu | cached | 8 | 32768 | 111.091 | 119.552 | 122.030 |
| gpu | cached | 8 | 32768 | 32.399 | 45.578 | 47.501 |
| cpu | persistent | 8 | 32768 | 102.374 | 107.288 | 109.791 |
| gpu | persistent | 8 | 32768 | 11.459 | 18.185 | 20.116 |
| gpu | persistent | 16 | 4096 | 5.470 | 10.596 | 12.290 |
| gpu | persistent | 16 | 8192 | 5.298 | 10.575 | 12.334 |
| gpu | persistent | 16 | 16384 | 5.663 | 11.493 | 13.328 |

## Inference versus 240825-node FVM

All model values below are direct 128-sample cycles. Cold production includes graph build; cached production includes graph-cache load; model-core excludes graph and full-field reconstruction. The FVM comparison is explicitly nonmatched-DOF.

| Query nodes | CPU core/no graph s | CPU cold/cached/persistent production s | GPU B8 core/no graph s | GPU B8 cold/cached/persistent production s | FVM cold/warm s per 128 | GPU persistent speedup cold/warm |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 25.992 | 34.450/32.537/21.321 | 31.552 | 47.492/42.689/16.028 | 343.868/186.528 | 21.45×/11.64× |
| 8192 | 37.561 | 45.984/43.967/33.356 | 30.036 | 46.095/41.309/16.044 | 343.868/186.528 | 21.43×/11.63× |
| 16384 | 65.026 | 73.759/70.776/59.568 | 31.805 | 49.094/43.300/16.768 | 343.868/186.528 | 20.51×/11.12× |
| 32768 | 112.161 | 122.675/119.552/107.288 | 32.662 | 50.975/45.578/18.185 | 343.868/186.528 | 18.91×/10.26× |

## Legal structured-FVM mesh sensitivity

| Mesh | Nodes | Cold mean/median/P95 s | Warm mean/median/P95 s | Full-field RMSE K |
|---|---:|---:|---:|---:|
| coarse | 212097 | 2.331/2.335/2.417 | 1.276/1.280/1.342 | 0.02521 |
| medium | 226233 | 2.528/2.524/2.607 | 1.397/1.385/1.498 | 0.02579 |
| reference | 240825 | 2.686/2.694/2.774 | 1.457/1.473/1.580 | 0.00000 |

## Persistent GPU

| B | Nodes | Production s/128 | sample/s | VRAM GB | cold/warm FVM speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 4096 | 16.028 | 7.986 | 0.973 | 21.45×/11.64× |
| 8 | 8192 | 16.044 | 7.978 | 1.686 | 21.43×/11.63× |
| 8 | 16384 | 16.768 | 7.634 | 3.128 | 20.51×/11.12× |
| 16 | 4096 | 10.596 | 12.080 | 1.483 | 32.45×/17.60× |
| 16 | 8192 | 10.575 | 12.104 | 2.645 | 32.52×/17.64× |
| 16 | 16384 | 11.493 | 11.137 | 5.222 | 29.92×/16.23× |

## Corrected confirmatory holdout

| Nodes | Support point-global | Full point-global | Full RMSE K |
|---:|---:|---:|---:|
| 4096 | 1.4669% | 3.5874% | 1.4550 |
| 8192 | 1.5437% | 3.0976% | 1.2563 |
| 16384 | 1.7704% | 2.8652% | 1.1620 |

The confirmatory table is descriptive only. It did not change the frozen 4096/8192/16384 roles. 32768 is excluded; hard remains sealed.

## Frozen decision

- 4096 remains the default/hotspot-oriented mode.
- 8192 remains the balanced full-field mode.
- 16384 remains the maximum full-field accuracy mode.
- 32768 remains experimental and was not included in the confirmatory table.
- The decision was fixed from valid_iid before holdout opening; the corrected confirmatory holdout is descriptive only.
