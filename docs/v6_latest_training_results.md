# V6 latest completed training results

Scope: saved `valid_iid` predictions only. No test role, checkpoint inference,
training, or checkpoint mutation was performed by this collector.

## Primary checkpoint comparison

| config | dataset | host | epoch | point-global % | sample-first % | raw RMSE K | final point-global % |
|---|---|---|---:|---:|---:|---:|---:|
| V6_01_V4best | heat3d_v6_p1g_geometry_deconfounded1024_v0 | wsl2 | 407 | 13.842562 | 10.232678 | 5.821767 | 16.905952 |
| V6_02_V5best | heat3d_v6_p1g_geometry_deconfounded1024_v0 | devbox | 406 | 2.305728 | 1.461580 | 0.969720 | 2.356095 |
| V6_03_V5best_P1h | heat3d_v6_p1h_shared_support1024_v0 | wsl2 | 111 | 0.912029 | 0.750343 | 0.390107 | 1.047920 |
| V6_04_V5best_P1h_DualAttention | heat3d_v6_p1h_shared_support1024_v0 | devbox | 111 | 0.906373 | 0.763042 | 0.387688 | 1.059358 |

## Primary-checkpoint diagnostics

| config | amp ratio | correlation | hotspot K | strong-q K | low-DeltaT RMSE K | shape CV-RMSE | scale log-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6_01_V4best | 0.987140 | 0.879691 | 6.881616 | 8.078297 | 5.341135 | 0.043267 | 0.137181 |
| V6_02_V5best | 0.998672 | 0.979361 | 1.894153 | 2.487113 | 0.449278 | 0.014347 | 0.004180 |
| V6_03_V5best_P1h | 0.999346 | 0.992498 | 0.966818 | 1.165834 | 0.218119 | 0.007421 | 0.001608 |
| V6_04_V5best_P1h_DualAttention | 0.999290 | 0.992460 | 0.930583 | 1.012886 | 0.224785 | 0.007510 | 0.001720 |

## Diagnosis

- All four runs completed e600. V6_01's training and exports completed, but its original strict max-absolute reload audit raised after export; the preserved post-export recovery audit passed without retraining or artifact mutation.
- At the point-global checkpoint, V6_04−V6_03 is -0.005656 percentage points for point-global and -0.002419 K for raw RMSE, but +0.012699 percentage points for sample-first (positive is worse).
- V6_04 also changes shape/scale error by +0.000089/+0.000112; the point-global gain is therefore small and not a uniform shape-scale gain.
- V6_04−V6_03 paired sample-relative win rate: 40.62%.
- Mean/median sample-relative delta: 0.012699/0.011683 percentage points.
- Total point-SSE delta: -246.639834 K² (negative favors V6_04).
- Under each run's sample-first-selected checkpoint, V6_04 is 0.706416% versus 0.723250% for V6_03; checkpoint selection materially changes the apparent conclusion.
- P1g and P1h contain identical physical cases but use different operator supports; V6_02→V6_03 is therefore a representation comparison, not an identical-point metric replay.
- V6_03→V6_04 is the clean shape-attention ablation because dataset, support, training contract, and seed are identical.

## Metric formulas

- point-global: `sqrt(sum(error^2) / sum(true_DeltaT^2))`
- sample-first: mean per-sample `RMS(error) / RMS(true_DeltaT)`
- raw RMSE: equal-weight RMSE over 128×1024 valid operator points
- hotspot/top5: true-DeltaT per-sample top 10% / top 5%
- strong-q: per-sample positive-q top decile
- shape/scale: RMS-normalized field error and log RMS-scale error
