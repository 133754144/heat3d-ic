# Gate 6Q final valid-only closeout

Scope: frozen CPU/NumPy true-RMS metrics on `valid_iid`; only `train` was used for persisted normalization/context checks. No test/hard/sealed access and no training.

The metric inputs are checkpoint-bound prediction NPZ files whose training-time parameter reload audits passed. Direct cross-backend CPU model execution drift is retained in JSON as a diagnostic and is not used to alter the metric fields.

## Formal point-global-best ranking

| rank | model | epoch | point-global % | sample-first % | raw CV RMSE K | shape CV-RMSE | scale log-RMSE | <20% |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | V42 | 257 | 21.936815 | 19.250517 | 0.156347 | 0.143008 | 0.147133 | no |
| 2 | V38 | 231 | 21.944915 | 20.605917 | 0.157328 | 0.148421 | 0.178126 | no |
| 3 | V44 | 329 | 22.060219 | 18.907835 | 0.158943 | 0.140377 | 0.153252 | no |
| 4 | V43 | 276 | 22.450097 | 20.068498 | 0.161387 | 0.143975 | 0.158713 | no |

## Preregistered paired comparisons (point-global-best)

Candidate-minus-baseline deltas; negative error/SSE means improvement.

| comparison | point-global pp | sample-first pp | raw CV K | point SSE K2 | win rate | Q1 SSE | Q2 SSE | Q3 SSE | Q4 SSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V42_minus_V38 | -0.008099 | -1.355400 | -0.000981 | -1.924030 | 0.5781 | 0.805059 | -40.180087 | -53.263788 | 90.714787 |
| V43_minus_V38 | 0.505182 | -0.537419 | 0.004059 | 121.410321 | 0.5000 | 0.163770 | -37.182263 | -54.313531 | 212.742345 |
| V44_minus_V43 | -0.389878 | -1.160663 | -0.002444 | -93.942644 | 0.6250 | -4.186366 | 2.298462 | -44.451674 | -47.603065 |

## Point-SSE concentration

| model | top-5 share | top-10 share |
|---|---:|---:|
| V38 | 0.433070 | 0.599493 |
| V42 | 0.474862 | 0.641589 |
| V43 | 0.492456 | 0.636272 |
| V44 | 0.504536 | 0.652948 |

## Independent contribution conclusions

- **objective**: V42 is the formal point-global leader, but its 0.0081 percentage-point margin over V38 is negligible; the clearer benefit is lower sample-first and scale error. Treat the objective contribution as weakly positive, not decisive.
- **xy_scale_features**: V43 improves sample-first/shape/scale versus V38 but regresses point-global and raw CV RMSE; an independent XY-feature benefit is not established.
- **deepsets**: V44 improves point-global, sample-first, raw CV, shape, scale, background, and hotspot versus V43, but strong-q and top-5 RMSE regress. The latent DeepSets contribution is positive but non-uniform within the V43 lineage, and it does not beat V38/V42 on point-global RMSE.
- **threshold**: No point-global-best checkpoint reaches the frozen <20% valid threshold.
- **next_stage**: Do not advance architecture complexity from this gate; reproduce V42 versus V38 with paired seeds before treating the very small point-global margin as real.

## Artifacts

- Unified JSON: `configs/heat3d_v5/gate6q/gate6q_final_closeout.json`
- Four-checkpoint metrics CSV: `configs/heat3d_v5/gate6q/gate6q_final_checkpoint_metrics.csv`
- Paired sample CSV: `configs/heat3d_v5/gate6q/gate6q_final_paired_samples.csv`
- Quartile/decomposition CSV: `configs/heat3d_v5/gate6q/gate6q_final_attribution.csv`
