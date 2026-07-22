# Heat3D V5 phase closeout

Status: `completed_research_phase_threshold_unmet`. The frozen V5 candidate remains V42 point-global best e257. No training, checkpoint reselection, model mutation, hard-role access, or sealed-IID access occurred during closeout.

## Frozen valid ranking

| model | epoch | point-global | sample-first | raw CV RMSE K | shape CV-RMSE | scale log-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V42 | 257 | 21.936815% | 19.250517% | 0.156347 | 0.143008 | 0.147133 |
| V38 | 231 | 21.944915% | 20.605917% | 0.157328 | 0.148421 | 0.178126 |
| V44 | 329 | 22.060219% | 18.907835% | 0.158943 | 0.140377 | 0.153252 |
| V45 | 358 | 23.125696% | 19.993253% | 0.166421 | 0.144355 | 0.167092 |
| V46 | 484 | 23.615429% | 19.797011% | 0.169723 | 0.140332 | 0.169015 |

## Final V42 e257 test_iid

The checkpoint was frozen before test access. Test was not used for selection or tuning.

| metric | valid_iid | test_iid | test-valid |
|---|---:|---:|---:|
| point_global_relative_rmse_pct | 21.9368154 | 23.2496162 | +1.31280089 |
| sample_first_cv_relative_rmse_pct | 19.2505172 | 19.4976639 | +0.247146653 |
| raw_cv_weighted_rmse_K | 0.15634711 | 0.201884595 | +0.0455374857 |
| amplitude_ratio | 1.01955479 | 1.02155681 | +0.00200202243 |
| spatial_correlation | 0.981572048 | 0.982404351 | +0.000832302981 |
| hotspot_cv_weighted_rmse_K | 0.282622606 | 0.32736936 | +0.0447467536 |
| top5_cv_weighted_rmse_K | 0.42645112 | 0.416359517 | -0.0100916032 |
| strong_q_cv_weighted_rmse_K | 0.338530487 | 0.316915321 | -0.0216151667 |
| low_deltaT_background_bias_K | 0.0060001116 | 0.00636367922 | +0.000363567611 |
| low_deltaT_background_rmse_K | 0.0197277753 | 0.0215784203 | +0.00185064498 |
| low_deltaT_background_over_ratio | 0.40299458 | 0.416891894 | +0.0138973132 |
| shape_cv_rmse | 0.143007725 | 0.140209751 | -0.00279797419 |
| scale_log_rmse | 0.147132997 | 0.154422177 | +0.0072891804 |
| legacy_normalized_valid_base_mse | 0.031320426 | 0.0549215587 | +0.0236011327 |

## Batch-1 inference timing

Device `cuda:0`, backend `gpu`, float32 parameters, 10 warmups, synchronized per sample; checkpoint load and file I/O excluded.

| path | mean ms | median ms | P90 ms | N |
|---|---:|---:|---:|---:|
| model forward | 342.084 | 344.911 | 355.428 | 128 |
| graph/preprocess + forward | 438.097 | 441.093 | 459.833 | 128 |

## Q4 root cause and V6 hypothesis

All pairwise top-5 difficult sets are identical; top-10 intersections range 8–10 samples. Q4 contributes V38 79.0%, V42 82.6%, V44 84.5%, V45 83.2%, V46 84.1% of point SSE.

The identical top-5 and high top-10 overlap, Q4-dominated SSE, and repeatable positive q-weighted inverse-kz association indicate an energy/path-resistance failure shared across architectures. Weak 24D train-nearest-neighbor, top-h, and stack-count correlations do not support generic coverage distance or simple categorical stack sparsity as the primary cause.

Unique V6 hypothesis: V6 should test a stack-aware source-to-sink thermal-resistance representation that preserves layer/path structure for the interaction of source power, through-plane inverse conductivity, vertical distance, and terminal BCs; it should not begin with more generic capacity, flat XY features, or undirected coverage expansion.

## Integration

`integration/v5-core@3a85d53a` contains only stable runner/model/metric/context/pooling implementation and synthetic tests. Research tables, generated YAML, closeout utilities, datasets and run artifacts are excluded. No merge was executed.
