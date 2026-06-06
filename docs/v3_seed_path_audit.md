# Heat3D v3 Seed-Path Audit

## Scope

This is a compact diagnostic note for latent96 B96 graph-policy runs. It does not
claim formal benchmark quality.

Actions completed on devbox:

- Stopped the nearest_repair seed2 e400 process with normal `kill` on PID 525674.
- Preserved all existing logs and partial output.
- Verified no remaining Heat3D training process after termination.
- Completed final and best diagnostics for:
  - `latent96_s6_mlp2_B96_base_mse_warmup_cosine_discrete_radius_e400_seed0`
  - `latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_e400_seed1`

Diagnostics present for both target runs: baseline comparison, error bins,
condition diagnostics, run summary, and field-shape diagnostics for final and
best predictions.

## nearest_repair seed0 vs seed1

| run | initial valid_iid loss | final valid_iid loss | best valid_iid loss | valid_stress final loss | best epoch | final/best | valid_iid raw DeltaT RMSE | valid_stress raw DeltaT RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 | 1.143566 | 0.036087 | 0.036037 | 0.054690 | 398 | 1.0014 | 0.008206 | 0.010102 |
| seed1 | 1.212440 | 0.658122 | 0.658122 | 0.658614 | 400 | 1.0000 | 0.035043 | 0.035057 |

Selected valid_iid loss curve:

| epoch | seed0 | seed1 |
| ---: | ---: | ---: |
| 0 | 1.143566 | 1.212440 |
| 1 | 1.017707 | 1.135381 |
| 5 | 0.613730 | 1.052838 |
| 10 | 0.540756 | 1.037915 |
| 50 | 0.294715 | 0.927891 |
| 100 | 0.245821 | 0.828494 |
| 200 | 0.118068 | 0.712472 |
| 300 | 0.042922 | 0.665712 |
| 380 | 0.036090 | 0.658261 |
| 400 | 0.036087 | 0.658122 |

Both best epochs are in the final 5% of training (seed0 epoch 398, seed1 epoch
400). This is a possible undertraining or LR-decay timing signal, not a
standalone conclusion.

## Diagnostics Highlights

| run | bin0 signed bias | bin0 over_ratio | bin0 RMSE | field corr | amplitude ratio | field variance ratio | p95 error | peak abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 best | 0.001801 | 0.8087 | 0.003052 | 0.9782 | 0.9860 | 0.9677 | 0.012721 | 0.023481 |
| seed1 best | 0.000465 | 0.5476 | 0.002801 | 0.7963 | 0.2886 | 0.3994 | 0.036705 | 0.264672 |

Condition diagnostics top background-bias groups:

| run | top groups |
| --- | --- |
| seed0 best | `bc_category=very_low_top_h_candidate`, `source_category=multi_block_power`, `source_category=low_power_near_zero_background_cases`, `k_region_mode=high_contrast_interface_k`, `bc_category=high_top_h` |
| seed1 best | `bc_category=held_out_top_h_candidate`, `split=test_ood_bc_candidate`, `bc_category=very_low_top_h_candidate`, `split=test_ood_combined_candidate`, `bc_category=very_high_top_h_candidate` |

Seed1 has lower bin0 bias but much worse global field shape and RMSE. The failure
mode is therefore not simply background overprediction.

## Batch Composition Audit

Audit target:
`configs/heat3d_v2/frozen_v1_e400_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml`

Train split: 704 samples, B96, epoch-1 shuffled order with batch seed 0 produced
8 train batches: seven B96 batches and one B32 tail batch.

Observed concentration:

| batch | dominant composition |
| ---: | --- |
| 1 | 86/96 `interposer_like_4_layer`; mixed high/low top-h; diagonal/high-contrast k |
| 2 | 95/96 `baseline_4_layer`; 96/96 `nominal_top_h`; mostly `layerwise_isotropic_k` |
| 3 | 96/96 `baseline_4_layer`; 96/96 `nominal_top_h`; blockwise/layerwise isotropic k |
| 4 | 75/96 `interposer_like_4_layer`, 21/96 `compact_3_layer`; 96/96 `low_top_h` |
| 5 | 85/96 `compact_3_layer`; mostly `nominal_top_h`; blockwise/interposer k |
| 6 | 96/96 `dual_active_4_layer`; 96/96 `high_top_h`; high-contrast/low-k modes |
| 7 | 96/96 `compact_3_layer`; 96/96 `low_top_h`; 96/96 interposer-equivalent k |
| 8 | 32/32 `dual_active_4_layer`; 32/32 `high_top_h`; low-k TIM variation |

The current batch construction groups by metadata shape before chunking. Even
with epoch-level batch shuffling, individual B96 batches can be structurally
concentrated. Before this task, `optimizer.seed` affected both model init and
batch order, so seed comparisons were confounded by initialization and
mini-batch order.

## Seed Decoupling

Runner/config changes add optional:

- `optimizer.model_seed`
- `optimizer.batch_order_seed`
- `optimizer.graph_seed`

Compatibility rules:

- If `model_seed` is missing, model initialization uses `optimizer.seed`.
- If `batch_order_seed` is missing, train batch order uses 0.
- If `graph_seed` is missing, graph metadata uses 0.
- `optimizer.seed` is retained as the legacy seed field.

New e20 configs isolate model seed and batch-order seed while holding
`graph_seed=0`. The new e400 config keeps model_seed=1 while resetting
batch_order_seed=0 and graph_seed=0.

## Conclusion

Nearest_repair seed0 is strong, while nearest_repair seed1 is much worse under
the old coupled seed path. Because batch composition is structurally
concentrated and `optimizer.seed` previously drove both model init and batch
order, the next diagnostic should compare decoupled e20 probes before launching
another e400 run.
