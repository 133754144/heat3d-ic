# V5 Global FiLM Architecture Smoke

## Result

- Frozen baseline: `V4P5_02_clean_baseline_raw_B28_e600` epoch `405`.
- Mode: read-only checkpoint smoke; no training, checkpoint, data, or graph-topology write occurred.
- V4 disabled-FiLM valid archive replay max error: `0.00549316 K`.
- Identity FiLM max raw-DeltaT replay error (batched): `0.00637273 K`; batch-1: `0.00152054 K` (tolerance `0.02 K`).
- JIT/eager max raw-DeltaT difference: `0.0208245 K` (tolerance `0.03 K`); finite gradient: `True`.
- Gate-1 physics-scale crosscheck max error: `7.21645e-15 K`.

## Global Context Contract

- Context uses only inference-time `coords`, `k`, `q`, BC fields, and control-volume weights; it has no target, residual, prediction, or oracle input.
- The standardizer is fit only on `train=672` and records the frozen feature order/hash.
- FiLM target: `rnodes_processed`; latent dimension remains `96`.
- The gamma/beta output layer is zero initialized, so `h' = (1 + gamma) * h + beta` leaves the processed-rnode latent exactly unchanged at initialization. GPU sparse replay is compared in raw K using the frozen V4 archive tolerance.

## Feature Schema

| index | feature | provenance |
| ---: | --- | --- |
| 0 | `log_s_phys_K` | coords+k_z+q+bottom_mask+top_h+BC_temperature_offset via z_collapsed_1d_operator |
| 1 | `P_operator_W` | q+control_volume+bottom_BC_mask |
| 2 | `log_P_operator_W` | P_operator_W |
| 3 | `q_weighted_local_kz_W_mK` | q+k_z+control_volume |
| 4 | `q_weighted_inverse_kz_mK_W` | q+k_z+control_volume |
| 5 | `q_low_k_overlap_fraction` | q+k_z+control_volume |
| 6 | `source_concentration` | q+control_volume |
| 7 | `source_z_centroid_normalized` | coords+q+control_volume |
| 8 | `source_layer_kz_heterogeneity_cv` | coords+k_z+q+control_volume |
| 9 | `harmonic_kx_W_mK` | k_x+control_volume |
| 10 | `harmonic_ky_W_mK` | k_y+control_volume |
| 11 | `harmonic_kz_W_mK` | k_z+control_volume |
| 12 | `anisotropy_xy_over_z` | harmonic conductivity features |
| 13 | `log_Lx_m` | coords |
| 14 | `log_Ly_m` | coords |
| 15 | `log_Lz_m` | coords |
| 16 | `log_top_area_m2` | coords |
| 17 | `log_top_h_W_m2K` | top_h BC feature |
| 18 | `T_bottom_K` | reference Dirichlet temperature plus bottom BC feature |
| 19 | `T_inf_K` | reference Dirichlet temperature plus top ambient BC feature |
| 20 | `T_inf_minus_T_bottom_K` | top/bottom BC features |
| 21 | `bc_top_cv_fraction` | top BC mask+control_volume |
| 22 | `bc_bottom_cv_fraction` | bottom BC mask+control_volume |
| 23 | `bc_side_cv_fraction` | side BC mask+control_volume |

The JSON contains the exact train-only standardizer and checkpoint partial-load evidence.
