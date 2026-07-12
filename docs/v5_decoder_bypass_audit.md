# V5 Frozen V4 Decoder-Bypass Audit

## Scope

- Frozen baseline: `V4P5_02_clean_baseline_raw_B28_e600` epoch `405`.
- Replayed only clean `train`, `valid_iid`, and `test_iid`; no hard role, training, checkpoint write, data write, or model modification occurred.
- The disabled-bypass counterfactual is exact residual subtraction in normalized DeltaT space from the captured V4 `decoder_bypass_residual` intermediate.

## Input Variation Audit

| feature | classification | node-varying samples | invariant samples | max within-sample range | V5 treatment |
| --- | --- | ---: | ---: | ---: | --- |
| k_x | node_varying | 928 | 0 | 428.1 | retain local bypass |
| k_y | node_varying | 928 | 0 | 787.43 | retain local bypass |
| k_z | node_varying | 928 | 0 | 757.29 | retain local bypass |
| q | node_varying | 928 | 0 | 1.8808e+09 | retain local bypass |
| is_top | node_varying | 928 | 0 | 1 | retain local bypass |
| is_bottom | node_varying | 928 | 0 | 1 | retain local bypass |
| is_side | node_varying | 928 | 0 | 1 | retain local bypass |
| is_interior | node_varying | 928 | 0 | 1 | retain local bypass |
| top_h | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| top_T_inf_minus_T_ref | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| bottom_T_fixed_minus_T_ref | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| log_Lx | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| log_Ly | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| log_Lz | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| log_Lx_over_Lz | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |
| log_Ly_over_Lz | sample_global_broadcast | 0 | 928 | 0 | move to Global FiLM only |

## Architecture Decision

- Decision: `retain_local_bypass_and_remove_global_broadcast_duplicates`.
- Rationale: At least one frozen full_condition channel is node-varying in clean scenes; the V5 local bypass must remain a separately switchable module.
- Local-capable bypass inputs: `k_x, k_y, k_z, q, is_top, is_bottom, is_side, is_interior`.
- Sample-global broadcast inputs to remove from local bypass: `top_h, top_T_inf_minus_T_ref, bottom_T_fixed_minus_T_ref, log_Lx, log_Ly, log_Lz, log_Lx_over_Lz, log_Ly_over_Lz`.

## Frozen Full Bypass Versus Residual-Disabled Counterfactual

Positive error reduction means retaining the frozen full bypass lowers that error; spatial-correlation gain is full minus disabled.

| role | full sample-first CV-rel % | disabled sample-first CV-rel % | bypass reduction pp | full raw CV-RMSE K | disabled raw CV-RMSE K | bypass hotspot reduction K | bypass shape-CV reduction | bypass scale-log reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13.153 | 91.152 | 77.999 | 0.042725 | 0.13021 | 0.038398 | 0.2122 | 0.51053 |
| valid_iid | 28.526 | 87.307 | 58.781 | 0.18651 | 0.22228 | 0.017365 | 0.16321 | 0.35256 |
| test_iid | 29.939 | 85.705 | 55.766 | 0.25551 | 0.27847 | 0.021403 | 0.15566 | 0.33791 |
| clean_all | 17.589 | 89.87 | 72.282 | 0.12298 | 0.17259 | 0.033153 | 0.19764 | 0.45419 |

The JSON contains the full required V5 metric suite for both variants, including point-global relative RMSE, background bias/RMSE/over-ratio, top-five and strong-q RMSE, amplitude ratio, and spatial correlation.
