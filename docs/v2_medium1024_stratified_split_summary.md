# Heat3D v2 medium1024 stratified split summary

Scope: split-map proposal only. No arrays were copied and no sample metadata or labels were edited.

Split map:

`configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`

## Split counts

| split | count |
|---|---:|
| train | 704 |
| valid_iid | 104 |
| valid_stress | 88 |
| test_id | 64 |
| test_ood_bc | 24 |
| test_ood_stack | 24 |
| test_ood_combined | 16 |

## Old vs new key categories

| category | old train | old valid | new train | valid_iid | valid_stress |
|---|---:|---:|---:|---:|---:|
| low_power | 1 | 113 | 76 | 12 | 22 |
| diag3 | 72 | 127 | 160 | 35 | 45 |
| high_top_h | 116 | 127 | 177 | 35 | 70 |
| low_k_barrier_or_TIM_variation | 2 | 67 | 59 | 18 | 43 |
| high_contrast_interface_k | 130 | 60 | 118 | 17 | 44 |

## Raw DeltaT distribution

| split | sample_count | raw_deltaT_mean | raw_deltaT_std | raw_deltaT_p95 | low_deltaT_fraction <=0.01K |
|---|---:|---:|---:|---:|---:|
| old train | 768 | 0.02929 | 0.04383 | 0.11001 | 0.390 |
| old valid | 128 | 0.01096 | 0.03205 | 0.05130 | 0.809 |
| new train | 704 | 0.02624 | 0.04320 | 0.10414 | 0.450 |
| valid_iid | 104 | 0.02795 | 0.04639 | 0.11072 | 0.443 |
| valid_stress | 88 | 0.03099 | 0.04671 | 0.11073 | 0.433 |

## Interpretation

`valid_iid` is now much closer to train for the main distribution checks:

- low-power is no longer concentrated in validation;
- diag3 is represented in train and validation;
- high-top-h is represented in train and validation;
- barrier/TIM cases are no longer almost absent from train;
- raw DeltaT mean/p95 and low-DeltaT fraction are close between train and `valid_iid`.

`valid_stress` keeps pressure-test value by concentrating harder cases, especially high-top-h, diag3, high-contrast interface k, and barrier/TIM variation. It should be reported separately from IID validation.

## Default split status

`configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_stratified_seed0.yaml` includes:

`dataset.split_map_path: configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`

The v2 command builder and controlled training runner now use this split map as
the default for `medium1024_gapA_full1024_v2`. Direct runner calls default to the
same stratified map, while configs may still pass `dataset.split_map_path`
explicitly. In split-map mode, `valid_iid` is the primary validation split and
`valid_stress` remains diagnostic-only.
