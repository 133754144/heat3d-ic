# Heat3D v2 train-valid split diagnostics

Scope: read-only split diagnostics; not a formal benchmark.

Subset: `data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2`

## Sample Counts

| split | sample_count |
|---|---:|
| train | 768 |
| valid | 128 |

### Source Category

| value | train | valid |
|---|---:|---:|
| broad_block_power | 112 | 0 |
| centered_single_hotspot | 110 | 1 |
| dual_active_layer_hotspots | 112 | 0 |
| edge_or_corner_hotspot | 111 | 0 |
| high_dynamic_range_power_cases | 1 | 0 |
| low_power_near_zero_background_cases | 1 | 113 |
| multi_block_power | 98 | 14 |
| shifted_single_hotspot | 111 | 0 |
| two_hotspots_same_layer | 112 | 0 |

### Power Scale Category

| value | train | valid |
|---|---:|---:|
| high_dynamic_range | 1 | 0 |
| low_power | 1 | 113 |
| nominal | 766 | 15 |

### BC Category

| value | train | valid |
|---|---:|---:|
| high_top_h | 116 | 127 |
| low_top_h | 326 | 0 |
| nominal_top_h | 326 | 1 |

### K Mode

| value | train | valid |
|---|---:|---:|
| diag3 | 72 | 127 |
| iso1 | 696 | 1 |

### K Region Mode

| value | train | valid |
|---|---:|---:|
| blockwise_isotropic_k | 158 | 0 |
| diagonal_anisotropic_k | 160 | 0 |
| high_contrast_interface_k | 130 | 60 |
| interposer_equivalent_k | 160 | 0 |
| layerwise_isotropic_k | 158 | 1 |
| low_k_barrier_or_TIM_variation | 2 | 67 |

### Integrated Power W

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.197433 | 0.0725547 | 0.0241091 | 0.19574 | 0.327053 | 0.41583 |
| valid | 128 | 0.0504197 | 0.0836289 | 0.0122004 | 0.0205873 | 0.287473 | 0.343397 |

### Q Field Max

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 54640983.30 | 28745472.95 | 11035820.16 | 50560345.68 | 116281970.30 | 158479325.82 |
| valid | 128 | 9200854.74 | 13323900.68 | 1715292.97 | 4588860.24 | 46852389.59 | 59723622.44 |

### Raw DeltaT Node Distribution

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 294912 | 0.0292894 | 0.0438295 | 0 | 0.0158524 | 0.110007 | 0.915282 |
| valid | 49152 | 0.0109554 | 0.0320469 | 0 | 0.00214542 | 0.0512955 | 0.85762 |

### Sample Raw DeltaT Mean

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.0292894 | 0.0176419 | 0.0047248 | 0.0240603 | 0.0676056 | 0.111496 |
| valid | 128 | 0.0109554 | 0.0191488 | 0.00175902 | 0.00426537 | 0.0620668 | 0.0988237 |

### Sample Raw DeltaT Std

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.0370719 | 0.0153449 | 0.0086768 | 0.0330405 | 0.0714305 | 0.116745 |
| valid | 128 | 0.0142927 | 0.0213552 | 0.00300805 | 0.00681595 | 0.0699945 | 0.108857 |

### Sample Raw DeltaT Max

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.330702 | 0.145721 | 0.121803 | 0.289094 | 0.615641 | 0.915282 |
| valid | 128 | 0.11296 | 0.165272 | 0.020875 | 0.0558986 | 0.504508 | 0.85762 |

### Sample Raw DeltaT P95

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.0882235 | 0.0414218 | 0.0109814 | 0.0790046 | 0.170142 | 0.333294 |
| valid | 128 | 0.0379973 | 0.057434 | 0.00800707 | 0.0182014 | 0.19064 | 0.311114 |

### Hotspot DeltaT P95

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.0882235 | 0.0414218 | 0.0109814 | 0.0790046 | 0.170142 | 0.333294 |
| valid | 128 | 0.0379973 | 0.057434 | 0.00800707 | 0.0182014 | 0.19064 | 0.311114 |

### Hotspot DeltaT Max

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.330702 | 0.145721 | 0.121803 | 0.289094 | 0.615641 | 0.915282 |
| valid | 128 | 0.11296 | 0.165272 | 0.020875 | 0.0558986 | 0.504508 | 0.85762 |

### Hotspot Fraction >= 0.05 K

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.168521 | 0.1542 | 0.0078125 | 0.122396 | 0.562891 | 0.731771 |
| valid | 128 | 0.0510661 | 0.137699 | 0 | 0.00260417 | 0.453906 | 0.580729 |

### Hotspot Fraction >= 0.10 K

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.0609572 | 0.0892881 | 0.00260417 | 0.03125 | 0.239583 | 0.591146 |
| valid | 128 | 0.0233154 | 0.0689266 | 0 | 0 | 0.213151 | 0.361979 |

### Low DeltaT Fraction <= 0.01 K

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.390035 | 0.146322 | 0.166667 | 0.373698 | 0.651042 | 0.9375 |
| valid | 128 | 0.809184 | 0.215566 | 0.179688 | 0.885417 | 0.936589 | 0.963542 |

### Low DeltaT Fraction <= 0.02 K

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.569519 | 0.188917 | 0.166667 | 0.595052 | 0.848047 | 0.981771 |
| valid | 128 | 0.883911 | 0.208129 | 0.236979 | 0.955729 | 0.989583 | 0.994792 |

### Low DeltaT Fraction <= 0.05 K

| split | count | mean | std | min | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 768 | 0.831479 | 0.1542 | 0.268229 | 0.877604 | 0.971354 | 0.992188 |
| valid | 128 | 0.948934 | 0.137699 | 0.419271 | 0.997396 | 1.0000 | 1.0000 |

## Available Metadata Fields

### sample_meta.json

`barrier_k_category`, `bc_category`, `boundary_params`, `boundary_types`, `description`, `generation_config`, `interfaces`, `k_contrast_category`, `k_field_mode`, `k_region_mode`, `power_scale_category`, `sample_id`, `schema_version`, `source_diagnostics`, `source_pattern_tag`, `split`, `stack`, `stage`, `subset_name`, `units`, `validation`

### metadata.json

`active_source_volume_discrete_m3`, `barrier_k_category`, `bc_category`, `bc_value_variant`, `bottom_T_fixed_K`, `bottom_dirichlet_error`, `convergence_flag`, `dataset_name`, `diagnostics_scope`, `generation_variant_version`, `integrated_power_W`, `integrated_q_power_relative_error`, `k_contrast_category`, `k_field_mode`, `k_region_mode`, `k_scale_factor`, `k_variant_id`, `manifest_path`, `manifest_version`, `metadata_schema_version`, `pattern_seed`, `power_scale_category`, `q_geometry_variant`, `q_scale_factor`, `residual_norm`, `sample_id`, `sample_plan`, `source_center_shift`, `source_missed`, `source_pattern_tag`, `source_size_scale`, `split`, `stack_template`, `temperature_max_K`, `temperature_min_K`, `top_ambient_temperature_K`, `top_h_W_m2K`, `top_h_value`, `variant_id`

### label_meta.json

`assembly`, `bottom_dirichlet_error`, `bottom_dirichlet_tolerance_K`, `convergence_flag`, `discretization_type`, `duplicate_merge`, `energy_balance_status`, `interface_status`, `label_role`, `not_formal_benchmark`, `not_high_fidelity`, `not_high_fidelity_solver`, `not_model_performance_evidence`, `not_ood_generalization_evidence`, `not_publication_ready_dataset`, `pde_residual_status`, `q_policy`, `residual_norm`, `residual_tolerance`, `sample_id`, `side_adiabatic_status`, `solver_name`, `solver_role`, `solver_version`, `source_assignment`, `source_diagnostics`, `supported_k_mode`, `top_robin_status`, `warnings`

## Notes

- `raw_deltaT` is computed as `temperature.npy - bottom_T_fixed_K` when available, falling back to 300 K.
- Hotspot and low-DeltaT fractions are simple node-level threshold summaries for split comparison.
- This report is intended to identify train/valid distribution mismatch candidates, not to rank model performance.
