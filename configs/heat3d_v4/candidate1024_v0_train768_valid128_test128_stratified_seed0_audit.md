# V4 P3c Candidate1024 Formal Split Audit

Read this file only for candidate1024 split protocol, registry handoff, or split-distribution audit.

## Summary

- dataset_id: `heat3d_v4_p3c_candidate1024_v0`
- split_map: `configs/heat3d_v4/candidate1024_v0_train768_valid128_test128_stratified_seed0.json`
- seed: `0`
- assignment_method: `deterministic_stratified_combination_largest_remainder_v0`
- counts: train 768, valid_iid 128, test_iid 128
- stratify keys: `qc_class, DeltaT_bin, q_family, cooling_regime, diag3_policy, k_mode, high_deltaT_triage`
- legacy smoke-only split map: `configs/heat3d_v4/candidate1024_v0_test_as_valid_iid_split_map.json`

The published manifest `test` split is no longer used as the formal validation split. It remains only in the legacy smoke bridge file.

## Distribution Bias

| key | max abs split fraction delta vs total |
| --- | ---: |
| `qc_class` | 0.056641 |
| `DeltaT_bin` | 0.065430 |
| `q_family` | 0.031250 |
| `cooling_regime` | 0.044922 |
| `diag3_policy` | 0.041016 |
| `k_mode` | 0.024414 |
| `high_deltaT_triage` | 0.047852 |

## `qc_class`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean_keep` | 766 | 561 | 103 | 102 | 0.056641 |
| `physical_hard_keep` | 145 | 121 | 12 | 12 | 0.047852 |
| `review_hold` | 113 | 86 | 13 | 14 | 0.008789 |

## `DeltaT_bin`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard` | 313 | 234 | 39 | 40 | 0.006836 |
| `low` | 64 | 51 | 7 | 6 | 0.015625 |
| `nominal` | 389 | 276 | 57 | 56 | 0.065430 |
| `reject_high` | 145 | 121 | 12 | 12 | 0.047852 |
| `review_high` | 113 | 86 | 13 | 14 | 0.008789 |

## `q_family`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `compact_hotspot_q_density` | 148 | 109 | 19 | 20 | 0.011719 |
| `dual_z_q_density` | 145 | 111 | 18 | 16 | 0.016602 |
| `elongated_q_density` | 148 | 107 | 21 | 20 | 0.019531 |
| `multi_block_q_density` | 150 | 113 | 19 | 18 | 0.005859 |
| `tsv_adjacent_q_density` | 144 | 115 | 14 | 15 | 0.031250 |
| `weak_background_hotspot_q_density` | 139 | 105 | 16 | 18 | 0.010742 |
| `weak_background_q_density` | 150 | 108 | 21 | 21 | 0.017578 |

## `cooling_regime`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nominal_package` | 337 | 255 | 40 | 42 | 0.016602 |
| `strong_forced_or_effective_heatsink` | 338 | 245 | 48 | 45 | 0.044922 |
| `weak_effective_air` | 349 | 268 | 40 | 41 | 0.028320 |

## `diag3_policy`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hbm_like_strong` | 162 | 128 | 15 | 19 | 0.041016 |
| `mild` | 637 | 469 | 84 | 84 | 0.034180 |
| `scalar` | 225 | 171 | 29 | 25 | 0.024414 |

## `k_mode`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `diag3` | 799 | 597 | 99 | 103 | 0.024414 |
| `scalar` | 225 | 171 | 29 | 25 | 0.024414 |

## `high_deltaT_triage`

| category | total | train | valid_iid | test_iid | max abs delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `not_high_deltaT` | 879 | 647 | 116 | 116 | 0.047852 |
| `physical_low_k_enclosed_compact_hotspot` | 97 | 80 | 8 | 9 | 0.032227 |
| `physical_multi_source_or_high_power_bottleneck` | 28 | 23 | 3 | 2 | 0.011719 |
| `physical_weak_cooling_high_deltaT` | 20 | 18 | 1 | 1 | 0.011719 |
