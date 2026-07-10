# V4 Physical-Hard Clean-Split Audit

Read this file only for candidate1024 physical-hard classification or
clean/hard split decisions.

## Scope And Definitions

The audit covers all 1024 samples under the formal split map: train `768`,
valid_iid `128`, and test_iid `128`. The hard cohort is exactly
`qc_class=physical_hard_keep`; non-hard combines `clean_keep` and
`review_hold`.

`low-k volume` is the point fraction where
`min(kx, ky, kz) <= 5 W/m/K`. Values below are sample medians. q total uses the
solver control-volume power audit.

## Integrity And Invariants

- Counts are train `121 hard / 647 non-hard`, valid `12 / 116`, test `12 / 116`,
  and all `145 / 879`.
- All 1024 samples have `solver_called=true`, `solver_status=solved`, and
  `array_preflight_passed=true`.
- Both cohorts have zero source/side boundary violations and zero power
  deposited on the boundary.
- Geometry is identical: extents are `0.01 x 0.01 x 0.002 m`, aspect ratio is
  `5`.
- BC layout is identical: top `256`, bottom `256`, side `120`, interior `392`;
  features are `is_top/is_bottom/is_side/is_interior`.
- Boundary temperatures are identical: top ambient `300 K`, bottom fixed
  `300 K`.

These checks reject schema corruption, failed solves, geometry changes, and BC
layout changes as explanations for the hard cohort.

## Physical Distributions

| split | cohort | n | top_h | q max W/m3 | q total W | q+ frac | k min | k contrast | low-k frac | peak DeltaT K | mean abs DeltaT K | p95 DeltaT K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | non-hard | 647 | 868.7 | 1.11e8 | 0.847 | 0.053 | 2.631 | 90.5 | 0.069 | 1.966 | 0.171 | 0.712 |
| train | physical-hard | 121 | 874.1 | 2.59e8 | 1.662 | 0.049 | 0.775 | 241.3 | 0.371 | 35.181 | 1.610 | 7.552 |
| valid_iid | non-hard | 116 | 803.4 | 1.19e8 | 0.906 | 0.051 | 2.633 | 90.7 | 0.085 | 1.748 | 0.149 | 0.598 |
| valid_iid | physical-hard | 12 | 1443.6 | 2.29e8 | 1.508 | 0.062 | 0.516 | 344.5 | 0.329 | 32.512 | 1.848 | 5.209 |
| test_iid | non-hard | 116 | 840.2 | 7.45e7 | 0.825 | 0.058 | 2.705 | 100.9 | 0.099 | 1.759 | 0.167 | 0.692 |
| test_iid | physical-hard | 12 | 1205.9 | 4.21e8 | 1.595 | 0.060 | 0.866 | 150.6 | 0.348 | 42.530 | 1.651 | 6.080 |
| all | non-hard | 879 | 857.9 | 1.10e8 | 0.851 | 0.055 | 2.659 | 92.0 | 0.075 | 1.929 | 0.168 | 0.705 |
| all | physical-hard | 145 | 930.7 | 2.59e8 | 1.649 | 0.049 | 0.774 | 236.0 | 0.359 | 35.225 | 1.624 | 7.011 |

Hard/non-hard median ratios are:

| split | peak DeltaT | mean abs DeltaT | q max | q total | k min | k contrast | low-k volume |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 17.89x | 9.42x | 2.34x | 1.96x | 0.29x | 2.67x | 5.35x |
| valid_iid | 18.60x | 12.42x | 1.92x | 1.66x | 0.20x | 3.80x | 3.85x |
| test_iid | 24.18x | 9.89x | 5.65x | 1.93x | 0.32x | 1.49x | 3.53x |
| all | 18.26x | 9.68x | 2.35x | 1.94x | 0.29x | 2.56x | 4.78x |

The hard cohort is therefore a low-conductivity, high-contrast, higher-power
thermal tail. q-positive volume itself is not larger; the difference is source
intensity/power combined with substantially more low-k volume.

## Physical-Keep Reasons

Across all hard samples, reason tags are `low_k_trapped_hotspot=97`,
`multi_source_or_high_power_bottleneck=73`, and `weak_cooling=50`; tags may
overlap. Exclusive high-DeltaT triage counts are:

- `physical_low_k_enclosed_compact_hotspot=97`
- `physical_multi_source_or_high_power_bottleneck=28`
- `physical_weak_cooling_high_deltaT=20`

Train/valid/test contain the same reason families. The evaluation hard samples
are represented in training, so they are a target-distribution hard tail and
an OOD-style stress cohort relative to clean IID, not a wholly unseen or
abnormal data class.

## Clean-Nohard Gate

| criterion | observed | threshold | pass |
| --- | ---: | ---: | --- |
| hard/non-hard median peak DeltaT | 18.26x | >10x | yes |
| hard/non-hard median mean abs DeltaT | 9.68x | >5x | yes |
| hard share of V4P4 all-IID MSE | 80.05%-95.65% | >80% | yes |

The combined valid_iid + test_iid contributions are:

| config | best hard MSE % | final hard MSE % |
| --- | ---: | ---: |
| V4P4_01 | 94.94 | 95.65 |
| V4P4_02 | 84.35 | 87.49 |
| V4P4_03 | 82.25 | 85.30 |
| V4P4_04 | 80.05 | 80.32 |

All three requested conditions pass. A fixed `clean_nohard` view is justified,
but it must not replace the original formal result. Report three stable views:

- `clean_iid`: non-hard samples only;
- `hard_challenge`: `physical_hard_keep` samples only;
- `all_iid`: the unchanged formal valid/test split.

Do not replace or relabel the 145 hard samples in `candidate1024_v0`. If a
clean training dataset is later required, create a versioned dataset/split and
retain these samples as the fixed hard challenge set.
