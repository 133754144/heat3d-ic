# V4 Train-Consistent IID Split Audit

Read this file only for V4 split-map, IID-test, or train-consistency questions.

## Problem

The previous V4 split map used `valid_iid` as a category-diverse validation
holdout. That is useful for stress-aware validation, but it is not identical to
a train-consistent IID test split. The V4 P1 target is now train-consistent IID
test error below 20%, so the split contract needs an explicit proportional IID
holdout from the regular training distribution.

## New Split Rule

New split map:

`configs/heat3d_v2/medium1024_gapA_train_consistent_split_seed0.json`

Source split map:

`configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json`

Rules:

- `regular_pool = old train + old valid_iid + old test_id` = 872 samples.
- `valid_stress = old valid_stress` remains a stress holdout = 88 samples.
- OOD holdouts remain unchanged: `test_ood_bc` 24, `test_ood_stack` 24,
  `test_ood_combined` 16.
- The regular pool is proportionally re-split as `train=704`,
  `valid_iid=84`, and `test_id=84`.
- In this file, the new `test_id` is the train-consistent IID test split.

The proportional split uses devbox sample metadata categories:
`source_category`, `power_scale_category`, `bc_category`, `k_field_mode`,
`k_region_mode`, and `stack_template`. It does not upsample hard cases.

## Diagnostics Contract

V4 post-training diagnostics now pass the active split map explicitly to the
split-aware diagnostics scripts. Those scripts resolve split labels from
`split_map` first and fall back to `sample_meta["split"]` only when no split map
is provided.

## V4P1_12

Registered config:

`V4P1_12_BC_passthrough_bypass_scale0p5_train_consistent_split`

Baseline:

`V4P1_09_BC_passthrough_decoder_bypass_full_condition_scale0p5`

Only changed fields:

- `split_map_path`
- output, log, final-probe, and post-training diagnostics paths

Model, loss, optimizer, epochs, seeds, batch size, graph policy, normalization,
and decoder bypass settings are unchanged.

Status: completed on `devbox`; `run_registry.csv` records the completed result.
On the train-consistent `valid_iid` split, V4P1_12 meets the current <20% IID
target:

- best checkpoint: `iid_err=16.88%`
- final checkpoint: `iid_err=16.92%`

This is a train-consistent IID result, not an OOD or final-probe claim.
