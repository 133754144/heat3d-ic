# Heat3D v1 Medium1024 Gap-A Generation Plan

## Scope

`medium1024_gapA` is a generation-ready research diagnostic candidate. It moves
one step beyond the planned medium1024 roadmap by implementing only low-risk
extensions that target issues observed in medium256. It is not a formal
benchmark, not a publication-ready dataset, and not evidence of solved OOD
generalization.

The full 1024-sample dataset has not been generated in this worktree. The
required local step is a small 16/32-sample smoke under ignored `data/`.

## Motivation

Medium256 multi-seed diagnostics show:

- overall metrics beat `zero_delta` across seeds 0/1/2;
- seed sensitivity is visible;
- `bin_0` low-DeltaT background overprediction remains;
- `bin_3` and `bin_4` high-temperature regions remain prone to
  underprediction;
- OOD BC and OOD stack candidate mean-field behavior is not stable enough.

Gap-A targets these issues without changing model structure or adding physical
PDE/BC residual losses.

## Implemented Gap-A Modes

### Source Patterns

- `low_power_near_zero_background_cases`: low-amplitude nonzero sources,
  intended to stress low-DeltaT background calibration without creating trivial
  all-zero labels.
- `high_dynamic_range_power_cases`: strong compact hotspot plus weak extended
  background, intended to stress high-bin and hotspot fidelity.

Both modes keep `q_policy=fixed_density` and `source_assignment=volume_fraction`.
Sample metadata records `power_scale_category`.

### k-Region Modes

- `high_contrast_interface_k`: moderate adjacent-layer conductivity contrast,
  with active-region lateral contrast where applicable.
- `low_k_barrier_or_TIM_variation`: lower TIM/interposer/barrier-like
  conductivity while avoiding extreme values that would intentionally stress
  solver conditioning.

Sample metadata records `k_contrast_category=high_contrast` or
`barrier_k_category=low_k` when applicable.

### BC Categories

- `very_low_top_h_candidate`
- `very_high_top_h_candidate`

These remain inside the existing V1 BC model: top Robin, bottom Dirichlet, and
adiabatic sides. Only `top_h` changes. Bottom Dirichlet is unchanged.

The Gap-A candidate split policy keeps these held-out top-h categories out of
`train`, `valid`, and `test_id`; they only appear in
`test_ood_bc_candidate` or `test_ood_combined_candidate`.

## Still Planned-Only Modes

The following roadmap modes remain intentionally unimplemented in Gap-A:

- `random_sparse_hotspots`
- `elongated_strip_power`
- `ring_or_annular_like_power`
- `checkerboard_or_patchy_power`
- `mixed_blockwise_diag_anisotropic_k`
- `equivalent_interconnect_region_variants`
- `locally_randomized_k_blocks`
- new deep / bridge / five-layer stack templates
- optional bottom-temperature variants

They should not appear in the Gap-A manifest coverage targets.

## Generation Strategy

The staged strategy is:

1. 16-sample local smoke covering all six Gap-A modes.
2. 32/64-sample partial smoke if the first smoke is clean.
3. 64/128-sample pilot only after reviewing label ranges and diagnostics.
4. Full 1024 generation only after the manifest and generator behavior are
   accepted.

The Gap-A manifest uses a deterministic sample-generation plan rather than an
explicit 1024-entry sample list. The generator materializes sample plans from
split counts and coverage counts, while keeping the first 16 samples as smoke
probes that cover the new source, k, and BC modes.

## Pilot Coverage And Checker Fixes

The first SSH-side 128-sample Gap-A pilot showed that generation and labels can
run, with label diagnostics passing, but it exposed tooling issues:

- `sample-limit=128` selected the first materialized samples, so after the first
  16 probe samples the pilot was dominated by train/base conditions;
- downstream coverage scripts expected `metadata.json`, but the generator only
  wrote `sample_meta.json` and `label_meta.json`;
- the medium256 checker could not be reused because it assumes a fixed
  256-sample manifest with explicit `samples` entries.

The fix is tooling-only:

- every generated sample now writes `metadata.json` with direct coverage fields
  such as `split`, `source_pattern_tag`, `k_region_mode`, `k_field_mode`,
  `stack_template`, `bc_category`, `power_scale_category`, and solver/source
  diagnostics;
- Gap-A `--sample-limit` uses balanced deterministic selection when no explicit
  `--sample-ids` are requested;
- `sample-limit=16` still returns the probe set covering all six Gap-A modes;
- `sample-limit=128` now targets the manifest split ratio:
  `train=96`, `valid=16`, `test_id=8`, `test_ood_bc_candidate=3`,
  `test_ood_stack_candidate=3`, `test_ood_combined_candidate=2`;
- `scripts/check_heat3d_v1_physics_label_medium1024_gapA_subset.py` checks
  generated subsets directly and does not depend on `manifest.samples`.

The next recommended step is to rerun the 128-sample pilot with the fixed
selection and subset checker. Full 1024 generation should still wait until the
fixed 128 pilot is reviewed; a 256-sample pilot is a reasonable intermediate
step if the 128-sample coverage looks clean.

## Full1024 Loader And Diversity Follow-Up

The subsequent SSH-side `medium1024_gapA` full1024 generation, generated-subset
checker, and label diagnostics passed. That result only validates the generation
and label file chain; it is still diagnostic and not a formal benchmark.

A short training smoke then exposed a loader compatibility gap:
`Heat3DV1MetadataDataset` rejected
`physics_label_medium1024_gapA_generation_candidate` because the supervised V1
stage allowlist had not been updated for Gap-A. The fix is to add this exact
stage to the V1 supervised/native supervised loader path without allowing
arbitrary stages.

The full1024 pilot also motivates a separate diversity diagnostic. Passing label
diagnostics does not prove that the dataset has enough condition diversity:
coarse combinations can repeat, generated arrays can duplicate, and repeated
`q_field`, `k_field`, or `temperature` hashes can hide behind valid metadata.
`scripts/analyze_heat3d_v1_medium1024_gapA_diversity.py` therefore reports:

- metadata counters for split/source/k/stack/BC categories;
- coarse condition-combo repetition;
- array-hash diversity for `q_field`, `k_field`, and `temperature`;
- per-combo unique hash counts and `T_max` / `q_max` ranges;
- conservative diagnostic flags for training-smoke readiness and formal
  benchmark readiness.

`diversity_ready_for_training_smoke` only means the generated subset is coherent
enough for short pipeline checks. `diversity_ready_for_formal_benchmark` is
intentionally stricter and should not be treated as evidence of publication-ready
data without additional review of the fixed manifest, split protocol,
condition-wise coverage, and multi-seed evaluation.

## Risks

- Low-power samples change label scale and can make relative errors dominate
  interpretation.
- High dynamic range source maps can increase hotspot difficulty while
  preserving weak-background sensitivity.
- Very high `top_h` changes the temperature rise range and may shift the
  background-hotspot tradeoff.
- High k contrast and low-k barrier variants may increase solver conditioning
  risk; smoke diagnostics must check residual norm and bottom Dirichlet error.
- Adding too many axes at once can make attribution difficult, so Gap-A avoids
  larger stack-template and random-k expansions.

## Reporting Boundary

Use conservative wording:

- research diagnostic candidate;
- generation smoke;
- benchmark-candidate preparation;
- controlled dataset tooling.

Do not describe Gap-A as a formal benchmark, high-fidelity validation,
publication-ready dataset, or solved OOD generalization result.
