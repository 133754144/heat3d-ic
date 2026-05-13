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
