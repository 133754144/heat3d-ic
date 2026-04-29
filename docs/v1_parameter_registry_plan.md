# Heat3D v1 Parameter Registry Plan

## Purpose

The v1 small supervised smoke dataset uses named parameter categories for stack
templates, thermal conductivity, heat sources, boundary conditions, and split
roles. Many current values are provisional.

This document plans a parameter registry so future generated labels can be
reproducible and auditable without presenting provisional smoke assumptions as
literature-backed physical truth.

## Source Categories

Every important parameter should be classified as exactly one of:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

These labels should be stored near the resolved parameter values and propagated
into `sample_meta.json` or label-generation metadata.

## Registry Scope

### Thermal Conductivity Values / Ranges

The registry should define material or equivalent-region conductivity values
for:

- active die regions
- TIM-equivalent regions
- interposer-equivalent regions
- substrate-equivalent regions
- heatsink or spreader-equivalent regions
- block-wise equivalent material patches

Current small/smoke values should be treated as
`provisional_engineering_assumption` unless specific literature citations or
user-confirmed ranges are added.

### Anisotropy Ratios

The registry should define diagonal anisotropy ratios for `(N,3)` diagnostic
samples:

```text
k_field = [k_x, k_y, k_z]
```

Current anisotropy settings are diagnostic and should remain provisional until
validated against material assumptions or literature ranges.

### Volumetric Heat-Generation Scales

The registry should resolve categories such as:

- `low`
- `nominal`
- `high`

Current relative multipliers may remain:

```text
low = 0.5 x nominal
nominal = 1.0 x nominal
high = 1.5 x nominal
```

The nominal absolute value must be tagged. If it is not literature-backed or
user-confirmed, it should remain `requires_user_confirmation`.

### Top Robin HTC Categories

The registry should resolve top heat-transfer categories such as:

- `low`
- `nominal`
- `high`
- `held_out_top_h`

Current relative multipliers may remain:

```text
low = 0.5 x nominal
nominal = 1.0 x nominal
high = 1.5 x nominal
held_out_top_h = 2.0 x nominal
```

These multipliers should not be described as literature-backed by default. The
nominal HTC value and held-out range require user confirmation or literature
support before benchmark use.

### BC Baseline Temperatures

The registry should define baseline temperature categories such as:

- `baseline_300K`
- `shifted_350K`

The current 300 K / 350 K values are useful for baseline-shift diagnostics.
They should remain `provisional_engineering_assumption` unless the dataset
scope later defines them as validated operating points.

### Layer Thicknesses

The registry should store layer thicknesses for each stack template. It should
separate:

- physical thickness value
- unit
- source category
- role in the stack
- whether it is an equivalent-layer abstraction

Current values should remain provisional unless backed by literature or user
confirmation.

### Footprint Sizes

The registry should store footprint dimensions and resolution assumptions.
Current v1 smoke data uses regular rectangular footprints. Footprint and grid
settings should be tagged as provisional until the target physical scenario is
confirmed.

### Stack Templates

The registry should define stack templates such as:

- `baseline_4_layer`
- `compact_3_layer`
- `dual_active_4_layer`
- `interposer_like_4_layer`
- `heldout_interposer_4_layer`

Each template should record:

- layer sequence
- intended role
- smoke / diagnostic / held-out tag
- allowed heat-source layers
- equivalent-layer notes
- parameter source tags

`heldout_interposer_4_layer` should remain a smoke candidate, not proof of
stack OOD generalization.

### Equivalent-Layer Simplifications

Equivalent-layer assumptions should be explicit. For example:

- TSV / bump / interposer-like fine structures are not explicit geometry
- their first-stage representation is block-wise or layer-wise equivalent
  thermal conductivity
- the simplification source category must be recorded

The existence of an equivalent layer does not make the sample industrially
realistic by itself.

### Split / Purpose Tags

The registry should standardize tags such as:

- `train`
- `valid`
- `test_smoke`
- `test_ood_bc`
- `test_ood_stack`
- `diagnostic_anisotropy`
- `baseline_shift_diagnostic`

`test_ood_bc` and `test_ood_stack` should be marked as smoke candidates until a
formal benchmark protocol exists.

### Seed / Reproducibility Fields

Each sample should include:

- deterministic seed
- manifest version
- generator version or commit
- parameter registry version
- label solver version
- source manifest path

This is required for reproducible data generation and for future comparison
between smoke labels and solver v2 labels.

## Recommended Registry Format

A future machine-readable registry can be JSON or YAML. It should include:

- named categories
- resolved values when available
- units
- source category
- citation or user-confirmation placeholder
- notes
- allowed use: smoke, diagnostic, benchmark-candidate, or deprecated

The registry should not silently fill unresolved physical values. If a value is
unknown, it should be explicit and should block benchmark use.

## Current Caution

The current small/smoke dataset contains many provisional values. Those values
are acceptable for smoke diagnostics, but not for formal benchmark claims.

Before using Heat3D v1 labels in a paper or public benchmark, the parameter
registry must be reviewed and upgraded with literature-backed or user-confirmed
values.
