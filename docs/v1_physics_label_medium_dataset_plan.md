# Heat3D v1 Physics-Label Medium Dataset Plan

## Objective

This document plans a Heat3D v1 physics-label medium-small dataset for the
publication-oriented physics-label pipeline. The first step is an 8-sample
pilot. If the pilot validates cleanly, the plan can expand to a 64-sample
benchmark-candidate dataset.

The dataset is intended to support later training and validation of the current
v1 graph / RIGNO pipeline using the existing relative-BC, zero-delta, normalized
DeltaT route. It remains a research reference / benchmark-candidate plan. It is
not a formal benchmark, not high-fidelity solver evidence, not model-performance
evidence, and not OOD generalization evidence.

## Core Generation Strategy

The medium-small dataset should use a region-first generation policy:

- Define physical regions first: domain, stack layers, material regions, source
  regions, and boundary-condition categories.
- Sample the rectilinear grid from those physical regions.
- Map material, conductivity, heat source, and boundary metadata from physical
  regions to grid nodes / control volumes.
- Use `volume_fraction` source assignment for source regions.
- Keep `center_in_box` only as a diagnostic baseline for source-assignment
  smoke checks.

The region-source discretization smoke showed that `center_in_box` can miss an
off-grid source on a coarse grid and can unintentionally change integrated
source power as resolution changes. `volume_fraction` preserved source volume
and integrated power in the controlled smoke.

## Planned 64-Sample Split

Recommended split for the eventual 64-sample medium-small dataset:

- train: 48
- valid: 8
- test_id: 4
- test_ood_bc_candidate: 2
- test_ood_stack_candidate: 2

The `test_ood_bc_candidate` and `test_ood_stack_candidate` splits are diagnostic
smoke candidates only. They must not be used to claim OOD generalization unless
later work adds a formal protocol and stronger evidence.

## 8-Sample Pilot Coverage

The first pilot should cover one compact representative set:

| sample | split | purpose |
|---|---|---|
| pilot_000 | train | baseline single hotspot |
| pilot_001 | train | shifted hotspot |
| pilot_002 | train | two hotspots |
| pilot_003 | train | dual active layers |
| pilot_004 | train | block-wise equivalent k |
| pilot_005 | valid | diagonal anisotropy diagnostic |
| pilot_006 | test_ood_bc_candidate | held-out top HTC candidate |
| pilot_007 | test_ood_stack_candidate | held-out stack candidate |

This pilot is only a generator / label / diagnostics pilot. It is not a model
comparison dataset by itself.

## Region Schema Draft

Each sample should be generated from a region-level schema before array
projection:

- `domain`: physical bounds, footprint, stack height, grid resolution category.
- `stack_template`: ordered layer regions, layer names, thickness categories,
  material-region tags, and equivalent-layer assumptions.
- `layer_regions`: z-intervals and layer-level metadata.
- `material_regions`: region IDs, material IDs, conductivity policy, and
  source-category references from the parameter registry.
- `source_regions`: physical boxes with layer target, center / size in physical
  or fractional coordinates, q policy, and source assignment policy.
- `bc_categories`: bottom Dirichlet, top Robin, side adiabatic, baseline
  temperature category, and top HTC category.
- `resolution_policy`: pilot / medium-small grid policy and allowed node-count
  categories.
- `q_policy`: first version `fixed_density`; future extension
  `fixed_total_power`.
- `source_assignment`: default `volume_fraction`.

## q Policy

The first version should use:

- `q_policy = fixed_density`

Under `fixed_density`, each source region records a volumetric heat generation
density, and the generator must compute the integrated source power implied by
the region volume and discretization.

Future extension:

- `q_policy = fixed_total_power`

Under `fixed_total_power`, the generator would infer the density required to
match a target total power for a physical source region. This is not the first
implementation target.

Every generated sample must record:

- `source_region_volume_target`
- `active_source_volume_discrete`
- `integrated_q_power`
- `active_source_cell_count`
- `source_volume_relative_error`
- `integrated_q_power_relative_error`
- `source_missed`

## Required Checks After Generation

The generator and validation scripts must reject or warn on unsafe samples:

- `source_missed = false`
- integrated source power within the configured tolerance
- `label_meta.json` exists for each labeled sample
- solver `convergence_flag = true`
- `residual_norm` within tolerance
- `bottom_dirichlet_error` within tolerance
- label diagnostics pass
- no unsupported k-field shape except planned diagnostic handling
- generated arrays stay under ignored `data/` paths

The first medium-small implementation should also re-run existing v1 smoke
checks on the generated subset before any model training or comparison.

## Non-Goals

This plan does not introduce:

- irregular footprint
- explicit TSV / BEOL / bump geometry
- contact resistance
- transient simulation
- electro-thermal / fluid / reliability multiphysics
- `(N,6)` full tensor conductivity
- formal OOD generalization claim
- model-performance claim
- industrial package-level benchmark claim

## Next Step

The next safe implementation step is a no-write manifest / resolver dry-run for
the 8-sample pilot, followed by metadata-only generation into ignored `data/`.
Only after source-power and label diagnostics pass should solver v2 labels be
generated for the pilot.
