# Heat3D v1 Parameter Registry Contract

## Purpose

The parameter registry is the first machine-readable planning artifact for the
physics-label pipeline stage.

It records named parameter categories, source tags, allowed use, units, and
unresolved fields before the registry is connected to metadata generation or
label generation.

This is smoke / diagnostic / planning infrastructure. It is not a formal
benchmark definition and does not make current parameter values
literature-backed.

## Registry Path

The current registry file is:

```text
configs/heat3d_v1/parameter_registry_v1.json
```

The current checker is:

```text
scripts/check_heat3d_v1_parameter_registry.py
```

The current loader / validator helper is:

```text
rigno/heat3d_v1_parameter_registry.py
```

## Required Top-Level Fields

The registry must contain:

- `registry_version`
- `parameter_groups`

The registry may also contain:

- `description`
- `non_claims`
- `allowed_source_categories`
- `allowed_uses`

## Parameter Groups

The initial registry covers:

- `thermal_conductivity`
- `anisotropy_ratios`
- `q_scales`
- `top_robin_htc`
- `bc_baseline_temperatures`
- `layer_thicknesses`
- `footprint_sizes`
- `stack_templates`
- `equivalent_layer_simplifications`
- `split_purpose_tags`
- `reproducibility_fields`

Every group must have non-empty `entries`, unless it is explicitly marked as
`planned_empty` with a `planned_empty_reason`.

## Entry Schema

Each entry should include:

- `key` or `name`
- `value`
- `unit`
- `unresolved`
- `source_category`
- `allowed_use`
- `notes`

If `unresolved` is true, the entry must include:

- `unresolved_reason`

Resolved entries must include:

- non-null `value`
- `unit`

## Source Category Rules

Allowed source categories are:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

Unknown, implicit, or untagged source categories are forbidden.

`literature_backed` entries must include `citation` or `reference`. The current
registry intentionally avoids marking smoke parameters as literature-backed.

## Allowed Use Rules

Allowed use values are:

- `smoke`
- `diagnostic`
- `benchmark_candidate`
- `deprecated`

`benchmark_candidate` does not mean formal benchmark. It only means a parameter
may later be considered for a benchmark after solver, data, and evaluation
protocols are upgraded.

## Unresolved Parameters

Unresolved values must be explicit. They should not be silently filled with
defaults.

Examples include:

- nominal volumetric heat generation
- nominal top Robin HTC
- layer thicknesses
- footprint dimensions
- equivalent material conductivity ranges

These can support smoke planning but must be resolved before any formal
benchmark or publication-facing physical claims.

## Current Non-Integration

The registry currently does not drive:

- metadata generation
- temperature label generation
- label diagnostics
- reference solver v2
- training or validation metrics smoke

Those integrations should happen in later additive steps after the registry
contract is stable.

## Validation Command

Run:

```bash
python3 scripts/check_heat3d_v1_parameter_registry.py
```

The checker prints:

- registry version
- parameter group count
- source category counts
- allowed use counts
- requires-user-confirmation entries
- provisional entries
- unresolved entries

Validation failure exits non-zero.

## Non-Claims

The registry does not establish:

- high-fidelity solver validity
- formal benchmark readiness
- model performance
- OOD generalization
- industrial 3D IC thermal simulation readiness
