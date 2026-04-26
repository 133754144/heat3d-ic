# Heat3D v1 Small Supervised Dataset Plan

This document plans the next small supervised dataset step for the v1 research
branch. It is a design document only. It does not define a formal benchmark,
does not generate data, and does not claim model performance.

## Current Scaffold Baseline

The current v1 scaffold mainline is:

```text
coords + condition_features -> target_temperature
```

The current recommended supervised smoke route is:

```text
condition_features = relative BC feature view
internal bridge = zero_delta_u_bridge
target = DeltaT = T - T_ref
recovery = T_pred = T_ref + DeltaT_pred
loss = MSE(normalized_DeltaT_pred, normalized_DeltaT_target)
```

`temperature.npy` is the supervised label and prediction target. It is not an
inference input.

The historical `u = k_x` bridge is retained only as compatibility smoke. It is
not the canonical v1 semantics and should not be the default route for the next
dataset step.

## Current Implemented Capabilities

### Metadata Generator

`tools/generate_heat3d_v1_metadata_smoke.py` currently generates a small
metadata-only subset with six configured samples:

| Sample | Split | Purpose |
| --- | --- | --- |
| `sample_000` | `train` | baseline 4-layer `(N,1)` metadata smoke |
| `sample_001` | `train` | compact 3-layer `(N,1)` metadata smoke |
| `sample_002` | `valid` | 4-layer block-wise `(N,1)` validation smoke |
| `sample_003` | `test_id` | dual-active-layer metadata smoke |
| `sample_004` | `test_ood_stack` | held-out interposer-like stack smoke |
| `sample_005` | `valid` | real `(N,3)` diagonal anisotropic diagnostic smoke |

The generator supports the current stack templates and one diagnostic
anisotropic `k_field` mode. It does not yet support an external manifest,
parameter sweeps, deterministic seed tables, systematic BC variation, or a
multi-sample supervised split.

### Supervised Smoke Generator

`tools/generate_heat3d_v1_supervised_smoke.py` currently copies only:

```text
sample_000
sample_005
```

into `subsets/v1_multilayer_bc_eq_supervised_smoke/` and adds
`temperature.npy` with the minimal reference steady solver.

### Reference Solver Scope

`rigno/heat3d_v1_reference_solver.py` is a smoke-only reference solver. It
currently supports:

- regular layered rectangular stacks
- top Robin boundary condition
- bottom Dirichlet boundary condition
- side adiabatic boundary condition
- perfect-contact interfaces
- `(N,1)` isotropic thermal conductivity
- `(N,3)` diagonal anisotropic thermal conductivity

It does not support:

- high-fidelity industrial data generation
- irregular geometry
- explicit TSV / bump / BEOL / package microstructure
- contact resistance
- `(N,6)` symmetric thermal-conductivity tensors
- transient simulation
- general FEM validation

## Small Supervised Dataset Goal

The next dataset should be a very small supervised dataset for train / valid
smoke development.

Its purpose is to verify that the current v1 supervised route can organize,
load, batch, normalize, train-smoke, validate-smoke, and recover temperatures
across more than two supervised samples.

It is not:

- a formal benchmark
- a model-performance experiment
- an OOD generalization claim
- a complete 3D IC dataset
- an industrial thermal simulator dataset

## Recommended Size and Splits

Recommended initial size:

```text
16 supervised samples
```

Recommended split:

| Split | Count | Meaning |
| --- | ---: | --- |
| `train` | 10 | common stacks, common BCs, heat-source and material variations |
| `valid` | 3 | same-distribution small variants for train / valid smoke |
| `test_smoke` | 1 | held-out source-pattern smoke only |
| `test_ood_bc` | 1 | held-out top Robin HTC range smoke candidate |
| `test_ood_stack` | 1 | held-out stack-template smoke candidate |

The `test_ood_bc` and `test_ood_stack` samples should be described as smoke
candidates only. They should not be used to claim OOD generalization until
there is a larger controlled dataset and evaluation protocol.

## Variation Dimensions

### Thermal-Conductivity Field

- Most samples should use `(N,1)` isotropic equivalent conductivity.
- One or two diagnostic samples should use real `(N,3)` diagonal anisotropic
  conductivity.
- `(N,6)` should remain schema-supported but not generated in this small step.

### Stack Templates

Use only regular layered rectangular stacks:

- baseline 4-layer stack
- simplified 3-layer stack
- interposer-like / TIM-like variation
- dual-active-layer variation, where supported by the current schema

Do not introduce irregular footprints, unequal die overhang, or explicit TSV /
BEOL / package microstructure in this step.

### Heat Sources

Vary only smoke-level heat-source patterns:

- single active-layer source
- two source spots in one active layer
- dual active layers where the stack template supports it
- source location shifts within the active layer
- low / nominal / high `q` scale categories

`q_field` remains volumetric heat generation in `W/m^3`.

### Boundary Conditions

Keep the current first-stage BC family:

```text
top Robin
bottom Dirichlet
sides adiabatic
```

The design should include:

- baseline `300 K` BC temperature smoke cases
- shifted `350 K` BC temperature smoke cases for relative-feature diagnostics
- small top Robin HTC variation in train / valid
- a held-out top Robin HTC category as one `test_ood_bc` smoke candidate

Raw absolute BC temperatures should not be the recommended model-facing feature
view for temperature-rise learning. The recommended feature view is relative to
`T_ref`.

### Anisotropy

Include one or two true `(N,3)` diagonal anisotropic diagnostic samples. These
should verify the loader / bridge / smoke-training contract, not establish
anisotropic generalization.

## Parameter Source Categories

All important parameters should be explicitly tagged as one of:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

For this small supervised dataset plan, exact numerical ranges should remain
conservative:

| Parameter Group | Current Planning Status |
| --- | --- |
| BC pattern: top Robin, bottom Dirichlet, side adiabatic | `literature_backed` as a common thermal-modeling pattern, but exact values still need confirmation |
| Equivalent-layer abstraction for fine structures | `literature_backed` as a modeling direction, but exact stack definitions still need confirmation |
| Footprint and layer thickness values | `provisional_engineering_assumption` |
| Thermal conductivity values and anisotropy ratios | `provisional_engineering_assumption` / `requires_user_confirmation` |
| Volumetric heat generation scales | `provisional_engineering_assumption` / `requires_user_confirmation` |
| Top Robin HTC categories | `provisional_engineering_assumption` / `requires_user_confirmation` |
| `300 K` and `350 K` baseline temperatures | `provisional_engineering_assumption` for diagnostic shift testing |
| Final train / valid / OOD split semantics | `requires_user_confirmation` before public benchmark use |

Do not present provisional values as precise literature-derived ranges.

## 16-Sample Manifest Draft

This table is a planning manifest only. It should become an explicit config or
manifest before any data are generated.

| sample_id | split | stack_template | k_field_shape | anisotropy_type | heat_source_pattern | q_scale_category | T_ref / BC baseline | top_h category | expected purpose | parameter_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sample_000` | `train` | baseline_4_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 300 K | nominal | baseline train anchor | provisional_engineering_assumption |
| `sample_001` | `train` | baseline_4_layer | `(N,1)` | isotropic equivalent | single left-shifted active_die_0 | low | 300 K | nominal | heat-source position variation | provisional_engineering_assumption |
| `sample_002` | `train` | baseline_4_layer | `(N,1)` | block-wise equivalent | single right-shifted active_die_0 | high | 300 K | nominal | q-scale and blockwise-k variation | provisional_engineering_assumption |
| `sample_003` | `train` | compact_3_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 300 K | nominal | simplified stack training smoke | provisional_engineering_assumption |
| `sample_004` | `train` | compact_3_layer | `(N,1)` | block-wise equivalent | single offset active_die_0 | low | 300 K | low | HTC and source variation | provisional_engineering_assumption |
| `sample_005` | `train` | dual_active_4_layer | `(N,1)` | isotropic equivalent | dual active layers | nominal | 300 K | nominal | multi-layer heat-source smoke | provisional_engineering_assumption |
| `sample_006` | `train` | baseline_4_layer | `(N,1)` | block-wise equivalent | two spots in active_die_0 | high | 300 K | high | multi-hotspot smoke | provisional_engineering_assumption |
| `sample_007` | `train` | interposer_like_4_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 300 K | nominal | interposer-like train variation | provisional_engineering_assumption |
| `sample_008` | `train` | baseline_4_layer | `(N,3)` | diagonal anisotropic diagnostic | single centered active_die_0 | nominal | 300 K | nominal | anisotropic training-contract diagnostic | provisional_engineering_assumption |
| `sample_009` | `train` | dual_active_4_layer | `(N,1)` | block-wise equivalent | dual active layers | high | 350 K | nominal | shifted-baseline train smoke | provisional_engineering_assumption |
| `sample_010` | `valid` | baseline_4_layer | `(N,1)` | block-wise equivalent | single offset active_die_0 | low | 300 K | nominal | same-family validation smoke | provisional_engineering_assumption |
| `sample_011` | `valid` | compact_3_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 350 K | nominal | shifted-baseline validation smoke | provisional_engineering_assumption |
| `sample_012` | `valid` | baseline_4_layer | `(N,3)` | diagonal anisotropic diagnostic | single offset active_die_0 | nominal | 300 K | high | anisotropic validation-contract diagnostic | provisional_engineering_assumption |
| `sample_013` | `test_smoke` | baseline_4_layer | `(N,1)` | block-wise equivalent | held-out source location | nominal | 300 K | nominal | held-out source-pattern smoke | provisional_engineering_assumption |
| `sample_014` | `test_ood_bc` | baseline_4_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 300 K | held-out_top_h | held-out top HTC smoke candidate | requires_user_confirmation |
| `sample_015` | `test_ood_stack` | heldout_interposer_4_layer | `(N,1)` | isotropic equivalent | single centered active_die_0 | nominal | 300 K | nominal | held-out stack smoke candidate | requires_user_confirmation |

## Recommended Generation Strategy

Do not generate data until this plan is confirmed.

The next implementation step should be additive:

1. Add a small-dataset manifest or config file for the 16 planned samples.
2. Extend the metadata generator to read explicit sample parameters from that
   manifest instead of hardcoding every variation.
3. Extend the supervised generator to accept the same manifest and sample IDs.
4. Keep deterministic seed values in the manifest.
5. Write generated data only under ignored `data/`.
6. Do not commit generated sample directories.
7. Do not change v0 entrypoints.
8. Do not modify `rigno/models/*`.

The generator extension should make sample differences explicit:

- split
- stack template
- k-field mode
- source pattern
- q scale
- BC baseline temperature
- top Robin HTC category
- parameter source tags

## Recommended Smoke Commands After Generation

After the small supervised dataset is generated, run at least:

```bash
python3 scripts/validate_heat3d_v1_schema.py
python3 scripts/check_heat3d_v1_loader.py
python3 scripts/check_heat3d_v1_supervised_targets.py
python3 scripts/check_heat3d_v1_supervised_batch.py
python3 scripts/check_heat3d_v1_native_supervised_contract.py
python3 scripts/check_heat3d_v1_relative_bc_features.py
python3 scripts/check_heat3d_v1_zero_delta_bridge.py
python3 scripts/check_heat3d_v1_zero_delta_tiny_training.py
```

Additional next-stage smoke to add before real training:

```text
small train / valid split loader smoke
small train / valid normalized DeltaT stats smoke
small train / valid one-epoch-or-few-step training smoke
validation-loop metric-shape smoke
```

These commands remain smoke checks. They should not be reported as formal model
performance.

## User Confirmation Needed

Before implementation, the following need user confirmation:

- exact physical ranges for thermal conductivity and anisotropy ratios
- exact volumetric heat-generation scale categories
- exact top Robin HTC categories and held-out HTC range
- whether `350 K` shifted-baseline cases should enter train / valid or remain
  diagnostic only
- whether `test_ood_stack` should use an interposer-like template first, or a
  different held-out layer-count / thickness template
- whether the 16-sample plan should keep all samples at the current coarse
  rectilinear point resolution, or slightly vary resolution only after the first
  train / valid smoke succeeds
