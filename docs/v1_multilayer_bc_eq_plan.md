# v1 Multilayer BC Equivalent Demo Plan

## Scope

The first v1 target subset is:

`subsets/v1_multilayer_bc_eq_demo/`

This subset is a definition-layer research skeleton for a more realistic 3D IC-like steady thermal task. It is not a solver-complete dataset and is not an industrial chiplet / TSV / package thermal simulation platform.

## First-stage task definition

The first-stage task is steady 3D temperature-field prediction on a regular multilayer rectangular stack represented as sampled points / graph nodes.

Inputs:

- point coordinates `coords`
- equivalent thermal conductivity field `k_field`
- volumetric heat-generation field `q_field`
- explicit thermal boundary-condition encoding

Output for later solver samples:

- steady temperature field `temperature`

The current metadata-only smoke stage does not include `temperature.npy`. Solver samples must add `temperature.npy`.

## Geometry and abstraction

The first-stage geometry is intentionally restricted to:

- regular layered rectangular stack
- point-cloud / sampled-node representation
- 3-layer and 4-layer stack templates
- layer-wise and block-wise equivalent thermal conductivity

The first-stage geometry does not include:

- irregular footprint
- unequal die overhang
- explicit TSV / BEOL / package microstructure
- transient thermal simulation
- full multiphysics coupling

Fine structures such as TSV, bump, and interposer-like regions should first be represented by equivalent thermal-conductivity layers or blockwise equivalent regions. The approximation must be recorded in metadata.

## Boundary-condition setup

Boundary conditions must be explicit metadata and later numerical features. They must not be represented as a boundary mask only.

The first-stage main boundary setup is:

- top: Robin
- bottom: Dirichlet
- sides: adiabatic

The first OOD boundary-condition target is reserved as:

- held-out top Robin HTC range

This metadata-only smoke batch does not generate a `test_ood_bc` sample. It only reserves the design.

## Split design

The split design is not random. It is designed to support cross-structure and cross-boundary-condition generalization.

Current metadata-only smoke samples:

| Sample | Split | Purpose |
| --- | --- | --- |
| `sample_000` | `train` | 4-layer baseline stack |
| `sample_001` | `train` | 3-layer compact stack |
| `sample_002` | `valid` | 4-layer stack with block-wise equivalent conductivity and passive `tim_equiv` coupling layer |
| `sample_003` | `test_id` | same-distribution stack with unseen multi-layer heat-source pattern |
| `sample_004` | `test_ood_stack` | held-out stack template with interposer-equivalent layer |

Additional diagnostic sample:

| Sample | Split | Purpose |
| --- | --- | --- |
| `sample_005` | `valid` | diagonal anisotropic diagnostic sample with real `(N,3)` `k_field`; used only to verify metadata/schema/loader support |

Reserved later split:

- `test_ood_bc`: held-out top Robin HTC range

Optional later split:

- `test_ood_material`: held-out equivalent material range

## Parameter-source policy

Important parameters must be classified as one of:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

This applies to:

- thermal conductivity values and ranges
- volumetric heat-generation values and ranges
- Robin HTC values and ranges
- ambient / fixed temperatures
- layer thicknesses and footprint
- equivalent-layer simplifications

For this metadata-only smoke batch, concrete numerical values are provisional engineering assumptions unless explicitly tagged otherwise in `sample_meta.json`.

## Model-input policy

The long-term target is a mostly pure-physics input:

`coords + k_field + q_field + BC encoding`

The schema keeps `layer_id`, `region_id`, and `material_id` because they are useful for data construction, validation, OOD splits, interface descriptions, and future physics metrics. They should be treated as optional auxiliary model features, not as mandatory long-term inputs.

The current first loader skeleton should default to this pure-physics input mode. `layer_id`, `region_id`, and `material_id` are reserved as metadata bookkeeping and optional auxiliary features only.

For conductivity encoding, the loader should support:

- `k_encoding_mode="native"` for preserving `(N,1)`, `(N,3)`, or `(N,6)`
- `k_encoding_mode="diag3"` for expanding isotropic `(N,1)` conductivity to diagonal `(N,3)` features

The current metadata-only samples remain `(N,1)` only. A later solver/train stage should add real `(N,3)` diagonal anisotropy support in data generation, not only loader-side expansion.

The main smoke benchmark set remains `sample_000` through `sample_004`, which are still `(N,1)`-based. `sample_005` is intentionally diagnostic rather than benchmark-facing, and exists only to exercise a real `(N,3)` metadata-only path.

For the current smoke set, `sample_002` does not treat `tim_equiv` as a heat-generating layer. That layer is kept passive in the metadata because the first-stage benchmark is intended to separate active heat-source placement from passive thermal-coupling layers. Multi-layer heat-source metadata is demonstrated instead by `sample_003`.

## Current non-goals

This first v1 batch does not implement:

- solver smoke samples
- `temperature.npy`
- v1 training or evaluation
- v1 loader integration with the existing v0 training pipeline
- model-core changes
- PDE residual metrics
- boundary-condition violation metrics
- interface heat-flux mismatch metrics

## Tiny supervised smoke subset

To keep the metadata-first mainline intact while testing the supervised steady-learning path, a separate very small subset may be created under:

`subsets/v1_multilayer_bc_eq_supervised_smoke/`

This tiny subset should:

- reuse only selected v1 metadata samples,
- add `temperature.npy` as a supervised target,
- keep the mainline semantics unchanged:
  - inputs: condition fields and coordinates
  - target: steady temperature field

Any temperature labels in this tiny subset are for interface smoke validation only. They are not a formal high-fidelity batch data-generation pipeline.
