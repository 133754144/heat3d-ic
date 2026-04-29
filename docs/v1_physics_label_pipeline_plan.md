# Heat3D v1 Physics-Label Pipeline Plan

## Stage Purpose

This stage moves Heat3D v1 from runnable smoke scaffolds toward a more
structured physics-label dataset pipeline.

The goal is not to expand training, tune models, or claim model performance.
The goal is to make temperature-label generation and dataset construction more
explicit, reproducible, and physically auditable.

This stage remains planning and smoke-diagnostic work. It is not a formal
benchmark, not evidence of OOD generalization, and not a high-fidelity
industrial 3D IC thermal simulation workflow.

## Current Starting Point

The previous stages established:

- metadata-only v1 demo samples
- a two-sample supervised smoke subset
- a 16-sample small supervised smoke dataset
- relative BC feature view
- `zero_delta_u_bridge`
- normalized `DeltaT = T - T_ref` target
- train / valid smoke
- validation metrics smoke
- a validation metrics closeout report

The current validated smoke route is:

```text
coords + condition_features -> target_temperature
condition_features = relative BC feature view
bridge = zero_delta_u_bridge
target = DeltaT = T - T_ref
recovery = T_pred = T_ref + DeltaT_pred
loss = normalized DeltaT MSE
```

All conclusions so far are limited to smoke diagnostics.

## Dataset and Sample Status

### Metadata-Only Demo

`v1_multilayer_bc_eq_demo` contains metadata-first samples with arrays such as
`coords.npy`, `layer_id.npy`, `region_id.npy`, `material_id.npy`, `k_field.npy`,
`q_field.npy`, and `sample_meta.json`.

This is an implemented smoke scaffold. It is not a solver-complete dataset.

### Supervised Smoke

`v1_multilayer_bc_eq_supervised_smoke` adds smoke-only `temperature.npy` labels
for selected samples. It verifies that supervised targets can be read and
packed.

This is implemented smoke infrastructure. It is not a benchmark.

### Supervised Small 16-Sample Dataset

`v1_multilayer_bc_eq_supervised_small` is manifest-driven and covers:

- 10 train samples
- 3 valid samples
- 1 `test_smoke` sample
- 1 `test_ood_bc` candidate
- 1 `test_ood_stack` candidate

It is implemented as a smoke dataset under ignored `data/` paths. It supports
train / valid smoke and validation metrics smoke. It is not evidence of model
performance or OOD generalization.

### Diagnostic Candidates

The current `test_ood_bc` and `test_ood_stack` samples are diagnostic smoke
candidates only. They do not support OOD claims without a larger controlled
dataset, stronger labels, and a formal evaluation protocol.

The current real `(N,3)` diagonal anisotropic samples are diagnostic samples
for data-contract and loader support. They do not establish anisotropic
generalization.

## Pipeline Structure

The intended v1 physics-label pipeline is:

```text
manifest/config
-> parameter registry
-> metadata generation
-> steady thermal label generation
-> label diagnostics
-> supervised dataset validation
-> reuse existing train/valid/metrics smoke
```

### Manifest / Config

The manifest describes sample IDs, splits, stack templates, source patterns,
thermal-conductivity modes, BC categories, parameter source tags, and smoke
roles.

It should remain the source of truth for dataset construction.

### Parameter Registry

The parameter registry should resolve named categories into explicit values or
documented unresolved placeholders. It should track whether each value is:

- `literature_backed`
- `provisional_engineering_assumption`
- `requires_user_confirmation`

The registry should prevent provisional smoke values from being presented as
literature-backed physical ranges.

### Metadata Generation

Metadata generation should remain manifest-driven and deterministic. It should
write only ignored generated data unless explicitly approved for publication.

The metadata stage should record:

- stack template and role
- layer and region definitions
- material and equivalent-layer assumptions
- source block definitions
- BC definitions
- split and purpose tags
- parameter source tags
- generation seed and reproducibility fields

### Steady Thermal Label Generation

The label generator should produce `temperature.npy` from metadata and arrays.

In this stage, label generation should move from smoke-only toward a more
credible reference solver v2, but it should still be described as a research
reference path, not a validated commercial-FEM replacement.

### Label Diagnostics

Label diagnostics should check array validity, temperature ranges, boundary
consistency, interface consistency proxies, energy balance proxies, and solver
convergence metadata.

Physics diagnostics must only be claimed once the underlying numerical
discretization supports them.

### Supervised Dataset Validation

Generated samples should reuse existing v1 smoke checks where possible:

- schema validation
- supervised target sanity
- supervised batch check
- zero-delta bridge check
- train / valid smoke
- validation metrics smoke

These remain smoke checks unless a formal benchmark protocol is defined.

## Current Strategy

This stage should not prioritize adding many new dataset types. It should first
standardize the current family:

- regular multilayer rectangular stacks
- block-wise / equivalent thermal conductivity
- limited diagonal anisotropy diagnostics
- explicit top Robin, bottom Dirichlet, side adiabatic BCs
- perfect-contact interfaces
- held-out HTC smoke candidate
- held-out stack smoke candidate
- 300 K / 350 K baseline-shift diagnostic

The immediate objective is a cleaner and more auditable pipeline around the
current 16-sample family, not a larger benchmark.

## Deferred Capabilities

Do not introduce the following in this stage:

- explicit TSV / BEOL / bump geometry
- irregular footprint
- unequal die overhang
- contact resistance
- transient simulation
- electro-thermal, fluid, reliability, or other multiphysics coupling
- `(N,6)` full tensor conductivity samples
- industrial package-level benchmark claims

These may be future research directions, but they should not be mixed into the
current physics-label pipeline stabilization step.

## Non-Goals

This stage does not establish:

- a formal benchmark
- OOD generalization
- model performance
- high-fidelity solver validity
- industrial 3D IC simulator readiness
- formal physical labels suitable for publication without further validation

Current smoke labels should not be described as final physics labels.

## Recommended Next Implementation Order

1. Add a parameter registry plan and then a small machine-readable registry.
2. Define reference solver v2 diagnostics and metadata outputs.
3. Add label diagnostics smoke that can run on current generated labels.
4. Upgrade label generation in a v2 path without breaking existing smoke data.
5. Reuse existing train / valid / metrics smoke only after label diagnostics
   pass.
