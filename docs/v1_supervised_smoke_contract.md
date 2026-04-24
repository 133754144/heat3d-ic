# v1 Supervised Smoke Contract

## Purpose

This document fixes the current v1 supervised smoke contract for the research branch.
It describes an interface-smoke setup, not a formal training experiment.

The project mainline is steady supervised operator learning for temperature prediction:

- condition inputs: `coords + encoded_k_field + q_field + BC encoding`
- supervised target: steady temperature field stored as `temperature.npy`
- prediction target: steady temperature field

This is not a transient task and not a coarse-to-fine task.

## Supervised Smoke Subset

The supervised smoke subset is:

`subsets/v1_multilayer_bc_eq_supervised_smoke/`

Current local sample path:

`data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_smoke/samples/`

It currently contains only:

- `sample_000`
- `sample_005`

The subset exists to verify that v1 metadata samples can carry supervised steady
temperature labels and enter the current model-facing interface. It is not a
benchmark split, not a data-scale experiment, and not evidence of model accuracy.

## Canonical Model-Facing Mode

The current canonical model-facing mode is:

`k_encoding_mode="diag3"`

Conductivity handling:

- raw `(N,1)` isotropic `k_field` expands to `k_x, k_y, k_z`
- raw `(N,3)` diagonal anisotropic `k_field` remains `k_x, k_y, k_z`
- raw `(N,6)` symmetric tensor input is not supported in this canonical smoke path

## Feature Ordering

The current pure-physics feature ordering is:

```text
k_x
k_y
k_z
q
is_top
is_bottom
is_side
is_interior
top_h
top_T_inf
bottom_T_fixed
```

Semantic arrays such as `layer_id`, `region_id`, and `material_id` remain metadata
bookkeeping / optional auxiliary features. They are not part of the default
model-facing input contract.

## Supervised Target

The supervised target is:

`temperature.npy`

It is the steady temperature field used as the supervised label. It must not be
treated as a required inference-time input.

The current target normalization contract is:

- compute target mean/std over the tiny supervised smoke batch
- train smoke loss is MSE on normalized temperature
- raw / denormalized temperature is reserved for reporting metrics

## Adapter Policy

The current RIGNO interface expects `Inputs(u, c, x_inp, x_out, t, tau)`.

For this smoke path only:

- `u` receives the first normalized feature channel
- `c` receives the remaining normalized feature channels

This `u/c` split is a compatibility adapter detail. It is not the intended
long-term physical meaning of the v1 input representation.

## Reference Solver Scope

The current reference solver is smoke-only. It exists to generate `temperature.npy`
for the two supervised smoke samples.

It is limited to:

- regular layered rectangular stacks
- top Robin / bottom Dirichlet / sides adiabatic
- perfect-contact interfaces
- `(N,1)` isotropic and `(N,3)` diagonal anisotropic `k_field`

It is not a formal high-fidelity data generator and must not be described as an
industrial 3D IC thermal solver.

## Tiny Training Smoke Scope

The tiny training smoke verifies only:

- batch-level input / target shapes
- graph metadata and graph construction
- forward pass
- backward pass
- optimizer update
- finite normalized loss values
- checkpoint-free repeatability under fixed seed and fixed hyperparameters

It does not prove:

- model accuracy
- generalization
- OOD performance
- physical fidelity
- adequacy of the reference solver
- readiness for a real training run
