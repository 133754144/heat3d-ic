# Heat3D v1 Research Status

## Scope

This document records the current v1 research scaffold status for:

```text
subsets/v1_multilayer_bc_eq_demo/
```

The v1 mainline is steady supervised operator learning for temperature
prediction:

```text
coords + condition_features -> target_temperature
```

`temperature.npy` is the supervised target. It is not an inference input.

## Completed Layers

### Metadata and Schema

- v1 metadata-first sample schema exists.
- Samples include arrays such as `coords.npy`, `layer_id.npy`, `region_id.npy`,
  `material_id.npy`, `k_field.npy`, `q_field.npy`, and `sample_meta.json`.
- `temperature.npy` is reserved for supervised / solver smoke samples.
- The schema supports `(N,1)`, `(N,3)`, and `(N,6)` thermal-conductivity field
  shapes, while current smoke data covers `(N,1)` and one diagnostic `(N,3)`
  sample.

### Loader

- A v1 metadata loader exists for pure-physics style inputs.
- Default model-facing condition features are based on coordinates, thermal
  conductivity, heat source, and boundary-condition encoding.
- Semantic labels such as `layer_id`, `region_id`, and `material_id` remain
  metadata / optional auxiliary information, not required model inputs.

### Graph Smoke

- v1 graph smoke checks show that existing Heat3D graph construction can use
  coordinates for topology while pure-physics features remain node features.
- This does not modify the v0 graph builder public behavior.

### Supervised Smoke Subset

- A very small supervised smoke subset exists for `sample_000` and `sample_005`.
- `sample_000` covers the main `(N,1)` isotropic-style path through `diag3`
  expansion.
- `sample_005` covers a real `(N,3)` diagonal anisotropic diagnostic path.

### Reference Solver Smoke

- A minimal reference steady solver exists only for smoke-label generation.
- It supports the current small regular layered stack setup:
  top Robin, bottom Dirichlet, side adiabatic, perfect-contact interfaces, and
  `(N,1)` / `(N,3)` thermal conductivity.
- It must not be described as a formal high-fidelity or industrial data
  generator.

### Native Supervised Contract

- v1-native semantics are separated from the legacy RIGNO `Inputs(u, c, ...)`
  interface.
- The native contract is:

```text
condition_features -> target_temperature
```

- `target_temperature` is stored as `temperature.npy`.

### Relative BC Feature View

- An optional relative boundary-condition feature view exists.
- It keeps:
  `k_x`, `k_y`, `k_z`, `q`, `is_top`, `is_bottom`, `is_side`, `is_interior`,
  and `top_h`.
- It replaces raw absolute BC temperatures with:
  `top_T_inf_minus_T_ref` and `bottom_T_fixed_minus_T_ref`.
- This reduces input distribution shift under pure boundary-baseline
  temperature changes.

### Zero-Delta Bridge

- The current recommended legacy bridge for smoke is:

```text
legacy_inputs.u = zero_delta_field
legacy_inputs.c = relative_condition_features
target = Delta T
```

- `T_ref` is metadata-derived and non-leaking.
- `T_ref` is used for target construction and final recovery, not as a
  model-facing input.

### Normalized DeltaT Tiny Training Smoke

- A very tiny training smoke exists to verify forward, backward, optimizer,
  normalized loss, raw DeltaT recovery, and repeatability.
- It uses only `sample_000` and `sample_005`.
- It uses normalized `Delta T` as the loss target.
- It is an interface and numerical-stability smoke, not a training experiment.

## Current Default Supervised Smoke Route

The current default v1 supervised smoke route is:

```text
condition_features = relative BC feature view
internal bridge = zero_delta_u_bridge
target = Delta T = T - T_ref
recovery = T_pred = T_ref + DeltaT_pred
loss = MSE(normalized_DeltaT_pred, normalized_DeltaT_target)
```

## Legacy u=k_x Route

The old route:

```text
u = k_x
c = [k_y, k_z, q, BC encoding]
```

is now historical compatibility smoke only.

It is:

- not canonical v1 semantics
- not the recommended future training interface
- not the route to extend for v1 steady supervised training

## Current Non-Claims

Do not claim:

- model performance
- OOD generalization
- formal training completion
- high-fidelity solver validity
- complete physics metrics
- industrial 3D IC / package simulator readiness
- final v1 training pipeline stability

The current state proves only that the v1 scaffold, data contracts, graph
construction, bridge contracts, and tiny smoke paths are internally runnable.
