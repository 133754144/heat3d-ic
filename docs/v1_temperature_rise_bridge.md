# v1 Temperature-Rise Legacy Bridge

## Purpose

The v1-native public semantics remain:

```text
coords + condition_features -> target_temperature
```

The temperature-rise bridge is an internal legacy bridge for the existing RIGNO
`Inputs(u, c, ...)` API. It is not the v1 public interface.

## Bridge Policy

The bridge constructs:

```text
legacy_inputs.u = T_ref
legacy_inputs.c = condition_features
target_delta_u = target_temperature - T_ref
```

The model-facing loss can then align the model output with:

```text
Delta T = T - T_ref
```

Inference recovers the temperature field as:

```text
T_pred = T_ref + Delta T_pred
```

## T_ref Baseline

`T_ref` is a non-leaking metadata-derived baseline. It is not ground truth and
not a coarse solution.

Baseline resolution order:

1. bottom Dirichlet fixed temperature if present
2. top Robin ambient temperature if present
3. fallback constant `300 K`

`T_ref` must never be computed from `temperature.npy`.

## Relation to Legacy u=k_x Smoke

The old bridge:

```text
u = k_x
c = [k_y, k_z, q, BC encoding]
```

is now historical compatibility smoke only. It should not be used as the next
v1 training design.

The temperature-rise bridge is still a legacy bridge because it targets the old
RIGNO `Inputs(u, c, ...)` API, but it has better physical semantics: the legacy
`u` slot is temperature-like and non-leaking.

## Current Limit

This bridge smoke verifies only interface consistency and a forward / loss-input
path. It does not prove model performance, physical fidelity, OOD behavior, or
training readiness.
