# v1 Zero-Delta Temperature-Rise Bridge

## Purpose

The v1-native problem statement remains:

```text
coords + condition_features -> target_temperature
```

`temperature.npy` is the supervised target. It is not an inference input.

For the current temperature-rise bridge, the learning target is:

```text
Delta T = T - T_ref
```

and inference recovers absolute temperature with:

```text
T_pred = T_ref + Delta T_pred
```

## Relative BC Features

The zero-delta bridge uses the optional relative BC feature view:

- `k_x`
- `k_y`
- `k_z`
- `q`
- `is_top`
- `is_bottom`
- `is_side`
- `is_interior`
- `top_h`
- `top_T_inf_minus_T_ref`
- `bottom_T_fixed_minus_T_ref`

Raw absolute BC temperatures are not part of this view.

## Bridge Contract

The internal legacy bridge is:

```text
legacy_inputs.u = zero_delta_field
legacy_inputs.c = relative_condition_features
target = Delta T
```

`T_ref` is still resolved from metadata, but it is used only for:

- constructing `target_delta_u`
- recovering `T_pred = T_ref + Delta T_pred`

It is not placed in `legacy_inputs.u` or `legacy_inputs.c`.

## Why Not tref_u_bridge

The earlier `tref_u_bridge` used:

```text
legacy_inputs.u = T_ref
```

This is non-leaking, but a pure baseline shift such as `300 K -> 350 K` still
changes the model-facing `u` input. The zero-delta bridge keeps model-facing
inputs invariant under that shift when all other physics are unchanged and BC
temperature features are represented relative to `T_ref`.

## Current Limit

This bridge smoke checks only interface consistency, graph construction, forward
execution, and loss-input shape compatibility. It is not a training result and
does not prove model accuracy, physical fidelity, or OOD generalization.
