# v1 Relative BC Feature Diagnostic

## Purpose

The v1-native task remains:

```text
coords + condition_features -> target_temperature
```

For temperature-rise learning, the supervised bridge target is:

```text
Delta T = T - T_ref
```

and inference recovers:

```text
T_pred = T_ref + Delta T_pred
```

`T_ref` is a non-leaking metadata-derived baseline. It is not ground truth and
not a coarse solution.

## Raw BC Temperature Risk

The current default condition feature contract includes raw absolute BC
temperature channels:

- `top_T_inf`
- `bottom_T_fixed`

The baseline-shift diagnostic showed that, within the current linear smoke
solver, shifting both boundary temperatures from `300 K` to `350 K` shifts the
absolute temperature field by `50 K` while leaving `Delta T` essentially
unchanged.

If training sees only `300 K` and evaluation uses `350 K`, raw absolute
temperature channels become distribution-shifted even though the
temperature-rise target is invariant.

## Relative BC Feature View

The optional relative view keeps:

- `k_x`
- `k_y`
- `k_z`
- `q`
- `is_top`
- `is_bottom`
- `is_side`
- `is_interior`
- `top_h`

and replaces:

- `top_T_inf`
- `bottom_T_fixed`

with:

- `top_T_inf_minus_T_ref`
- `bottom_T_fixed_minus_T_ref`

This view is diagnostic-only for now. It does not replace the current default
loader feature contract.

## Bridge Policy Comparison

Two internal legacy bridge policies are useful to compare:

```text
tref_u_bridge:
legacy_inputs.u = T_ref
legacy_inputs.c = relative_condition_features
target = Delta T
```

```text
zero_delta_u_bridge:
legacy_inputs.u = zero_delta_field
legacy_inputs.c = relative_condition_features
target = Delta T
```

`zero_delta_u_bridge` is more baseline-shift invariant because both the legacy
`u` slot and the relative condition features can remain unchanged under a pure
boundary baseline shift. `T_ref` is still used for target construction and final
temperature recovery, but it does not enter the legacy `u` input slot.

## Current Limit

This is a feature / bridge diagnostic only. It is not a training result, not a
model-performance result, and not a physics validation beyond the current smoke
solver assumptions.
