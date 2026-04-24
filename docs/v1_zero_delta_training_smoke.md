# v1 Zero-Delta Tiny Training Smoke

## Scope

This is a smoke test only. It is not a formal training experiment and does not
support performance claims.

The current v1-native task remains:

```text
coords + condition_features -> target_temperature
```

`temperature.npy` is the supervised target and is not used as an inference
input.

## Default Smoke Bridge

The smoke uses:

```text
legacy_inputs.u = zero_delta_field
legacy_inputs.c = relative_condition_features
target = Delta T
```

where:

```text
Delta T = T - T_ref
T_pred = T_ref + Delta T_pred
```

`T_ref` is metadata-derived and non-leaking. It is used for target construction
and final recovery only. It is not inserted into model-facing inputs.

## Relative BC Feature Contract

The condition feature view is:

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

Raw absolute BC temperatures are intentionally excluded from this view.

## Loss Contract

The tiny smoke normalizes the target temperature rise and uses:

```text
loss = MSE(predicted_normalized_DeltaT, normalized_DeltaT)
```

For reporting, predictions are denormalized back to raw `Delta T`, then
temperature is recovered as:

```text
T_pred = T_ref + denormalized_DeltaT_pred
```

The reported MSE values are smoke diagnostics only.

## Relation to Legacy u=k_x

The old `u = k_x` adapter remains historical compatibility smoke. It is not the
preferred v1 training path. This zero-delta bridge is the current recommended
tiny-training smoke route because it keeps the legacy `u` slot temperature-rise
neutral and avoids absolute baseline-temperature input.
