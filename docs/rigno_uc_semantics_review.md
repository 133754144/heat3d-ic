# RIGNO `u/c` Semantics Review for Heat3D v1

## Scope

This review checks how upstream RIGNO uses `u` and `c`, how the current Heat3D
v0 path reuses that interface, and what this implies for the v1-native steady
supervised interface.

No model, dataset, or training behavior is changed by this document.

## Upstream RIGNO Semantics

Upstream RIGNO defines `Inputs(u, c, x_inp, x_out, t, tau)`.

Observed behavior:

- `batch.u` is the solution / primary field loaded from the dataset.
- `batch.c` is an optional coefficient or condition field.
- If a dataset has no coefficient group, then `c=None`.
- Time-dependent examples such as `Heat-L-Sines` use `u` as the current state
  and predict a future state / residual / time derivative.
- Datasets such as `Poisson-Gauss`, `AF`, `Elasticity`, and `Wave-Layer` have
  `group_c`, so `c` represents known spatial coefficients or condition fields.
- The model internally concatenates `inputs.u` and `inputs.c` along the feature
  channel dimension before graph encoding.
- The model asserts `inputs.u.shape[-1] == num_outputs`. This is a model-facing
  interface constraint, not a general physical-law statement.

For time-independent upstream problems, the training code uses:

```text
inputs.u = batch.c[:, [0]]
inputs.c = None
u_tgt = batch.u[:, [0]]
```

So upstream already contains an engineering reuse pattern where a known
coefficient field is placed into the `Inputs.u` slot in order to drive a
steady coefficient-to-solution problem.

## Current Heat3D v0 Semantics

The v0 Heat3D dataset loads:

- `temperature.npy` as the target temperature field
- `coords.npy` as geometry / point coordinates
- `k.npy` as thermal conductivity
- `source.npy` as heat source

It stores:

```text
sample["u"] = temperature
sample["c"] = [k, source]
sample["x"] = coords
```

The v0 Heat3D pipeline then does:

```text
u_tgt = sample["u"]
u_inp = first coefficient channel
c_inp = remaining coefficient channels
```

This means v0 is already a steady physical-parameters-to-temperature operator
learning task. Its `u_inp/c_inp` split is an interface-compatibility strategy,
not a clean physical semantics. Physically, temperature is the predicted field,
while thermal conductivity and heat source are known conditions.

## v1 Native Semantics

The v1 mainline should remain:

```text
coords + condition_features -> target_temperature
```

where:

- `target_u` / `target_temperature` is the steady temperature field `T(x)`.
- `temperature.npy` is the supervised target.
- `condition_features` are known physical conditions:
  - encoded `k_field`
  - `q_field`
  - BC encoding
  - future physical condition fields
- `coords` / geometry are geometric inputs.
- Ground-truth temperature must not be required as inference-time input.

The old smoke split:

```text
u = k_x
c = [k_y, k_z, q, BC encoding]
```

should stay classified as legacy compatibility smoke. It proves that the old
model-facing path can execute, but it should not define v1 problem semantics.

## Route Comparison

### A. Author-style steady input-function reuse

This follows upstream time-independent practice: put a known coefficient /
input function into the model's `Inputs.u` slot and supervise against the
solution field.

Pros:

- Matches upstream steady-problem engineering pattern.
- Requires no model-core rewrite.
- Already compatible with the `inputs.u.shape[-1] == num_outputs` assertion if
  the selected input function has one channel.

Cons:

- The `Inputs.u` slot name remains semantically misleading for Heat3D v1.
- Multi-channel conditions still need a disciplined bridge.

### B. v1-native condition-to-temperature wrapper

The public v1 interface is:

```text
coords + condition_features -> target_temperature
```

Any conversion into `Inputs(u,c,...)` is hidden inside an internal bridge.

Pros:

- Cleanly matches Heat3D v1 physics semantics.
- Avoids teaching downstream code that `u=k_x`.
- Can still reuse the existing RIGNO model and graph code.
- Keeps v0 untouched.

Cons:

- Requires a small v1-only bridge / adapter layer.
- Needs explicit documentation around how condition channels are packed for the
  old model interface.

### C. temperature-baseline bridge

Use a non-leaking baseline temperature field as `Inputs.u`, such as ambient or
bottom fixed temperature, and put all physical conditions in `Inputs.c`.

Pros:

- `Inputs.u` is temperature-like, so it is semantically less misleading than
  `u=k_x`.
- Avoids feeding true temperature labels at inference time.
- Compatible with output / residual interpretations if designed carefully.

Cons:

- Needs a clear baseline policy.
- Current model internally concatenates `u` and `c`, so the benefit is mainly
  semantic unless the stepper/loss is designed around baseline-to-target output.
- Still needs v1-specific stats and bridge code.

### D. Continue legacy `u=k_x`

This should remain historical smoke only.

Pros:

- Already runs.
- Minimal immediate engineering effort.

Cons:

- Misstates v1 semantics.
- Encourages future code to treat thermal conductivity as the primary field.
- Confuses `target_u` with condition channels.
- Not recommended for next-stage design.

## Recommendation

Use route B as the next design target:

```text
v1-native condition-to-temperature wrapper
```

The next implementation should be v1-only and additive:

- keep v0 public scripts unchanged
- keep `rigno/models/*` unchanged
- keep old smoke scripts as legacy compatibility checks
- add a bridge named explicitly as `legacy_bridge` or `to_legacy_inputs`
- keep public v1 semantics in terms of `condition_features` and
  `target_temperature`

The next smoke should not train. It should verify that the v1-native example can
be converted through a clearly named internal bridge without exposing `u=k_x` as
the canonical interface.
