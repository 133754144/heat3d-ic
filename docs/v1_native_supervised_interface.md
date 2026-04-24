# v1 Native Supervised Interface

## Main Task

The v1 mainline is steady supervised operator learning for temperature prediction.

The operator should learn:

```text
(coords, condition_features) -> target_temperature
```

where `target_temperature` is the steady temperature field `T(x)`.

This is not a transient task, not a coarse-to-fine task, and not a formal
large-scale training setup yet.

## Native Semantics

The v1-native supervised interface uses:

- `coords` / geometry as geometric input
- `condition_features` as known physical condition fields
- `target_u` / `target_temperature` as the temperature field to predict

Current `condition_features` are:

```text
k_field
q_field
BC encoding
```

With the current canonical `diag3` conductivity encoding, feature ordering is:

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

Future condition features may add other known physical fields, but they should
remain separate from the supervised target.

## Temperature Target

`temperature.npy` is the supervised target:

```text
target_u = target_temperature = T(x)
```

It is used during supervised training as the label / prediction target. It must
not be required as an inference-time input.

At inference time, the model should receive physical conditions and coordinates,
then output the predicted temperature field.

## Legacy Bridge

The existing model stack still has an `Inputs(u, c, x_inp, x_out, t, tau)` API.

Any conversion from the v1-native contract into that API must be treated as a
legacy bridge, for example:

```text
to_legacy_inputs(...)
```

The old compatibility smoke path:

```text
u = k_x
c = [k_y, k_z, q, BC encoding]
```

is not the canonical v1 semantics. It is only an adapter-level implementation
detail used to verify that the old interface can execute.

## Interpretation of Existing Smoke Results

The existing legacy adapter smoke results show only that:

- metadata samples can be read
- condition features can be packed
- graph construction can run
- the old model-facing adapter can execute
- tiny optimizer smoke can run without numerical failure

They do not prove:

- model accuracy
- generalization
- OOD performance
- solver fidelity
- physical metric validity
- that the final v1 training interface has been settled
