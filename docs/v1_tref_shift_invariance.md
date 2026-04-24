# v1 T_ref Baseline-Shift Invariance Diagnostic

## Purpose

The temperature-rise bridge predicts:

```text
Delta T = T - T_ref
```

and recovers:

```text
T_pred = T_ref + Delta T_pred
```

For linear steady heat conduction with the same heat generation, same thermal
conductivity, same geometry, and boundary temperatures shifted by the same
constant, the absolute temperature should shift by that constant while the
temperature rise should remain unchanged.

This motivates a diagnostic shift:

```text
300 K boundary baseline -> 350 K boundary baseline
```

## Diagnostic Scope

The diagnostic uses only temporary copies of:

- `sample_000`
- `sample_005`

It changes only:

- bottom Dirichlet fixed temperature
- top Robin ambient temperature

It keeps unchanged:

- coordinates
- `k_field`
- `q_field`
- top HTC
- layers / regions / materials
- interfaces

The shifted samples are not formal dataset samples.

## Expected Check

For each sample:

```text
T_shifted - T_base ~= 50 K
DeltaT_shifted ~= DeltaT_base
```

where:

```text
DeltaT_base = T_base - 300 K
DeltaT_shifted = T_shifted - 350 K
```

## Input-Encoding Risk

The current condition feature contract includes raw absolute BC temperature
channels:

- `top_T_inf`
- `bottom_T_fixed`

If training only sees `300 K` and a future evaluation uses `350 K`, those raw
absolute channels are distribution-shifted even though the temperature-rise
physics is invariant.

For the temperature-rise bridge, a better future design may use relative BC
temperature features, for example:

- `T_ref`
- `top_T_inf_minus_T_ref`
- `bottom_T_fixed_minus_T_ref`

This document records the risk. It does not change the current feature contract.
