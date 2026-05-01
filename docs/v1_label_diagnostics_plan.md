# Heat3D v1 Label Diagnostics Plan

## Purpose

This document plans diagnostics for future Heat3D v1 temperature labels.

The current labels are smoke labels. This plan defines what should be checked
before any label set is described as physically credible.

Diagnostics in this document are planning items unless the required numerical
operators are implemented.

## Diagnostic Levels

### Array Sanity

These checks can be implemented early:

- array shape consistency
- dtype consistency
- finite-value checks for NaN / Inf
- unit consistency
- `sample_meta.json` and array consistency
- `coords`, `k_field`, `q_field`, and `temperature` length agreement
- allowed `k_field` shape: `(N,1)` or `(N,3)` for current solver paths
- absence of unsupported `(N,6)` generated samples in this stage

Array sanity checks are smoke diagnostics. They do not prove physical
correctness.

### Temperature Sanity

These checks can be reported for every label:

- `T_min`
- `T_max`
- `T_mean`
- `DeltaT_min`
- `DeltaT_max`
- `DeltaT_mean`
- true peak temperature
- peak temperature location
- bottom boundary temperature range
- warning if temperature rise is negative or implausibly large

Temperature sanity checks can flag obviously bad labels. They do not by
themselves validate the PDE solve.

### Boundary-Condition Diagnostics

Planned boundary diagnostics:

- bottom Dirichlet consistency
- top Robin residual or proxy
- side adiabatic flux proxy

Bottom Dirichlet consistency can be checked directly from boundary nodes and
metadata.

Top Robin and side adiabatic flux diagnostics require a credible gradient or
finite-volume flux operator. Until that exists, they should be reported only as
planned diagnostics or smoke-level proxies.

### Interface Diagnostics

Planned interface diagnostics:

- perfect-contact interface temperature continuity proxy
- interface flux mismatch proxy
- `k` jump handling check
- harmonic-mean face coefficient check when using finite volume

Interface temperature continuity can be approximated only if interface pairing
or cell adjacency is well defined. Interface flux mismatch requires a
conservative numerical operator and should not be faked from metadata alone.

### Global Diagnostics

Planned global diagnostics:

- global energy balance proxy
- integrated heat generation
- integrated heat removal through Robin boundaries
- residual norm from the linear system
- convergence flag
- iteration count or solver status
- warning / failure classification

These diagnostics should be emitted by the label-generation path and stored as
label metadata.

## Required Metadata

Future label diagnostics should record:

- solver name and version
- parameter registry version
- manifest version
- grid resolution
- boundary-condition values
- convergence status
- residual norm
- energy-balance proxy
- diagnostics pass / warning / fail status

This information should be separated from model metrics. It describes label
quality, not prediction quality.

## Failure Classification

Diagnostics should distinguish:

- `pass`
- `warning`
- `fail`
- `not_computed`
- `requires_numerical_operator`

For example, a label may pass array sanity but have
`interface_flux_mismatch = not_computed` until interface flux operators are
implemented.

## Current Limits

Only array sanity and simple temperature / boundary checks are safe to compute
with the current smoke infrastructure.

The following must not be reported as completed physics diagnostics until a
credible numerical discretization operator exists:

- PDE residual
- top Robin BC violation
- side adiabatic flux violation
- interface flux mismatch
- global energy conservation residual

They may be reported as planned diagnostics or smoke-level placeholders only.

## Recommended Implementation Order

1. Add an array and temperature sanity checker for existing labels.
2. Add explicit diagnostic metadata fields with `not_computed` status for
   physics checks not yet supported.
3. Implement reference solver v2 with flux-form operators.
4. Add BC and interface diagnostics from the same discrete operator used for
   label generation.
5. Run label diagnostics before train / valid or validation metrics smoke.

## Implemented Smoke

The initial label diagnostics smoke is implemented as:

```text
rigno/heat3d_v1_label_diagnostics.py
scripts/check_heat3d_v1_label_diagnostics.py
```

It covers array sanity, temperature sanity, peak-temperature reporting, and a
simple bottom Dirichlet consistency check. Flux, residual, interface, and global
energy checks remain `requires_numerical_operator` or `not_computed`.

## Non-Claims

Label diagnostics are not model metrics. They should not be used to claim model
performance or OOD generalization.

Until solver v2 and its diagnostics are implemented and reviewed, Heat3D v1
labels should remain smoke / research labels rather than formal benchmark
labels.
