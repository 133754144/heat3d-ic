# Heat3D v1 Label Diagnostics Smoke Contract

## Purpose

This document describes the implemented Heat3D v1 label diagnostics smoke.

The diagnostics check basic label health for supervised samples. They are label
quality smoke checks, not model metrics and not formal physics validation.

## Implemented Checks

The current implementation checks each labelled sample directory for:

- required files:
  - `coords.npy`
  - `k_field.npy`
  - `q_field.npy`
  - `temperature.npy`
  - `sample_meta.json`
- array shape
- dtype
- NaN / Inf
- consistent `N` across coordinates, thermal conductivity, heat source, and
  temperature
- supported current-smoke `k_field` shape:
  - `(N,1)`
  - `(N,3)`
- unsupported current-smoke `k_field` shape:
  - `(N,6)`
- `T_min`, `T_max`, `T_mean`
- `DeltaT_min`, `DeltaT_max`, `DeltaT_mean`
- peak temperature value, index, and coordinate
- `T_ref` resolved from:
  1. bottom Dirichlet fixed temperature
  2. top Robin ambient temperature
  3. fallback `300 K`
- bottom Dirichlet simple consistency using `z_min` bottom points

The bottom Dirichlet check reports `pass`, `warning`, or `fail` based on the
maximum absolute error from the fixed bottom temperature.

## Explicitly Not Implemented

The following are not implemented in this smoke:

- PDE residual
- top Robin flux violation
- side adiabatic flux violation
- interface flux mismatch
- global energy balance residual

They are reported as:

- `requires_numerical_operator`
- or `not_computed`

These checks require a credible numerical discretization / flux operator and
should be implemented with reference solver v2 or later.

## CLI

Run:

```bash
python3 scripts/check_heat3d_v1_label_diagnostics.py
```

The default path is:

```text
data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small
```

If the path does not exist, the script exits with a clear error. It does not
generate data.

## Status Semantics

Per-sample `overall_status` is:

- `pass`: required arrays and basic checks pass
- `warning`: basic checks run, but a non-fatal concern exists
- `fail`: required files, shape, finite-value, or bottom Dirichlet consistency
  checks fail

If any sample fails, the CLI exits non-zero.

## Non-Claims

This diagnostics smoke does not establish:

- high-fidelity labels
- physics validation
- formal benchmark readiness
- model performance
- OOD generalization

It is a minimal label-health check for the current v1 physics-label pipeline.
