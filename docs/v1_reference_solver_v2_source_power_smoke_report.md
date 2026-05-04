# Heat3D v1 Reference Solver v2 Source-Power Smoke Report

## Purpose

This report records the S3.5 source normalization / total-power consistency smoke for the Heat3D v1 reference solver v2.

The goal is to check whether temporary controlled samples at different node counts represent the same physical heat source, instead of changing total power accidentally as resolution changes.

This is a source-power consistency smoke and resolution diagnostic. It is not a formal energy-balance check, not formal grid convergence, not high-fidelity solver validation, not a formal benchmark, and not model-performance evidence.

## Controlled Setup

The smoke script is:

`scripts/check_heat3d_v1_reference_solver_v2_source_power.py`

It creates temporary samples only and removes them after the run. It does not write a formal dataset and does not modify the existing `v1_multilayer_bc_eq_supervised_small` subset.

Common physical setup:

- domain: `0.01 m x 0.01 m x 0.002 m`
- source box:
  - x: `0.003 m` to `0.007 m`
  - y: `0.003 m` to `0.007 m`
  - z: `0.0005 m` to `0.0015 m`
- source volume: `1.6e-08 m^3`
- q amplitude: `1.0e8 W/m^3`
- expected integrated power: `1.6 W`
- bottom Dirichlet: `300 K`
- top Robin ambient: `300 K`
- top HTC: `1000 W/m^2/K`

The source projection uses a control-volume overlap fraction. This keeps the same physical source box and q amplitude while making the integrated power consistent across coarse / mid / fine grids.

## Results

| k mode | resolution | node count | active nodes | active source volume | integrated q power | T range | DeltaT max | residual norm | top Robin proxy | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| isotropic | coarse | 64 | 8 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 302.246722]` | `2.246722` | `5.611527e-17` | `1.142857e-01 W` | pass |
| isotropic | mid | 180 | 12 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 304.630338]` | `4.630338` | `7.617027e-17` | `1.142857e-01 W` | pass |
| isotropic | fine | 384 | 64 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 304.570252]` | `4.570252` | `8.783543e-17` | `1.142857e-01 W` | pass |
| diag3 | coarse | 64 | 8 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 303.771909]` | `3.771909` | `5.142112e-17` | `2.000000e-01 W` | pass |
| diag3 | mid | 180 | 12 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 306.903036]` | `6.903036` | `3.990171e-17` | `2.000000e-01 W` | pass |
| diag3 | fine | 384 | 64 | `1.600000e-08` | `1.600000e+00` | `[300.000000, 306.991100]` | `6.991100` | `5.138110e-17` | `2.000000e-01 W` | pass |

Summary checks:

- node count increases with resolution: true
- source volume consistency: true
- source volume maximum relative difference:
  - isotropic: `2.067951531382569e-16`
  - diag3: `2.067951531382569e-16`
- integrated q power consistency: true
- integrated q power maximum relative difference:
  - isotropic: `0.0`
  - diag3: `0.0`
- residuals below current solver tolerance: true
- bottom Dirichlet errors below tolerance: true
- all cases finite and converged: true

## Interpretation

The source-power smoke did not find unintended total-power changes with resolution in these controlled samples. The integrated q power stayed at `1.6 W` for coarse, mid, and fine grids in both isotropic and diagonal anisotropic cases.

The earlier T_max changes observed in the resolution smoke can therefore be separated from a simple total-power normalization bug for this controlled source projection. In this smoke, T range still changes with resolution, but total source power is held fixed.

The top Robin heat removal proxy is reported only as a boundary heat-removal proxy. It is not a global energy balance because the bottom Dirichlet boundary can also absorb or supply heat.

## Non-Claims

Do not claim from this smoke:

- formal energy balance
- formal grid convergence
- high-fidelity solver validation
- COMSOL/FEM agreement
- formal benchmark readiness
- model performance
- OOD generalization

Flux, energy, and continuous PDE residual diagnostics remain `not_computed` or `requires_numerical_operator`.

## Next Step

If source-power consistency remains stable, the next planning step can be a physics-label medium-small dataset design.

Before moving to larger generated datasets, the project should still define:

- whether manifest-driven generators should use this control-volume source projection;
- source normalization metadata fields;
- source power checks in label diagnostics;
- manufactured-solution tests;
- external solver comparison;
- proper flux / energy diagnostics.
