# Heat3D v1 Reference Solver v2 Resolution Smoke Report

## Purpose

This report records a small resolution / node-count smoke for the Heat3D v1 reference solver v2.

The goal is to check whether the solver can run on controlled rectilinear samples with increasing node counts while producing finite temperatures, convergence metadata, residual proxies, and bottom Dirichlet diagnostics.

This is not a formal grid-convergence study, not high-fidelity solver validation, not a formal benchmark, and not model-performance evidence.

## Smoke Setup

The smoke script is:

`scripts/check_heat3d_v1_reference_solver_v2_resolution.py`

It creates temporary samples only and removes them after the run. It does not write a formal dataset and does not modify the existing `v1_multilayer_bc_eq_supervised_small` subset.

The checked resolutions are:

- coarse: `[4, 4, 4]`
- mid: `[6, 6, 5]`
- fine: `[8, 8, 6]`

The checked conductivity modes are:

- isotropic `(N,1)`, expanded internally by solver v2 to diagonal `(N,3)`
- diagonal anisotropic `(N,3)`

All cases use the current restricted solver v2 setting:

- steady rectilinear heat conduction
- top Robin
- bottom Dirichlet
- side adiabatic
- finite-volume-style / conservative finite-difference face treatment
- harmonic mean face conductivity

Flux, energy-balance, and continuous PDE residual diagnostics remain `not_computed` or `requires_numerical_operator`.

## Results

| k mode | resolution | node count | T min | T max | DeltaT max | peak coord | residual norm | bottom error | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| isotropic | coarse | 64 | 300.000000 | 308.321194 | 8.321194 | `[0.0066667, 0.0033333, 0.0013333]` | `7.501142e-17` | `0.000000e+00` | pass |
| isotropic | mid | 180 | 300.000000 | 307.000612 | 7.000612 | `[0.006, 0.004, 0.0015]` | `6.729159e-17` | `0.000000e+00` | pass |
| isotropic | fine | 384 | 300.000000 | 305.674189 | 5.674189 | `[0.0057143, 0.0042857, 0.0016]` | `8.767455e-17` | `0.000000e+00` | pass |
| diag3 | coarse | 64 | 300.000000 | 313.970034 | 13.970034 | `[0.0033333, 0.0033333, 0.0013333]` | `4.251802e-17` | `0.000000e+00` | pass |
| diag3 | mid | 180 | 300.000000 | 310.374352 | 10.374352 | `[0.004, 0.006, 0.0015]` | `3.750748e-17` | `0.000000e+00` | pass |
| diag3 | fine | 384 | 300.000000 | 307.792210 | 7.792210 | `[0.0042857, 0.0057143, 0.0016]` | `5.204449e-17` | `0.000000e+00` | pass |

Additional checks:

- node count increases with resolution: true
- all temperatures finite: true
- all convergence flags: true
- all residual norms below current solver tolerance: true
- all bottom Dirichlet errors below tolerance: true
- `DeltaT` ranges stayed within the smoke upper bound: true

## Interpretation

The solver v2 minimal path is stable on these temporary controlled cases as node count increases from 64 to 384.

The decreasing peak temperature trend in this smoke reflects the chosen temporary source discretization and grid sampling. It should not be interpreted as a formal convergence result.

## Non-Claims

Do not claim from this smoke:

- grid convergence
- high-fidelity solver validation
- COMSOL/FEM agreement
- formal benchmark readiness
- model performance
- OOD generalization

## Next Step

If this smoke remains stable, the next reasonable planning step is a medium-small physics-label dataset design, such as a 64-sample benchmark-candidate plan.

Before using such labels for publication-grade claims, the project still needs:

- manufactured-solution verification
- grid-convergence protocol
- external solver comparison
- energy / flux diagnostics
- stronger parameter provenance through the registry
