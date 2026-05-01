# Heat3D v1 Reference Solver v2 Smoke Contract

## Purpose

The reference solver v2 path is a minimal research reference implementation for
the Heat3D v1 physics-label pipeline.

Its purpose is to make temperature-label generation more auditable than the
initial smoke-only label path. It is not a high-fidelity solver, not a COMSOL
or commercial FEM replacement, and not a formal benchmark label generator.

## Current Equation Family

The current solver targets steady 3D heat conduction:

```text
-div(k grad T) = q
```

under the restricted Heat3D v1 setup:

- regular multilayer rectangular stack
- volumetric heat generation `q`
- isotropic conductivity `(N,1)`
- diagonal anisotropic conductivity `(N,3)`
- top Robin boundary condition
- bottom Dirichlet boundary condition
- side adiabatic boundary condition
- perfect-contact interfaces

## Discretization

The implementation uses a conservative finite-difference / finite-volume style
linear system on the current rectilinear point grid.

Conductance between neighboring nodes is assembled in flux form. At material
jumps, neighboring face conductivity uses a harmonic mean. This is intended to
avoid silently using arithmetic averaging at discontinuous conductivity jumps.

The implementation is still minimal and dense-matrix based. It is suitable for
small verification smoke, not large dataset generation.

## Boundary Conditions

Current support:

- bottom Dirichlet is enforced directly in the linear system
- top Robin is included as a boundary conductance term
- side adiabatic is treated as a natural zero-flux boundary

The verification smoke checks finite temperatures, residual proxy, convergence
flag, and bottom Dirichlet consistency.

## Metadata Output

`solve_reference_temperature_v2` returns:

```text
temperature, label_meta
```

`label_meta` includes:

- `solver_name`
- `solver_version`
- `solver_role`
- `discretization_type`
- `supported_k_mode`
- `convergence_flag`
- `residual_norm`
- `bottom_dirichlet_error`
- `top_robin_status`
- `side_adiabatic_status`
- `interface_status`
- `energy_balance_status`
- `pde_residual_status`
- assembly metadata
- duplicate-node merge metadata
- warnings

Some diagnostics are intentionally status-only in this first smoke. Global
energy balance and continuous PDE residual are not fully computed.

## Verification Smoke

Run:

```bash
python3 scripts/check_heat3d_v1_reference_solver_v2.py
```

The script uses the ignored local
`v1_multilayer_bc_eq_supervised_small` subset and checks `sample_000` and
`sample_005` by default.

It does not overwrite existing `temperature.npy`. It writes temporary
temperature and metadata files to a temporary directory and removes them after
the smoke.

## Non-Supported Features

This stage does not support:

- explicit TSV / BEOL / bump geometry
- irregular footprint
- contact resistance
- transient simulation
- multiphysics
- `(N,6)` full tensor conductivity
- industrial package benchmark claims

## Next Integration Steps

Future steps should:

1. add solver v2 output to a separate physics-label small v2 subset
2. store `label_meta` next to each generated `temperature.npy`
3. make label diagnostics read solver metadata
4. rerun existing zero-delta / normalized DeltaT train-valid smoke on v2 labels
5. rerun validation metrics smoke with clear non-claims
