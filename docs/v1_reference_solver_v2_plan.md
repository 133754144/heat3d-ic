# Heat3D v1 Reference Solver v2 Plan

## Purpose

The current Heat3D v1 training and metrics scaffolds can run end to end. The
next bottleneck is not model training. The next bottleneck is the physical
credibility of `temperature.npy` labels.

This document plans a reference solver v2 direction for more reliable steady
thermal labels. It is a planning document, not a claim that such a solver is
already implemented.

## Current Smoke Reference Solver

The current smoke reference solver supports:

- steady 3D heat conduction
- regular layered rectangular stacks
- top Robin boundary condition
- bottom Dirichlet boundary condition
- side adiabatic boundary condition
- perfect-contact interfaces
- isotropic thermal conductivity with `(N,1)` `k_field`
- diagonal anisotropic thermal conductivity with `(N,3)` `k_field`

The current solver is useful for smoke labels, interface validation, and
checking the current `DeltaT` learning contract.

## Current Limitations

The current smoke solver is not:

- a high-fidelity solver
- a COMSOL / commercial FEM benchmark
- an industrial package thermal simulation workflow
- a general solver for irregular geometry
- a solver for explicit TSV / BEOL / bump structures
- a solver with contact resistance support
- a transient thermal solver
- a multiphysics solver
- sufficient support for formal model-performance or generalization claims

Its labels should be treated as smoke labels only.

## Why Solver v2 Is Needed

The v1 scaffold now has:

- manifest-driven small supervised samples
- supervised target and batch smoke
- relative BC features
- zero-delta bridge
- train / valid smoke
- validation metrics smoke

Those stages show that interfaces are runnable. They do not prove that the
labels are physically strong enough for a benchmark.

Before expanding training or making stronger claims, the project should improve
the reference label generator and attach diagnostics that can identify bad or
weak labels.

## Recommended Numerical Direction

Reference solver v2 should still target:

- steady 3D heat conduction
- regular layered rectangular stacks
- block-wise / equivalent material regions
- isotropic and diagonal anisotropic conductivity
- top Robin, bottom Dirichlet, side adiabatic boundaries
- perfect-contact interfaces

The preferred direction is finite volume or conservative finite difference.

Finite volume is a strong fit because:

- it naturally expresses heat-flux conservation over control volumes
- it can treat discontinuous material coefficients at cell faces
- it gives a direct path to energy-balance diagnostics
- it aligns with Robin / Neumann / adiabatic boundary flux accounting
- it can support harmonic averaging at material jumps

A conservative finite-difference implementation may also be acceptable if it
uses flux-form discretization and exposes equivalent residual / flux checks.

## Minimal Implementation Stage

The first minimal implementation is tracked as:

```text
rigno/heat3d_v1_reference_solver_v2.py
scripts/check_heat3d_v1_reference_solver_v2.py
```

This implementation is a small dense linear-system path for verification smoke.
It uses conservative finite-difference / finite-volume style conductances,
harmonic mean at neighboring conductivity jumps, and emits solver metadata.

It is still a research reference path. It is not a high-fidelity solver and
should not be used as a formal benchmark label generator.

## Conductivity Treatment

The solver v2 design should support:

- `(N,1)` isotropic conductivity
- `(N,3)` diagonal anisotropic conductivity ordered as `k_x`, `k_y`, `k_z`

At jumps in `k`, face coefficients should use harmonic mean or an equivalent
conservative flux treatment. Arithmetic averaging should not be used silently
at discontinuities without justification.

`(N,6)` full tensor conductivity should remain a schema-reserved capability and
should not be implemented in this stage unless explicitly requested.

## Boundary Conditions

Solver v2 should explicitly support:

- top Robin: `-k grad(T) dot n = h (T - T_inf)`
- bottom Dirichlet: fixed temperature
- side adiabatic: zero normal heat flux

Boundary condition metadata should remain explicit in `sample_meta.json`.
Boundary parameters should not be hidden inside masks.

## Interface Treatment

The first solver v2 target should support perfect-contact interfaces:

- temperature continuity across interfaces
- conservative normal flux treatment across material jumps

Contact resistance should be reserved for later. It should not be mixed into
the first solver v2 step.

## Diagnostics to Output

Reference solver v2 should output or record:

- linear solver residual
- convergence flag
- iteration count or solve status
- bottom Dirichlet consistency
- top Robin boundary consistency proxy
- side adiabatic flux proxy
- interface flux mismatch proxy
- global energy balance proxy
- finite-value and range checks for `temperature.npy`
- warnings and failure classification

These diagnostics should be stored as label-generation metadata, not as model
performance metrics.

## Validation Role

Solver v2 diagnostics should answer:

- Did the numerical solve converge?
- Are boundary conditions represented consistently?
- Are interface fluxes plausible under the discretization?
- Is generated `temperature.npy` finite and physically plausible?
- Is the heat balance proxy within a smoke-level tolerance?

They should not be used to claim formal physical validity until the
discretization and tolerances have been reviewed.

## Non-Claims

Reference solver v2 should still not be described as:

- an industrial high-fidelity solver
- a validated commercial FEM replacement
- a formal benchmark label generator
- evidence of model performance
- evidence of OOD generalization

It should be described as a more auditable research reference label generator
for the restricted Heat3D v1 problem family.
