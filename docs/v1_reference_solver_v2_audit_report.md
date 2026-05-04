# Heat3D v1 Reference Solver v2 Audit Report

## Scope

This report audits the relationship between the legacy smoke label path and the reference solver v2 path.

The result is a solver audit and verification smoke. It is not high-fidelity solver validation, not a formal benchmark, not model-performance evidence, and not OOD-generalization evidence.

## Code Path Relationship

The legacy supervised-small label generator uses:

- `tools/generate_heat3d_v1_supervised_smoke.py`
- `rigno/heat3d_v1_reference_solver.py`
- function: `solve_reference_temperature`

The physics-label v2 generator uses:

- `tools/generate_heat3d_v1_physics_label_v2.py`
- `rigno/heat3d_v1_reference_solver_v2.py`
- function: `solve_reference_temperature_v2`

These paths do not call the same Python solver function. The v2 generator does not copy the old `temperature.npy`. It copies metadata/source arrays, removes any existing generated label files in the target copy, calls `solve_reference_temperature_v2`, and writes a new `temperature.npy` plus `label_meta.json`.

The two solver modules currently implement equivalent dense conservative finite-difference / finite-volume-style algebra for the controlled rectilinear smoke cases. That equivalence is the main reason the old smoke labels and v2 labels match exactly in the current 16-sample audit.

## Why Smoke-vs-v2 Label Difference Is Zero

The smoke-vs-v2 label audit found:

- common samples: 16
- mean label-difference RMSE: `0.00000000e+00`
- mean label-difference MAE: `0.00000000e+00`
- max absolute label difference: `0.00000000e+00`
- peak-temperature difference: `0.00000000e+00`
- hotspot index matches: 16 / 16

This does not mean solver v2 has been externally validated or upgraded to high fidelity.

The current reason is narrower:

1. both paths solve the same restricted steady linear problem;
2. both use the same rectilinear point ordering and duplicate-node merge assumptions;
3. both use harmonic-mean face conductance for neighboring nodes;
4. both encode bottom Dirichlet, top Robin, and side adiabatic boundaries in the same algebraic way;
5. v2 currently adds explicit solver metadata and diagnostic reporting without intentionally changing the computed labels for these smoke samples.

## Current v2 Improvement

The current v2 improvement is primarily auditability and diagnostics, not a demonstrated change in numerical label values.

Implemented v2 improvements include:

- explicit solver name and version;
- explicit discretization metadata;
- explicit supported k-mode reporting;
- convergence flag;
- residual norm proxy;
- bottom Dirichlet error;
- top Robin / side adiabatic / interface status fields;
- explicit warnings, including `(N,1)` to diagonal `(N,3)` expansion;
- `label_meta.json` written next to every v2 `temperature.npy`.

The current v2 path is therefore better suited for a publication-oriented physics-label pipeline because it exposes solver provenance and smoke diagnostics. It is still not a high-fidelity or externally validated label generator.

## Verification Smoke Coverage

`scripts/check_heat3d_v1_reference_solver_v2.py` now covers:

- isotropic baseline samples: `sample_000` and `sample_005`;
- diagonal anisotropic sample: `sample_008`;
- zero-q case: zero volumetric heat generation on a copied sample should produce near-zero `DeltaT` under matching 300 K bottom/top baseline conditions;
- baseline-shift case: shifting bottom Dirichlet and top Robin ambient from 300 K to 350 K should shift `T` by 50 K while preserving `DeltaT`;
- finite temperature checks;
- convergence flag check;
- residual norm proxy check;
- bottom Dirichlet consistency check.

Current smoke output:

- `sample_000`: pass, residual norm about `2.619908e-15`;
- `sample_005`: pass, residual norm about `3.643322e-15`;
- `sample_008`: pass, `diag3` k mode, residual norm about `2.212597e-15`;
- zero-q case: pass, max absolute `DeltaT` about `3.069545e-12` K;
- baseline-shift case: pass, max shift error about `5.684342e-13` K;
- all cases ok: true.

These checks verify consistency of the current algebraic smoke path. They do not verify physical accuracy against an independent solver.

## Not Yet Computed

The following remain not computed or require a credible numerical operator before they can be treated as physics diagnostics:

- continuous PDE residual;
- top Robin flux violation;
- side adiabatic flux violation;
- interface heat-flux mismatch;
- global energy balance residual;
- contact resistance effects;
- external solver comparison.

Do not report these as completed physics validation metrics.

## Why v2 Is Still Not High Fidelity

The current solver v2 remains a minimal research reference path because it:

- only supports regular multilayer rectangular stacks;
- does not support irregular footprints or unequal die overhang;
- does not model explicit TSV / BEOL / bump microstructure;
- does not model contact resistance;
- does not support transient or multiphysics coupling;
- does not support `(N,6)` full tensor conductivity;
- has not passed grid-convergence studies;
- has not been checked against manufactured solutions;
- has not been compared against external FEM/FVM tools.

## Next Requirements For Publishable Labels

Before treating generated labels as publication-grade or benchmark-ready, the project should add:

- grid-convergence checks;
- manufactured-solution tests for controlled steady heat equations;
- independent external solver comparison, such as FEM/FVM reference outputs;
- explicit flux and energy-balance diagnostics;
- documented parameter registry provenance for physical values;
- failure/warning classification for generated labels;
- a baseline/model comparison protocol that separates train/valid diagnostics from benchmark claims.

Until then, v2 labels should be described as research reference labels and benchmark candidates only.
