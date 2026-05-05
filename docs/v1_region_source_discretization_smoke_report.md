# Heat3D v1 Region Source Discretization Smoke Report

## Purpose

This report records a region-source discretization smoke for the Heat3D v1
physics-label pipeline. The goal is to compare two controlled source assignment
policies before choosing a default policy for later medium-small physics-label
dataset work:

- `center_in_box`: assign `q_density` only when the node / cell center lies in
  the source box.
- `volume_fraction`: assign `q_density * overlap_volume / cell_volume` using
  the overlap between each control volume and the physical source box.

This is a source-assignment diagnostic and research reference smoke. It is not
formal grid convergence, not a formal benchmark, not high-fidelity validation,
and not model-performance evidence.

## Controlled Setup

The smoke uses temporary rectilinear samples only. No formal dataset is written.

- Domain: `x,y in [0, 0.01] m`, `z in [0, 0.002] m`
- Source box: `x,y in [0.0039, 0.0061] m`, `z in [0.00075, 0.00125] m`
- Target source volume: `2.420000e-09 m^3`
- `q_density`: `1.000000e+08 W/m^3`
- Target integrated source power: `2.420000e-01 W`
- Boundary conditions: bottom Dirichlet `300 K`, top Robin `300 K`, sides adiabatic
- Conductivity: isotropic `(N,1)` smoke case
- Resolutions:
  - coarse: `[4, 4, 4]`, `64` nodes
  - mid: `[6, 6, 5]`, `180` nodes
  - fine: `[8, 8, 6]`, `384` nodes

The source box is intentionally not aligned with the coarse grid node locations
so that point/center assignment can expose a source miss or source-volume
distortion.

## Source Assignment Summary

| method | resolution | active cells | active volume | integrated power | power rel. error | source missed |
|---|---:|---:|---:|---:|---:|---|
| center_in_box | coarse | 0 | `0.000000e+00` | `0.000000e+00` | `1.000000e+00` | true |
| center_in_box | mid | 4 | `8.000000e-09` | `8.000000e-01` | `2.305785e+00` | false |
| center_in_box | fine | 8 | `6.530612e-09` | `6.530612e-01` | `1.698600e+00` | false |
| volume_fraction | coarse | 8 | `2.420000e-09` | `2.420000e-01` | `1.146925e-16` | false |
| volume_fraction | mid | 4 | `2.420000e-09` | `2.420000e-01` | `1.146925e-16` | false |
| volume_fraction | fine | 8 | `2.420000e-09` | `2.420000e-01` | `0.000000e+00` | false |

Observed diagnostic:

- `center_in_box` misses the coarse source entirely and has large integrated
  power drift at mid / fine resolution.
- `volume_fraction` preserves the physical source volume and integrated power
  across coarse / mid / fine within the smoke tolerance.

## Solver v2 Temperature Response

| method | resolution | T range K | DeltaT range K | peak T K | residual norm | bottom error K |
|---|---:|---:|---:|---:|---:|---:|
| center_in_box | coarse | `[300.000000, 300.000000]` | `[-2.842171e-13, 2.273737e-13]` | `300.000000` | `5.582149e-17` | `0.000000e+00` |
| center_in_box | mid | `[300.000000, 302.665111]` | `[0.000000e+00, 2.665111e+00]` | `302.665111` | `6.985898e-17` | `0.000000e+00` |
| center_in_box | fine | `[300.000000, 303.194530]` | `[0.000000e+00, 3.194530e+00]` | `303.194530` | `8.834938e-17` | `0.000000e+00` |
| volume_fraction | coarse | `[300.000000, 300.339817]` | `[0.000000e+00, 3.398167e-01]` | `300.339817` | `5.538756e-17` | `0.000000e+00` |
| volume_fraction | mid | `[300.000000, 300.806196]` | `[0.000000e+00, 8.061960e-01]` | `300.806196` | `6.378037e-17` | `0.000000e+00` |
| volume_fraction | fine | `[300.000000, 301.183773]` | `[0.000000e+00, 1.183773e+00]` | `301.183773` | `8.503487e-17` | `0.000000e+00` |

All solver v2 runs produced finite temperatures, `convergence_flag = true`,
residual norms below `1e-8`, and bottom Dirichlet errors below `1e-6 K`.

## Recommendation

For the next medium-small physics-label dataset design, source assignment should
prefer region-first `volume_fraction` projection, or at minimum require a
source-power consistency check. `center_in_box` is fragile at low resolution
because it can miss small off-grid source regions and can unintentionally
change integrated source power as node count changes.

The current smoke does not compute formal PDE residuals, full energy balance,
or flux diagnostics. Those remain `not_computed` / `requires_numerical_operator`
until the reference solver and diagnostics stack expose the needed operators.
