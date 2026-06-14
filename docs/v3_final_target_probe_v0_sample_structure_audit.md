# Heat3D v3 Final-Target Probe v0 Sample Structure Audit

`v3_final_target_probe_v0` is a 10-sample, 1024-point diagnostic probe set for
Heat3D v3 final-target review. It is not a formal benchmark, not a
publication-ready validation set, and not evidence of final model performance.

The probe is intended to stress qualitative structure handling in the current
RIGNO path: material discontinuities, non-layered blocks, sparse conductive
routes, anisotropic diag3 patches, heat-source topology, and extreme but still
V1-compatible boundary settings.

## Global Sample Space

- Resolution: 1024 points per sample.
- Grid shape: `16 x 16 x 4`.
- Unit: meter.
- Domain bounds:
  - `x = [0.0, 0.01]`
  - `y = [0.0, 0.01]`
  - `z = [0.0, 0.002]`
- Paired 4096 version: deferred.
- Model inputs remain pure physics: coordinates, k field, q field, and BC
  features. Layer, region, material IDs are metadata only.

## Probe Summary

| probe | intended stressor | k_mode | k_region_mode | source_category | bc_category | high-k | low-k | strong-q | weak-q bg | S5 RMSE | S5 Tmax err | v0 status |
|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| P01 | non_layered_conductivity_routing | iso1 | high_low_k_mix | multi_block_power | nominal_top_h | 3 | 1 | 3 | false | 0.247039 | -1.99567 | accepted diagnostic |
| P02 | disconnected_conduction_paths | iso1 | sparse_high_k_bridge | compact_hotspot_with_weak_background | nominal_top_h | 2 | 2 | 1 | true | 0.631598 | -5.93617 | accepted diagnostic |
| P03 | local_hotspot_confinement | iso1 | low_k_barrier | contained_hotspot | low_top_h | 2 | 1 | 1 | false | 0.980923 | -7.08919 | accepted diagnostic |
| P04 | multi_scale_material_discontinuity | iso1 | multi_scale_interface | multi_block_power | high_top_h | 3 | 2 | 3 | false | 0.395817 | -3.13041 | accepted diagnostic |
| P05 | non_ic_source_topology | iso1 | random_block_background | multi_blob_power | nominal_top_h | 1 | 3 | 3 | false | 0.229521 | -1.14794 | accepted diagnostic |
| P06 | anisotropic_power_distribution | iso1 | random_block_background | elongated_power | nominal_top_h | 2 | 2 | 1 | true | 0.206724 | -1.12625 | accepted diagnostic |
| P07 | vertical_heat_escape_amid_random_materials | iso1 | tsv_like_high_k_path | via_adjacent_hotspot | high_top_h | 4 | 3 | 1 | false | 0.128922 | -1.76410 | accepted diagnostic |
| P08 | ic_source_pattern_off_manifold_material_context | iso1 | random_block_background | active_hotspot_motif | nominal_top_h | 3 | 1 | 2 | false | 0.210765 | -2.74908 | accepted diagnostic |
| P09 | diag3_tensor_like_spreading_mismatch | diag3 | localized_diag3_anisotropic_patch | patch_adjacent_hotspot | nominal_top_h | 2 | 2 | 1 | false | 0.547794 | -5.76051 | accepted diagnostic |
| P10 | V1_extreme_top_h_boundary_extrapolation | iso1 | random_block_background | compact_hotspot | very_high_top_h_candidate | 1 | 2 | 1 | false | 0.148068 | -1.70461 | accepted diagnostic |

## P01

Design target: non-layered high/low-k material routing with multiple heat-source
blocks.

Actual generated structure: `high_low_k_mix`, with three high-k components, one
low-k component, and three strong-q components in a 16 x 16 x 4 grid.

K distribution: background effective k is `32.0`, with effective k range
`2.0691` to `163.6543`. High-k threshold is `40.0`; low-k threshold is `24.0`.

Q distribution: no weak full-domain background; max q is `8.25e7`; strong-q
threshold is `1.65e7`.

BC setting: nominal top Robin h, bottom fixed temperature.

S5 smoke result: RMSE `0.247039`, Tmax error `-1.99567`. The model underestimates
the hottest local structure but keeps moderate average error.

Limitations: axis-aligned components are simplified diagnostic geometry, not a
full continuous material description.

## P02

Design target: disconnected conductive paths plus a compact hotspot on weak
background heating.

Actual generated structure: `sparse_high_k_bridge`, with two high-k components,
two low-k components, one compact strong-q component, and weak q across the full
domain.

K distribution: background effective k is `32.0`; effective k range is
`3.8665` to `230.0`.

Q distribution: weak full-domain background exists. Background q is `4.0e6`,
max q is `1.94e8`, and strong-q threshold is `4.2e7`. Strong/background ratio
is `48.5`.

BC setting: nominal top Robin h, bottom fixed temperature.

S5 smoke result: RMSE `0.631598`, Tmax error `-5.93617`. This is one of the
harder probes; the model underestimates the compact high-power peak.

Limitations: weak background q is included to stress source separation; it
should not be interpreted as a red full-domain source in schematic figures.

## P03

Design target: local hotspot confinement by a low-k barrier.

Actual generated structure: `low_k_barrier`, with one dominant low-k barrier
component and one contained hotspot. The audit records the strong-q bbox as
inside/intersecting the low-k bbox.

K distribution: background effective k is `32.0`; effective k range is `1.25`
to `158.8154`.

Q distribution: no weak background. Max q is `1.55e8`; strong-q threshold is
`3.1e7`.

BC setting: low top h, bottom fixed temperature.

S5 smoke result: RMSE `0.980923`, Tmax error `-7.08919`. This is the worst S5
probe by RMSE. The dominant failure mode is peak underestimation for a confined
hotspot near/inside the low-k barrier.

Limitations: the barrier is represented by grid-aligned components; it is still
accepted as the v0 diagnostic for local confinement.

## P04

Design target: multi-scale material discontinuity with multiple power blocks.

Actual generated structure: `multi_scale_interface`, with three high-k
components, two low-k components, and three strong-q components.

K distribution: background effective k is `32.0`; effective k range is `2.5` to
`165.0`.

Q distribution: no weak background. Max q is `1.21e8`; strong-q threshold is
`2.42e7`.

BC setting: high top h, bottom fixed temperature.

S5 smoke result: RMSE `0.395817`, Tmax error `-3.13041`.

Limitations: v0 uses coarse 1024-point geometry; fine interface curvature is out
of scope.

## P05

Design target: off-manifold source topology in random material blocks.

Actual generated structure: random-block background, one high-k component,
three low-k components, and three strong-q source components.

K distribution: background effective k is `32.0`; effective k range is `3.1764`
to `156.7941`.

Q distribution: no weak background. Max q is `8.25e7`; strong-q threshold is
`1.65e7`.

BC setting: nominal top Robin h.

S5 smoke result: RMSE `0.229521`, Tmax error `-1.14794`.

Limitations: this sample is diagnostic source topology only, not a calibrated IC
layout.

## P06

Design target: elongated power distribution on a weak full-domain background.

Actual generated structure: random-block background with two high-k components,
two low-k components, one elongated strong-q component, and weak q across the
domain.

K distribution: background effective k is `32.0`; effective k range is `5.3190`
to `167.6990`.

Q distribution: weak full-domain background exists. Background q is `2.0e6`,
max q is `7.7e7`, strong-q threshold is `1.7e7`, and strong/background ratio is
`38.5`.

BC setting: nominal top Robin h.

S5 smoke result: RMSE `0.206724`, Tmax error `-1.12625`.

Limitations: weak background q is only a diagnostic stressor and should be
drawn faintly or omitted from overview schematics.

## P07

Design target: TSV-like vertical heat escape amid random materials.

Actual generated structure: `tsv_like_high_k_path`, with four high-k components,
three low-k components, and one via-adjacent strong-q component. The audit
detects a vertical high-k component with z-span fraction `1.0`.

K distribution: background effective k is `32.0`; effective k range is `4.6098`
to `260.0`.

Q distribution: no weak background. Max q is `1.55e8`; strong-q threshold is
`3.1e7`.

BC setting: high top h.

S5 smoke result: RMSE `0.128922`, Tmax error `-1.76410`. This probe is one of
the easier cases for the current S5 checkpoint.

Limitations: the TSV-like path is a simplified high-k vertical component, not a
full via process stack.

## P08

Design target: IC-like active hotspot motif in off-manifold random material
context.

Actual generated structure: random-block background, three high-k components,
one low-k component, and two strong-q components.

K distribution: background effective k is `32.0`; effective k range is `6.2484`
to `136.1660`.

Q distribution: no weak background. Max q is `1.94e8`; strong-q threshold is
`3.88e7`.

BC setting: nominal top Robin h.

S5 smoke result: RMSE `0.210765`, Tmax error `-2.74908`.

Limitations: IC motif is intentionally simplified and metadata-only; no layer
label is used by the model input.

## P09

Design target: diag3 anisotropic spreading mismatch.

Actual generated structure: localized diag3 anisotropic patch with one
anisotropic component, two high-k components, two low-k components, and one
patch-adjacent strong-q component.

K distribution: `k_mode = diag3`. Effective background k is `32.0`; effective k
range is `3.0380` to `179.4897`. Anisotropy ratio max is `14.5455`, with mean
`kx/ky/kz = 39.0523 / 33.6563 / 32.7463`.

Q distribution: no weak background. Max q is `7.5e7`; strong-q threshold is
`1.5e7`.

BC setting: nominal top Robin h.

S5 smoke result: RMSE `0.547794`, Tmax error `-5.76051`. The model underestimates
the patch-adjacent peak.

Limitations: P09 is diag3 anisotropic patch only. It is not full tensor-k and
does not include off-diagonal conductivity terms.

## P10

Design target: V1-compatible extreme top boundary extrapolation.

Actual generated structure: random-block background, one high-k component, two
low-k components, and one compact strong-q component.

K distribution: background effective k is `32.0`; effective k range is `3.9246`
to `134.9453`.

Q distribution: no weak background. Max q is `1.1e8`; strong-q threshold is
`2.2e7`.

BC setting: V1 global top Robin `very_high_top_h`, bottom fixed temperature.

S5 smoke result: RMSE `0.148068`, Tmax error `-1.70461`.

Limitations: P10 only implements V1 global top Robin very-high h. Localized top
contact and side asymmetry remain unsupported and must not be claimed.

## Compliance Table

| probe | accepted as v0 diagnostic probe | formal benchmark | publication-ready validation |
|---|---|---|---|
| P01 | yes | no | no |
| P02 | yes | no | no |
| P03 | yes | no | no |
| P04 | yes | no | no |
| P05 | yes | no | no |
| P06 | yes | no | no |
| P07 | yes | no | no |
| P08 | yes | no | no |
| P09 | yes | no | no |
| P10 | yes | no | no |

