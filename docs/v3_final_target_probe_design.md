# Heat3D v3 Final-Target Probe Design

## Goal

Design a 10-sample OOD probe for the final target setting: arbitrary 3D IC
structure, arbitrary thermal conductivity field `k(x)`, arbitrary power
distribution `q(x)`, and boundary-condition inputs.

This is a design document only. It does not generate data, create pseudo-labels,
run solvers, or start training.

## Design Principle

Use a random-block-first probe rather than a stack-first probe. IC-like motifs
can appear inside the random-block background, but the main stressor should be
general 3D material and source structure rather than layer-template
interpolation.

## Resolution Plan

- `1024` points: compatibility resolution for current RIGNO/B88 diagnostic
  tooling.
- `4096` points: main probe resolution for final-target evaluation once graph
  and memory behavior are stable.
- `8192` points: optional stress resolution for later scalability checks only.

The paired 1024/4096 versions should share the same semantic scene and metadata
where possible.

## Shared Constraints

- Inputs remain coordinates `x,y,z`, `k`, `q`, and boundary-condition features.
- No layer label is required as a model input.
- Metadata must include enough tags for condition-wise diagnostics:
  `probe_family`, `geometry_type`, `k_mode`, `k_region_mode`,
  `source_category`, `q_power_range`, `bc_category`, `resolution`, and explicit
  generator capability notes.
- If the current generator or solver cannot represent a sample, record it as a
  schema/generator gap rather than fabricating labels.

## Ten Probe Samples

| id | family | intended structure | intended stressor | expected metadata tags |
| --- | --- | --- | --- | --- |
| P01 | random material-block composite | many rectangular high/low-k blocks in full 3D volume | non-layered conductivity routing | `random_block_composite`, `high_low_k_mix` |
| P02 | random material-block composite | sparse high-k bridges through low-k background | disconnected conduction paths | `random_block_composite`, `sparse_high_k_bridge` |
| P03 | random material-block composite | dense low-k barriers around source region | local hotspot confinement | `random_block_composite`, `low_k_barrier` |
| P04 | random material-block composite | mixed block sizes with high contrast interfaces | multi-scale material discontinuity | `random_block_composite`, `multi_scale_interface` |
| P05 | random volumetric heat source | several random volumetric heat blobs | non-IC source topology | `random_volumetric_source`, `multi_blob_power` |
| P06 | random volumetric heat source | elongated heat source plus weak background heating | anisotropic power distribution | `random_volumetric_source`, `elongated_power` |
| P07 | IC motif in random-block background | TSV/via-like vertical high-k path embedded in random blocks | vertical heat escape amid random materials | `ic_motif_random_background`, `tsv_like_path` |
| P08 | IC motif in random-block background | active-die hotspot motif embedded in non-layered background | IC source pattern with off-manifold material context | `ic_motif_random_background`, `active_hotspot_motif` |
| P09 | anisotropic/tensor-k patch | localized anisotropic k patch inside random block volume | tensor-like spreading mismatch | `anisotropic_tensor_k_patch`, `localized_anisotropy` |
| P10 | extreme BC/contact | small-area top contact plus side asymmetry | boundary/contact extrapolation | `extreme_bc_contact`, `localized_contact_asymmetry` |

## Generator And Solver Gap Checklist

- Can the generator produce random 3D material blocks independent of layer
  labels?
- Can paired 1024/4096 samples preserve semantic scene identity?
- Can `k(x)` represent anisotropic or tensor-like patches, or only scalar/diag
  encodings?
- Can the solver handle high-contrast random material blocks without mesh
  artifacts?
- Can boundary metadata express contact area, top/bottom/side HTC variants, and
  asymmetry clearly?
- Can diagnostics recover all condition tags from metadata without custom
  per-sample parsing?

## Use In v3

Introduce this probe only after current long-run result audit and P3 mechanism
diagnostics are stable. It should test whether improvements move toward the
final problem framing, not replace controlled train/valid/stress diagnostics.
