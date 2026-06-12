# Heat3D v3 Final-Target Probe Design

## Goal

Design a 10-sample OOD probe for the final target setting: arbitrary 3D IC
structure, arbitrary thermal conductivity field `k(x)`, arbitrary power
distribution `q(x)`, and boundary-condition inputs. This document only defines
the intended probe schema. It does not generate data, create pseudo-labels, run
solvers, or start training.

## Shared Constraints

- `1024` point cloud nodes per sample.
- Inputs remain coordinates `x,y,z`, `k`, `q`, and boundary-condition features.
- No layer label is required as a model input.
- Metadata must include enough tags for condition-wise diagnostics:
  `probe_family`, `stack_or_geometry_type`, `k_mode`, `k_region_mode`,
  `source_category`, `q_power_range`, `bc_category`, and explicit generator
  capability notes.
- If the current generator or solver cannot represent a sample, record it as a
  schema/generator gap rather than fabricating labels.

## Ten Probe Samples

| id | family | intended structure | intended stressor | expected metadata tags |
| --- | --- | --- | --- | --- |
| P01 | complex layered stack | six-or-more material layers with nonuniform thickness | multi-block power across separated active regions | `complex_layered_stack`, `multi_block_power` |
| P02 | complex layered stack | layer stack with thin TIM plus high-k spreader discontinuity | multi-block power with asymmetric placement | `complex_layered_stack`, `multi_block_power_asymmetric` |
| P03 | non-layered inclusion | embedded high-k inclusion inside lower-k bulk | lateral heat spreading and non-layered geometry | `non_layered_inclusion`, `high_k_inclusion` |
| P04 | non-layered inclusion | low-k void/barrier inclusion near power source | local thermal blocking and hotspot distortion | `non_layered_inclusion`, `low_k_barrier` |
| P05 | TSV/via path | vertical high-k via connecting source layer to top spreader | narrow vertical heat escape path | `vertical_high_k_path`, `tsv_like` |
| P06 | TSV/via path | multiple vias with one offset from power centroid | competing vertical paths and local routing | `vertical_high_k_path`, `multi_via_offset` |
| P07 | anisotropic k patch | patch with high in-plane, low vertical effective k | anisotropic lateral spreading | `anisotropic_k_patch`, `in_plane_high_k` |
| P08 | anisotropic k patch | patch with low in-plane, high vertical effective k | vertical channeling with lateral isolation | `anisotropic_k_patch`, `vertical_high_k` |
| P09 | extreme BC/contact | very high top HTC and small contact region | extreme cooling/contact localization | `extreme_bc_contact`, `very_high_top_h` |
| P10 | extreme BC/contact | very low top HTC plus side contact asymmetry | weak cooling and asymmetric boundary response | `extreme_bc_contact`, `very_low_top_h_side_asym` |

## Generator And Solver Gap Checklist

- Can the generator emit non-layered inclusions without relying on layer labels?
- Can `k(x)` represent anisotropic or effective tensor-like patches, or only
  scalar/diag encodings?
- Can the solver handle TSV/via-like narrow high-k vertical paths without mesh
  artifacts?
- Can boundary-condition metadata express contact area, top/bottom/side HTC
  variants, and asymmetry clearly?
- Can diagnostics recover condition tags from metadata without custom per-sample
  parsing?

## Use In v3

This probe should be introduced only after current long-run result audit and P3
mechanism diagnostics are stable. It is intended to expose whether improvements
generalize toward the final problem framing, not to replace controlled
train/valid/stress diagnostics.
