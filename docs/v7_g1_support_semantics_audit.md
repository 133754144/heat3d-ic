# V7 G1 P1i support semantics audit

The frozen P1i support is not source-amplitude-aware. Its accurate name is
**physics-layout-aware q/k-block/interface/surface/CV-weighted sparse
support**. The historical `local_regions` field is an alias for the 256-node
`block` quota; it is not a learned or label-derived region.

The registered `retry_deterministic_geometry_only_v1` path is implemented by
the historical `validate_layout` → `select_group_support` chain in
`scripts/heat3d_v6_p1i_continuous_core.py`. It receives q/k block layout masks,
coordinates, layer boundaries, control-volume weights, a group identifier and
a deterministic seed. It first makes one deterministic pick per q/k layout
block, completes the block quota with CV-weighted choices, then selects
interface, top, bottom and remaining volume strata. The frozen counts are
512/256/128/64/64 for volume/block/interface/top/bottom.

For the native 1024 support selector, numeric q values are not used. The
selector also does not read temperature, labels, solver output or model error.
Numeric q values are materialized later by `build_case_fields`; the high-N
`scripts/heat3d_v6_p1i_anchor_query.py` path may use q values to order added
query nodes, but that is a different query-selection role and must not be
used to rename the native support.

## Claim and ablation boundary

H2 is therefore frozen as “source-layout-aware sparse conditioning.” A claim
about source-amplitude-aware conditioning is prohibited without new evidence.
The registered support-attribution deltas are precise and label independent:

* **Layout-agnostic stratified support** (`layout_agnostic_stratified_v1`)
  uses the fixed quotas block=0, interface=128, top=64, bottom=64 and
  CV-weighted interior volume=768. It deliberately does not receive q/k
  layout masks.
* **CV-only support** (`cv_only_v1`) samples 1024 CV-weighted interior-volume
  nodes; block/interface/top/bottom quotas are all zero.

Both providers consume only the frozen 240825-node shared geometry, layer
boundaries, control volumes, sample ID and registered run seed. Generated
support hashes are qualification artifacts, not historical V6 evidence.
No-FiLM changes only `global_context_mode: film -> none`; the standardized
24-D physical context, native shape-scale path and scale-context path remain.
Physics-scale-only retains the physics scale and disables only the learned
residual correction; it is not a direct-output architecture.

The exact machine-readable record is
`docs/v7_g1_support_semantics_audit.json`. V6 implementation and evidence
files remain read-only.
