# V7 G1 P1i support semantics audit

The frozen P1i support is not source-amplitude-aware. Its accurate name is
**source-layout-aware block/interface/surface and CV-weighted geometry
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
The generic-uniform and volume-only variants remain registered deltas, but no
separate label-independent support artifact/provider is present in the V7
native P1i path. The historical volume-only rejection documents an outcome; it
does not define a runnable provider. Both variants remain fail-closed. No
context is also not silently substituted with a partial context removal,
because the native scale head and global context are coupled in the frozen
Full contract.

The exact machine-readable record is
`docs/v7_g1_support_semantics_audit.json`. V6 implementation and evidence
files remain read-only.
