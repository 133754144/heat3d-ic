# V8 MASS-HBM geometry/topology audit

Date: 2026-09-19
Status: `PHASE1_READONLY_AUDIT_COMPLETE`

This report keeps paper statements, raw-handoff observations, and engineering inferences separate.  No MASS-HBM source file was modified, moved, or copied into Git.  The companion JSON is `docs/v8_mass_hbm_geometry_topology_audit.json`.

## PAPER FACT

- Fig. 3 represents package, die/BEOL/bonding, material, and interface scales explicitly; it does not state that every physical domain is a filled rectangular cuboid.
- Fig. 8 distinguishes 2.5D, 3D-P, and 3D-V geometry.
- Fig. 9 starts from structure/interface-aware meshing and then adapts the mesh using the evolving temperature field, retaining resolution around hotspots and coarsening low-gradient regions.
- Fig. 10 shows architecture-dependent heat-flow paths.
- Fig. 11 places cooling channels between vertical memory groups.  These are internal thermal sinks, not merely a top or bottom package boundary.
- The geometry-optimization study varies at least pitch, slab count/thickness, and group count.

The paper therefore does **not** justify a regular-cuboid-only V8 contract.  Fig. 9 also establishes a possible `T -> final mesh` leakage path unless the exported grid is proven to be a pre-solve mesh.

## RAW FACT

The read-only scan covered all 314 exported cases.

- Every exported `k` cell is finite and positive, and all `k/q/T` arrays align with the full declared `shape_zyx`.  The current handoff therefore uses every stored array cell as a computational control volume.
- No per-cell inactive/void mask is exported.  The current numerical domain is a full axis-aligned structured bounding box, with nonuniform `z` thickness and uniform `x/y` spacing.
- This does not mean all cells are active devices.  Component, filler/mold, coolant, GPU and HBM logic occupy masked subregions inside that box.  `lc_region_mask_audit.csv` contains aggregate counts, not a reconstructable cell mask.
- Exported external faces lie on `x/y/z` extrema and have normals in `±x/±y/±z`.
- 232 LC-V cases contain nonempty internal-sink audit records.  The handoff gives aggregate physical-sink cell count, conductance, and sink temperature but no per-cell indices, face areas, normals, or connectivity.  Internal sink topology is therefore not reconstructable.
- The selected non-LC smoke case has 59,150 cells (`14×65×65`), a full computational mask, 12,090 exterior faces incident on 11,522 cells, and no unresolved internal sink.

### Target-independent geometry fingerprint

The fingerprint hashes only numeric grid/cell geometry, component placement topology, material/interface topology, boundary/cooling topology, and geometry design parameters.  It explicitly excludes temperature, final `k/q/Rint`, hotspot/refinement fields, workload/model/phase, architecture labels, and all case identifiers including `geometry_case_id`.

| Quantity | Result |
|---|---:|
| Cases | 314 |
| Unique geometry fingerprints | 74 |
| Cases per geometry, min/q25/median/q75/max | 1 / 1 / 1 / 9 / 23 |
| Singleton geometries | 42 |
| `14×130×130` cases | 260 |
| Unique geometry within those 260 cases | 54 |
| Architecture families (metadata only) | 4 |

The observed cases-per-geometry histogram is: `1:42, 2:1, 3:7, 4:1, 5:1, 6:1, 7:1, 9:11, 10:2, 11:1, 12:2, 16:2, 18:1, 23:1`.  A fingerprint is a split-group key only; none of its categorical topology names is passed to the model.

## INFERENCE / DECISION

### Why the P1i 11D boundary schema is invalid for V8

`is_top/is_bottom/is_side/is_interior` describes a particular axis-aligned cuboid.  It cannot preserve arbitrary normals, nonrectangular exterior surfaces, multiple sink temperatures, or internal cooling surfaces.  Numerical structured storage is not a physics guarantee and must not freeze V8 to P1i semantics.

The implemented candidate-A local schema is continuous and label-free:

1. cell physics: `kx, ky, kz, q, control_volume`;
2. boundary aggregates: area/volume, `G/volume`, `G*(T_sink-T_ref)/volume`, prescribed-flux power/volume, three first normal moments and six symmetric second normal moments, with `G=hA`;
3. interface aggregates kept in a separate guarded channel.

Robin and internal sink faces share `Q_face = G_face (T - T_sink)`; no `is_liquid_cooling` categorical input is used.

Candidate A is intentionally lossy: aggregation cannot recover the number and separation of incident faces, face-to-face sink-temperature distribution, detailed channel connectivity, or interface pairing.  It is sufficient for a non-LC engineering smoke only.  Candidate B is the formal design direction: boundary/sink virtual nodes connected by face edges carrying `(A, n_x,n_y,n_z,G,T_sink-T_ref,q''A)`, and paired material cells connected by guarded interface edges.  Candidate B was specified but not implemented in this phase.

### Split protocol (not frozen)

- Keep every geometry fingerprint wholly within one split.
- Keep all iterations/repeats/states of a solver case together.
- Stratify training/validation/test only after grouping, using design variables for coverage rather than random sample rows.
- Reserve explicit OOD suites for unseen pitch, slab count/thickness, group geometry, cooling design, workload/power state, and cross-architecture transfer.
- Because 42 groups are singletons and some geometry groups contain up to 23 states, formal split ratios must be selected after a group-level coverage table is reviewed; no formal manifest is frozen here.

### Verdict

- Current exported non-LC numerical cell domain: `FULL_BOX_STRUCTURED_RAW_FACT`.
- Physical component/cooling topology: `MASKED_SUBREGIONS_RAW_FACT`.
- LC internal boundary reconstruction: `FAIL_CLOSED_MISSING_PER_CELL_OR_FACE_TOPOLOGY`.
- V7/P1i adapter-only compatibility: `NO_GO`.
- Independent V8 generic-boundary schema: `CONDITIONAL_GO / MODEL_SCHEMA_CHANGE_GO`.

## Author questions that remain genuinely unresolved

1. Is `solver_geometry.json` the initial/pre-solve mesh, the final temperature-adapted mesh, or a separate reduced-order grid that was never temperature-adapted?
2. Please provide the per-cell/per-face LC sink map, face area/normals, sink temperature and conductance for each internal cooling surface.
3. Please provide an explicit pre-solve region/material map (including void/filler/coolant semantics) if it exists upstream of the handoff.
