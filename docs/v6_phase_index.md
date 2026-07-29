# V6 phase index

This index is the short route through the complete evidence retained on
`research/v6-p1h-shared-support`. It does not replace the phase reports.

## Dataset evolution

| Phase | Dataset | Decision |
|---|---|---|
| P1a | `heat3d_v6_p1a_power_calibration16_v0` | archived calibration |
| P1b | `heat3d_v6_p1b_logic_package_power_calibration16_v0` | archived calibration |
| P1c | `heat3d_v6_p1c_package_path_calibration8_v0` | archived calibration |
| P1d | `heat3d_v6_p1d_asymmetric_dual_robin1024_v0` | archived |
| P1e | `heat3d_v6_p1e_deconfounded1024_v0` | archived |
| P1f | `heat3d_v6_p1f_geometry_deconfounding_pilot_v0` | archived |
| P1g-v0 | `heat3d_v6_p1g_geometry_deconfounded1024_v0` | archived geometry-adaptive baseline |
| P1g-v1 | `heat3d_v6_p1g_geometry_deconfounded1024_v1` | retained noncanonical |
| P1h-v0 | `heat3d_v6_p1h_shared_support1024_v0` | sole canonical V6-layer dataset |

P1g→P1h keeps the 1,024 physical cases, 128 groups, split assignment, source,
power, and BC metadata fixed. It replaces sample-varying projected supports
with one ordered label-independent 1,024-node solver support and retains a
reusable full-field archive.

## Model configurations

| Config | Dataset | Decision |
|---|---|---|
| V6_01 | P1g-v0 | V4-derived historical baseline |
| V6_02 | P1g-v0 | V5-derived historical baseline |
| V6_03 | P1h-v0 | canonical; seed0 reference and seed1/2 replications |
| V6_04 | P1h-v0 | DualAttention ablation; retained but not canonical |

V6_03 seed0 point-global checkpoint e111 is frozen as the reference.

## Probe and inference decisions

- Rejected: volume-only support. It left 40 source boxes uncovered; using its
  own context produced 144.98% point-global error, while frozen anchor context
  still left 10.06%.
- Retained: Anchor-derived inference. The 1,024 source-aware anchors define
  Global Context and scale while added source-aware nodes provide the query
  support.
- Retained: `sparse_kdtree_v1` edge lists and graph cache. They remove dense
  N-by-R graph-distance materialization without changing graph semantics.
- 4096 is the default/hotspot-oriented mode.
- 8192 is the balanced full-field mode.
- 16384 is the highest IID-average full-field accuracy mode.
- 32768 remains experimental and excluded from the formal holdout/hard table.

## Performance and governance

The production closeout separates model-core, full-field production, and
evaluation timing, and reports CPU→CPU and GPU→CPU speedups against a
240,825-node CPU FVM reference. These are nonmatched-DOF comparisons. FVM
coarse/medium/reference results are a **legal structured-FVM mesh
sensitivity** study.

The opened test split is a corrected confirmatory holdout. The hard role is a
**preregistered IID stress subgroup within that already-opened corrected
confirmatory holdout**. Neither affected selection or tuning. Canonical P1h has
no registered distribution-shift OOD role.

Wrong-ladder temporary 4096/8192 results are excluded by SHA and were not used
for selection or formal reporting. The authoritative bindings are in
`configs/heat3d_v6/v6_phase_index.json` and
`configs/heat3d_v6/v6_total_governance_manifest.json`.
