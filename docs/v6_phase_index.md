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
| P1i-v1 | `heat3d_v6_p1i_continuous_physics1024_v1` | canonical role: `formal_v6_randomblock` |
| legacy random-block v2 | `heat3d_v6_randomblock_formal1024_v2` | `deprecated_engineering_history`; superseded by P1i |

P1g→P1h keeps the 1,024 physical cases, 128 groups, split assignment, source,
power, and BC metadata fixed. It replaces sample-varying projected supports
with one ordered label-independent 1,024-node solver support and retains a
reusable full-field archive.

Canonical is role-scoped rather than a single global dataset label. P1h remains
the canonical **V6-layer** dataset. P1i formal1024_v1, with its existing dataset
ID and manifest SHA `f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514`,
is the canonical **formal V6 random-block** dataset. This governance label does
not make P1i an OOD or independent benchmark. The older
`heat3d_v6_randomblock_formal1024_v2` is immutable engineering history only;
its historical manifests, results and audits remain unchanged.
The machine-readable role amendment is
`configs/heat3d_v6_p1i/v6_dataset_role_governance.json`.

## Model configurations

| Config | Dataset | Decision |
|---|---|---|
| V6_01 | P1g-v0 | V4-derived historical baseline |
| V6_02 | P1g-v0 | V5-derived historical baseline |
| V6_03 | P1h-v0 | canonical; seed0 reference and seed1/2 replications |
| V6_04 | P1h-v0 | DualAttention ablation; retained but not canonical |
| V6_06 | P1i-v1 | formal random-block seed0 reference |
| V6_07 | P1i-v1 | formal random-block seed1 replication |
| V6_08 | P1i-v1 | formal random-block seed2 replication |

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
for selection or formal reporting. This integration retains the compact
bindings in `configs/heat3d_v6/v6_phase_index.json`; the complete governance
manifest remains in the frozen research evidence archive at `d7f72f1`.

## P1i valid-only inference qualification

P1i formal1024_v1 remains frozen. Its three-seed closeout uses point-global
true-RMS relative RMSE as primary, sample-first CV-relative RMSE as secondary,
and deploys the native 1024 source-aware support through a layer-aware
240825-node reconstruction. The corrected benchmark uses 32 fixed
`valid_iid` samples on one WSL2 host and separates fresh-process cold,
JIT-cached new-case, and graph/JIT/map-cached repeated inference. Production
timing excludes oracle and metric work. The historical 4.864x value is only a
cached steady-state speedup; structured-support Route A is only an OOD
compatibility diagnostic. See `docs/v6_inference_qualification_closeout.md`.

The legacy random-block-v2 transfer and timing records remain reproducibility
evidence for an engineering diagnostic. They no longer hold a current formal
OOD or independent-benchmark role and must not be used for new selection,
tuning, production or generalization claims.

## P1i final closeout

P1i scientific development is closed. After route/checkpoint freeze,
`E16384_reconstruction` with `model_seed0` was evaluated once on the corrected
confirmatory `test_iid=128`; it was not used for selection or tuning. The final
offline peak-tail analysis uses a fixed 180 K normalization scale and records a
primarily tail-driven test peak-RMSE increase as an applicability boundary.
The separately preregistered sealed IID set remains ungenerated and unopened.
See `docs/v6_p1i_closeout.md`, `docs/v6_p1i_error_tail_closeout.md`, and
`docs/v6_p1i_handoff.md`.
