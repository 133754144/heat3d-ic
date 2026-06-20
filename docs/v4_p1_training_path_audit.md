# V4 P1 Training Path Audit

Read this file only for V4 active training path, feature-manifest,
normalization, or input/target semantics questions.

For full medium1024 range/OOD and final-probe amplitude conclusions, use
`docs/v4_p1_full_medium1024_audit.md`; its full audit supersedes the proxy
16-sample range snapshot below.

## Scope

This is a control-plane audit only. It did not change model structure, solver,
loader behavior, training logic, registry semantics, or YAML control fields. It
did not train, SSH, or tmux.

The local range audit used the V3 worktree's 16-sample
`v1_multilayer_bc_eq_supervised_small` subset as a proxy plus the local V3
`v3_final_target_probe_v0` subset. The active V4 baseline still resolves to the
server-side medium1024 Gap-A subset.

## Active Path

The active V4 baseline path is:

```text
configs/heat3d_v4/v4_run_registry.json
-> configs/heat3d_v4/generated/V4_baseline.yaml
-> scripts/prepare_heat3d_v4_run.py / scripts/run_heat3d_v4_config.py
-> rigno/heat3d_v2_runner_command.py
-> scripts/run_heat3d_v1_medium_controlled_training_export.py
-> Heat3DV1NativeSupervisedDataset
-> Heat3DV1SupervisedDataset
-> Heat3DV1MetadataDataset
-> relative_bc_features + zero_delta_u_bridge
-> Inputs(u, c, x_inp, x_out)
-> RIGNO.apply(...)
-> normalized DeltaT MSE target
```

So V4 does not enter a separate new V4 runner. It uses the V4 registry/YAML
control plane to launch the current V1 controlled training/export runner.

The V0 legacy concern is partially resolved: the active path is not the old
`dataset_Heat3D.py` loader. It is the V1 native supervised wrapper, but it still
bridges into the legacy `Inputs` API before model application.

## Input Manifest

Current native task semantics remain:

```text
coords + k(x) + q(x) + BC -> T(x)
```

Boundary conditions are actual model inputs in the active path.

`x_inp` and `x_out` are the same physical-node coordinates from `coords.npy`,
in meters before normalization. The runner normalizes them with train-only
coordinate min/span to `[-1, 1]`.

`u` is not temperature and not a conductivity channel. Under
`zero_delta_u_bridge`, `Inputs.u` is an all-zero delta-temperature field and is
not z-scored.

`c` carries the physical condition channels that actually enter the model:

```text
k_x, k_y, k_z,
q,
is_top, is_bottom, is_side, is_interior,
top_h,
top_T_inf_minus_T_ref,
bottom_T_fixed_minus_T_ref
```

`k`, `q`, and BC channels therefore do reach `RIGNO.apply(...)` through
`Inputs.c`. The model concatenates `u` and `c` into physical-node features
before encoder processing.

`layer_id`, `region_id`, and `material_id` remain metadata for dataset
generation or evaluation grouping. They are not packed into `Inputs` and are
not current model-input features.

## Target And Recovery

The raw supervised label is `temperature.npy`, i.e. raw steady temperature
`T(x)`.

The active bridge derives:

```text
T_ref = bottom fixed temperature when available, else top ambient, else 300 K
target_deltaT = T(x) - T_ref
target_normalized = (target_deltaT - train_deltaT_mean) / train_deltaT_std
```

The loss target is normalized DeltaT. Recovery is:

```text
DeltaT_pred = pred_normalized * train_deltaT_std + train_deltaT_mean
T_pred = T_ref + DeltaT_pred
```

Default checkpoint selection is still `valid_base_mse` in normalized validation
space, not raw DeltaT error.

## Normalization

Confirmed code path:

- `coords`: train-only min/span normalization to `[-1, 1]`.
- `u`: zero field, no z-score.
- `c`: train per-feature z-score for all condition channels.
- `target`: train scalar DeltaT mean/std.
- BC flags: included in `c`, so they are z-scored as continuous features.

V4 risks to address later:

- `k` and `q` use linear z-score only; there is no log or physical-scale
  transform.
- BC flags become continuous normalized values.
- Coordinate min/max normalization can hide physical extent and aspect-ratio
  shifts unless raw geometry diagnostics are preserved.
- Target `T` vs DeltaT semantics are clear in code but should be recorded as
  structured run artifacts.
- Final-probe amplitude shifts need explicit scale-ratio diagnostics.

## Range/OOD Snapshot

This snapshot is from the local 16-sample proxy, not the full V4 server
medium1024 subset.

Proxy train range:

| field | train range |
| --- | --- |
| `k` | 2.0 to 210.0 |
| `q` | 0.0 to 1.50e8 |
| `top_h` | 1000.0 to 3000.0 |
| z extent | 0.00275 to 0.003 m |
| aspect ratio | 3.33 to 3.64 |
| raw `T` | 300.0 to 351.69 K |
| raw DeltaT | 0.0 to 1.93 K |

Proxy `valid` stayed within train ranges in this small audit.

Local final-probe was outside proxy train on:

- material/k: 0.668 to 423.37, outside 2.0 to 210.0;
- q: max 1.94e8, above train max 1.50e8;
- BC/top_h: 450 to 3400, outside 1000 to 3000;
- geometry: z extent 0.002 m and aspect ratio 5.0, outside proxy train;
- amplitude: raw DeltaT max 7.71 K, above proxy train max 1.93 K.

This supports treating final-probe as stress/OOD diagnostics, not default
checkpoint selection.

## Artifact Gaps

Current artifacts are close but not complete enough for paper-grade provenance:

- `run_config.json` records the route as prose, but not structured
  `target_mode`, `bridge_policy`, `feature_view`, or `normalization_profile`.
- `train_only_normalization` records feature names and c/target stats but omits
  `coord_min` and `coord_span`.
- `loss_summary.json` lacks an input feature-manifest hash or sample-manifest
  hash.
- `run_registry.csv` has metrics result columns but not
  `result_target_mode`, `result_normalization_profile`, or
  `result_bridge_policy`.

These are recommended artifact fields only. This audit does not add new
training-control YAML fields.

## Audit Outputs

Generated by:

```bash
python3 scripts/audit_heat3d_v4_p1_training_path.py \
  --subset "/Users/xuyihua/.codex/worktrees/f2dc/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small" \
  --final-probe-subset "/Users/xuyihua/.codex/worktrees/f2dc/3D IC Heat/data/heat3d-thermal-simulation/subsets/v3_final_target_probe_v0"
```

Small ignored outputs:

- `output/heat3d_v4_p1_audit/training_path_audit.json`
- `output/heat3d_v4_p1_audit/feature_manifest.json`
- `output/heat3d_v4_p1_audit/feature_manifest.csv`
