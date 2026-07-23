# V6_03 P1h resolved-config diff

Status: **passed**. `V6_03_V5best_P1h` resolves from
`V6_02_V5best`; the dataset binding is the only scientific variable.
P1g-v0 remains the sole global canonical dataset and P1h-v0 is a
`canonical_candidate`.

## Resolved leaf differences

| path | V6_02 | V6_03 |
|---|---|---|
| `config_id` | `V6_02_V5best` | `V6_03_V5best_P1h` |
| `dataset.manifest_path` | `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json` | `configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json` |
| `dataset.name` | `heat3d_v6_p1g_geometry_deconfounded1024_v0` | `heat3d_v6_p1h_shared_support1024_v0` |
| `dataset.subset_path` | `data/heat3d_v6_p1g_geometry_deconfounded1024_v0` | `data/heat3d_v6_p1h_shared_support1024_v0` |
| `description` | `V6 canonical P1g-v0 random-init transfer of V4P5_42_canonical through the dual-Robin runtime adapter. No V5 checkpoint is loaded.` | `V6 canonical-candidate P1h shared-support transfer of V6_02_V5best. The only scientific variable is the dataset binding; P1g remains the global canonical V6-layer dataset and no prior checkpoint is loaded.` |
| `export.output_dir` | `output/heat3d_v6_runs/V6_02_V5best` | `output/heat3d_v6_runs/V6_03_V5best_P1h` |
| `export.run_name` | `V6_02_V5best` | `V6_03_V5best_P1h` |
| `metadata.candidate_dataset_id` | `None` | `heat3d_v6_p1h_shared_support1024_v0` |
| `metadata.dataset_lifecycle_status` | `None` | `canonical_candidate` |
| `metadata.execution_host` | `devbox` | `None` |
| `metadata.launch_timestamp_utc` | `2026-07-22T19:29:32Z` | `None` |
| `metadata.log_path` | `output/heat3d_v6_logs/V6_02_V5best.log` | `output/heat3d_v6_logs/V6_03_V5best_P1h.log` |
| `metadata.runner_pid` | `389637` | `None` |
| `metadata.training_commit` | `ec72010250fcd210ae29c9d2dc48371de8b057c3` | `None` |
| `metadata.training_started` | `True` | `False` |

## Frozen scientific invariants

- model / graph / loss / optimizer / LR schedule: exactly equal
- epochs / effective batch / micro batch: 600 / 24 / 24
- forward-backward / optimizer updates per epoch: 32 / 32
- epoch-wise batch regrouping: false
- checkpoint selection: `valid_rel_rmse_v4_pct`
- train / valid: 768 / 128; test target materialized: false
- shared coordinates / graph: one / one
- formal training or optimizer update executed by this checker: false / false

## Manual command

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_03_V5best_P1h.yaml
```
