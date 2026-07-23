# V6_04 P1h DualAttention resolved-config diff

Status: **passed**. The sole scientific difference from
`V6_03_V5best_P1h` is `model.shape_attention_mode: none -> physics_gate`.
Scale attention remains `physics_gate`.

## Resolved leaf differences

| path | V6_03 | V6_04 |
|---|---|---|
| `config_id` | `V6_03_V5best_P1h` | `V6_04_V5best_P1h_DualAttention` |
| `description` | `V6 canonical-candidate P1h shared-support transfer of V6_02_V5best. The only scientific variable is the dataset binding; P1g remains the global canonical V6-layer dataset and no prior checkpoint is loaded.` | `V6 P1h shared-support single-variable DualAttention ablation of V6_03_V5best_P1h. Shape attention changes from none to physics_gate while scale attention remains physics_gate; all other scientific settings remain frozen.` |
| `export.output_dir` | `output/heat3d_v6_runs/V6_03_V5best_P1h` | `output/heat3d_v6_runs/V6_04_V5best_P1h_DualAttention` |
| `export.run_name` | `V6_03_V5best_P1h` | `V6_04_V5best_P1h_DualAttention` |
| `metadata.ablation_parent_config_id` | `None` | `V6_03_V5best_P1h` |
| `metadata.ablation_scientific_difference` | `None` | `model.shape_attention_mode:none_to_physics_gate` |
| `metadata.experiment_role` | `None` | `single_variable_dual_attention_ablation` |
| `metadata.log_path` | `output/heat3d_v6_logs/V6_03_V5best_P1h.log` | `output/heat3d_v6_logs/V6_04_V5best_P1h_DualAttention.log` |
| `model.shape_attention_mode` | `none` | `physics_gate` |

## Frozen invariants

- scientific diff paths: `['model.shape_attention_mode']`
- dataset / graph / loss / optimizer / LR / seed / B24 / e600: unchanged
- runtime graph backend/hash: `cpu` / `6d3d62830755872194766aad2a8ac7b0f1fabec57840dac78fcb2642a6ed771c`
- V6_03/V6_04 runtime graph equal: `True`
- train+valid materialized; test target materialized: false
- training or optimizer update executed: false / false

## Manual command

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_04_V5best_P1h_DualAttention.yaml
```
