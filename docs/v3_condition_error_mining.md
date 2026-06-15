# Heat3D v3 Condition Error Mining

Purpose: prepare read-only condition and hard-sample mining before any sample-weighted training.

Script: `scripts/analyze_heat3d_v3_condition_error_mining.py`.

Inputs are existing `predictions.npz` / `best_predictions.npz` plus sample metadata. The script does not import JAX, build graphs, execute a model, or train.

Default grouping keys:

- `split`
- `source_category`
- `k_region_mode`
- `bc_category`
- `q_power_range`
- `top_h_category`
- `k_mode`

Outputs under ignored `output/heat3d_v3_condition_error_mining/`:

- `condition_error_mining.json`
- `condition_error_mining.md`
- `hard_sample_weights.json`

The hard-sample JSON is intended only as an input to smoke configs using `--sample-weight-policy hard_sample_list`. Validation metrics remain unweighted. Long sample-weighted runs require a successful 1-5 epoch smoke first.
