# Heat3D v1 Medium Pipeline Runbook

## Purpose

This runbook records the current reproducible Heat3D v1 medium pipeline:

```text
generate -> check -> train/export -> compare
```

The current dataset is:

```text
data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

This is a research reference / benchmark-candidate pipeline. It is not a
formal benchmark, not model-performance evidence, not OOD generalization
evidence, and not high-fidelity solver evidence.

## Generate Medium V2

Generate the ignored local medium v2 subset:

```bash
python3 tools/generate_heat3d_v1_physics_label_medium.py --write --overwrite
```

Expected output:

```text
data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2/
```

Generated `data/` must remain ignored and must not be committed.

## Check Medium V2

Run the dataset generation checker:

```bash
python3 scripts/check_heat3d_v1_physics_label_medium.py
```

This checks sample count, split count, required arrays, source-volume /
integrated-power diagnostics, solver metadata, and label diagnostics.

## Label Diagnostics

Run label diagnostics directly:

```bash
python3 scripts/check_heat3d_v1_label_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

This is label-quality smoke diagnostics only. It checks array sanity,
temperature sanity, bottom Dirichlet consistency, and solver metadata presence.
It does not claim full physics validation.

## Zero-Delta Baseline

Run zero-delta baseline comparison:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py
```

Default behavior computes only:

```text
DeltaT_pred = 0
T_pred = T_ref
```

The script prints per-sample, split-wise, and condition-wise diagnostic
summaries.

## Controlled Training Export

Run a short controlled training export:

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --epochs 5 \
  --lr 1e-5 \
  --seed 0 \
  --output-dir output/heat3d_v1_medium_runs/export_smoke_seed0 \
  --save-predictions
```

The runner uses:

- relative BC features
- zero_delta_u_bridge
- normalized DeltaT target
- train-only normalization

It writes ignored local artifacts:

```text
output/heat3d_v1_medium_runs/export_smoke_seed0/run_config.json
output/heat3d_v1_medium_runs/export_smoke_seed0/loss_summary.json
output/heat3d_v1_medium_runs/export_smoke_seed0/predictions.npz
```

No checkpoint is saved by this smoke runner.

## Trained vs Zero-Delta Comparison

Compare exported predictions against zero_delta:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --trained-predictions output/heat3d_v1_medium_runs/export_smoke_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/export_smoke_seed0/baseline_comparison.json
```

`predictions.npz` stores recovered-temperature predictions keyed by sample id.

The JSON output is an ignored local diagnostic artifact and must not be
committed.

## Do Not Commit

Do not commit:

- `data/`
- `output/`
- checkpoints
- logs
- `AGENTS.md`
- `__pycache__/`
- `*.pyc`

## Current V1 Conclusion Boundary

The current V1 medium pipeline demonstrates that the repository can reproduce a
research-reference generate/check/train/export/compare loop. It does not
establish a formal benchmark, model performance, OOD generalization, or
high-fidelity solver validity.
