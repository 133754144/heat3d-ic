# Heat3D v1 Main Merge Readiness Checklist

## Scope

This checklist is for evaluating whether the current Heat3D v1 research branch
is ready to prepare a future merge proposal. It is not itself a merge request
and does not authorize pushing to `main`.

The current V1 status remains research reference / benchmark candidate only.

## Protected Code Paths

- [ ] v0 public entrypoints are unchanged:
  - `scripts/inspect_heat3d_dataset.py`
  - `scripts/check_heat3d_batch_graphs.py`
  - `scripts/train_heat3d_operator.py`
  - `scripts/evaluate_heat3d_operator.py`
- [ ] `rigno/models/*` is unchanged.
- [ ] Any v1 additions are additive and v1-only.

## Forbidden Files

- [ ] No `data/` files are staged or committed.
- [ ] No `output/` files are staged or committed.
- [ ] No checkpoints are staged or committed.
- [ ] No logs are staged or committed.
- [ ] `AGENTS.md` remains local ignored guidance.
- [ ] No `__pycache__/` or `*.pyc` files are staged or committed.

## Medium V2 Pipeline Checks

- [ ] Medium v2 generation checker passes:

```bash
python3 scripts/check_heat3d_v1_physics_label_medium.py
```

- [ ] Label diagnostics pass:

```bash
python3 scripts/check_heat3d_v1_label_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

- [ ] Zero-delta baseline comparison passes:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py
```

- [ ] Controlled training export smoke passes:

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --epochs 5 \
  --lr 1e-5 \
  --seed 0 \
  --output-dir output/heat3d_v1_medium_runs/final_smoke_seed0 \
  --save-predictions
```

- [ ] Trained-vs-zero_delta comparison smoke passes:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --trained-predictions output/heat3d_v1_medium_runs/final_smoke_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/final_smoke_seed0/baseline_comparison.json
```

## Legacy Default Smoke Regression

- [ ] Old default zero_delta bridge smoke still passes:

```bash
python3 scripts/check_heat3d_v1_zero_delta_bridge.py
```

- [ ] Old default validation metrics smoke still passes:

```bash
python3 scripts/check_heat3d_v1_validation_metrics_smoke.py
```

## Documentation Boundaries

- [ ] Docs do not claim formal benchmark status.
- [ ] Docs do not claim model performance.
- [ ] Docs do not claim OOD generalization.
- [ ] Docs do not claim high-fidelity solver status.
- [ ] Candidate test splits remain diagnostic only.

## Final Git Checks

- [ ] `git diff --check` passes.
- [ ] `git status --short` is clean after commit.
- [ ] `git status --short --ignored data output AGENTS.md` shows ignored local artifacts only.

## Merge Readiness Interpretation

Passing this checklist means the V1 research pipeline is organized enough to
prepare a future review or merge discussion. It does not mean the branch is a
formal benchmark, a production training pipeline, or validated physics-label
dataset.
