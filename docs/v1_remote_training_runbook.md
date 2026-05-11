# Heat3D v1 Remote Training Runbook

## Purpose

This runbook describes how to reproduce the current Heat3D v1 medium controlled
training export smoke on an SSH server. It is remote-ready operational guidance,
not a formal experiment protocol.

All generated `data/`, `output/`, checkpoints, and logs remain local ignored
artifacts unless a later explicit artifact-publishing policy is approved.

## Server Setup

Clone or update the repository on the server:

```bash
git clone https://github.com/133754144/heat3d-ic.git
cd heat3d-ic
git switch research/v1-physics-label-pipeline
git pull
```

Activate the project environment, for example:

```bash
conda activate <heat3d-env>
```

Use the actual server environment name. Do not encode local workstation paths in
committed config.

## Generate Local Data On Server

Generate the medium v2 subset locally on the server:

```bash
python3 tools/generate_heat3d_v1_physics_label_medium.py --write --overwrite
```

Validate it:

```bash
python3 scripts/check_heat3d_v1_physics_label_medium.py
python3 scripts/check_heat3d_v1_label_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium_v2
```

Do not commit generated `data/`.

## Run Controlled Training Export

Run the current short export smoke:

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --epochs 5 \
  --lr 1e-5 \
  --seed 0 \
  --output-dir output/heat3d_v1_medium_runs/export_smoke_seed0 \
  --save-predictions
```

For longer diagnostic runs, change `--epochs` and `--output-dir` together. Keep
outputs under ignored `output/`.

## Run Baseline Comparison

Compare trained predictions against zero_delta:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --trained-predictions output/heat3d_v1_medium_runs/export_smoke_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/export_smoke_seed0/baseline_comparison.json
```

The comparison remains diagnostic only. Candidate splits are observational and
do not support OOD claims.

## Medium256 Server Commands

Full medium256 generation and controlled training should run in the SSH
Git-only directory, not on the Mac local smoke worktree:

```bash
cd ~/myCodeGitOnly/heat3d-ic
git fetch origin
git switch research/v1-medium256-dataset
git pull --ff-only
```

Generate the full ignored medium256 subset:

```bash
python3 tools/generate_heat3d_v1_physics_label_medium.py \
  --manifest configs/heat3d_v1_physics_label_medium256_manifest.json \
  --output-subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2 \
  --write \
  --overwrite
```

Check the generated labels:

```bash
python3 scripts/check_heat3d_v1_physics_label_medium256.py \
  --manifest configs/heat3d_v1_physics_label_medium256_manifest.json \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2

python3 scripts/check_heat3d_v1_label_diagnostics.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2
```

Run a 50-epoch controlled diagnostic training export:

```bash
python3 scripts/run_heat3d_v1_medium_controlled_training_export.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2 \
  --epochs 50 \
  --lr 1e-5 \
  --seed 0 \
  --report-every 5 \
  --output-dir output/heat3d_v1_medium_runs/medium256_e050_seed0 \
  --save-predictions
```

Compare trained predictions against zero_delta:

```bash
python3 scripts/compare_heat3d_v1_medium_baselines.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_physics_label_medium256_v2 \
  --trained-predictions output/heat3d_v1_medium_runs/medium256_e050_seed0/predictions.npz \
  --output-json output/heat3d_v1_medium_runs/medium256_e050_seed0/baseline_comparison.json
```

Check ignored artifacts before any source/docs commit:

```bash
git status --short
git status --short --ignored data output AGENTS.md
```

These commands remain benchmark-candidate preparation diagnostics only; do not
claim formal benchmark status, model performance, OOD generalization, or
high-fidelity solver validity.

## Retrieve Run Artifacts

Useful ignored artifacts:

```text
output/heat3d_v1_medium_runs/<run-name>/run_config.json
output/heat3d_v1_medium_runs/<run-name>/loss_summary.json
output/heat3d_v1_medium_runs/<run-name>/predictions.npz
output/heat3d_v1_medium_runs/<run-name>/baseline_comparison.json
```

Copy them with `scp` or another approved transfer method when needed. Do not
commit them unless a future artifact policy explicitly changes.

## Git Hygiene

Before committing source/docs changes on the server, check:

```bash
git status --short
git status --short --ignored data output AGENTS.md
```

Do not commit:

- `data/`
- `output/`
- checkpoints
- logs
- `AGENTS.md`
- `__pycache__/`
- `*.pyc`

## Scaling Boundaries

Longer runs and larger datasets should remain controlled diagnostics until the
comparison protocol, artifact naming, seeds, and reporting format are fixed.
Do not describe remote runs as formal benchmarks or model-performance evidence.
