# V5-P0 Experiment Contract

This contract covers only `V5-P0-0` and `V5-P0-1` on `research/v5`.
It establishes the pre-experiment physics-scale baseline for the frozen P5
dataset; it does not authorize model changes, data generation, evaluation, or
training.

The machine-readable source is
`configs/heat3d_v5/v5_p0_experiment_contract.json`.

## Data And Split Scope

- Dataset: `heat3d_v4_p5_clean_nohard_v0` at
  `data/heat3d_v4_p5_clean_nohard_v0`.
- Split map:
  `configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`.
- Roles: train `672`, valid_iid `128`, test_iid `128`, hard_train_holdout
  `121`, hard_challenge_valid `12`, and hard_challenge_test `12`.
- Every audit must cover all 1,073 samples and retain those role boundaries.

## Read-Only Boundary

The P0 audit may read only the P5 arrays and `sample_meta.json`. It may write
only its tracked JSON and Markdown reports. It must not write `data/`,
`output/`, `checkpoints/`, or `logs/`; call a solver; generate data; modify a
model; or start training/evaluation.

## Required Measurements

- Derive rectilinear control-volume weights from `coords.npy`, and compare
  their total with the per-sample solver control-volume audit.
- Calculate effective source power as `sum(q * CV volume)` and compare it with
  the recorded q integral and target power.
- Use `temperature - bottom fixed temperature` to report CV-weighted RMS and
  mean DeltaT plus nodewise maximum DeltaT.
- Report descriptive physics-scale proxies: harmonic through-plane
  conductivity, top Robin and through-plane resistance proxies, their
  source-power DeltaT scales, and target-to-proxy ratio.
- Report descriptive q/BC linear statistics for source power, `top_h`, and the
  target CV mean. These are not causal claims across heterogeneous scenes.
- Reject continuation past P0 if any sample is unassigned or if exact
  model-input, full-sample, or P5-provenance fingerprints cross split roles.

## Reproducibility

Run the fixture check first:

```bash
python3 -B scripts/check_heat3d_v5_p0_physics_scale_audit.py
```

Then, on a server after `conda activate rigno`, run the real audit only where
the P5 dataset is available:

```bash
python scripts/audit_heat3d_v5_p5_physics_scale.py \
  --dataset data/heat3d_v4_p5_clean_nohard_v0 \
  --split-map configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json \
  --contract configs/heat3d_v5/v5_p0_experiment_contract.json \
  --output-json configs/heat3d_v5/v5_p0_1_p5_physics_scale_audit.json \
  --output-md docs/v5_p0_1_p5_physics_scale_audit.md
```

Use `--dry-run` with the same input arguments before the real audit. It reads
only the dataset directory and split map and creates no output files.
