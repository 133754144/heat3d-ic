# Heat3D v3 Model Path / Decoder Preparation

This is preparation only. No decoder/model-path change is promoted here.

## Decoder Capacity Smoke Plan

Prepared smoke YAMLs:

- `configs/heat3d_v2/v3_decoder_capacity_mlp3_smoke_e002.yaml`: `mlp_hidden_layers=3`.
- `configs/heat3d_v2/v3_decoder_capacity_mlp4_smoke_e002.yaml`: `mlp_hidden_layers=4`.
- `configs/heat3d_v2/v3_decoder_capacity_steps8_smoke_e002.yaml`: `processor_steps=8`.

All smoke configs use `epochs=2`, B16 `sample_shuffle`, nearest_repair graph policy, MSE, and disable prediction export, post-training diagnostics, and final-probe inference. They are meant to check config compatibility, finite loss/grad, and obvious OOM risk only.

OOM risk order, lowest to highest: `mlp3`, `mlp4`, `steps8`, latent128. Latent128 remains optional and is not prepared as a default long-run config.

## Conditioned Normalization / Feature Scaling Audit

Current code exposes `conditioned_normalization` and `cond_norm_hidden_size` in YAML/model config and wires them into `rigno/models/rigno.py` / `rigno/models/graphnet.py`. Existing v3 configs keep `conditioned_normalization=false`. Before any formal run with this flag, run a short smoke and verify the conditioning input path is actually used by the relevant modules.

Feature scaling is currently dominated by the dataset bridge and train-only normalization: coordinates are normalized, target deltaT uses train-only normalization, and condition features come from `relative_bc_features`. No new feature scaling policy is introduced here.

## Entry Criteria For Full Runs

Do not start full runs until the corresponding smoke has passed with finite loss/grad, no OOM, and a clear reason to prefer the change over LR/checkpoint-origin controls.
