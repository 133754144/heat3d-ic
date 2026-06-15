# Heat3D v3 Model Path / Decoder Preparation

This is preparation only. No decoder/model-path change is promoted here.

## Decoder Capacity Smoke Plan

Prepared smoke YAMLs:

- `configs/heat3d_v2/v3_decoder_capacity_mlp3_smoke_e002.yaml`: `mlp_hidden_layers=3`.
- `configs/heat3d_v2/v3_decoder_capacity_mlp4_smoke_e002.yaml`: `mlp_hidden_layers=4`.
- `configs/heat3d_v2/v3_decoder_capacity_steps8_smoke_e002.yaml`: `processor_steps=8`.

All smoke configs use `epochs=2`, B16 `sample_shuffle`, nearest_repair graph policy, MSE, and disable prediction export, post-training diagnostics, and final-probe inference. They are meant to check config compatibility, finite loss/grad, and obvious OOM risk only.

OOM risk order, lowest to highest: `mlp3`, `mlp4`, `steps8`, latent128. Latent128 remains optional and is not prepared as a default long-run config.

WSL2 smoke results:

| smoke | result | epoch_loop | status |
| --- | --- | ---: | --- |
| `mlp3` | passed | 331.11s | finite grad, no OOM |
| `mlp4` | passed | 374.33s | finite grad, no OOM |
| `steps8` | passed | 335.48s | finite grad, no OOM |

Prepared e50 long-test configs after these smokes:

- `D1`: S5 base best + MSE + `mlp_hidden_layers=3`.
- `D2`: S5 base best + MSE + `mlp_hidden_layers=4`.
- `D3`: S5 base best + MSE + `processor_steps=8`.

These configs use params-only warm-start from S5 base best with `checkpoint_load_strict=false` and `partial_load_policy=matching`, because their parameter trees differ from the S5 checkpoint. They keep final-probe inference, post-training diagnostics, and prediction export disabled.

## Conditioned Normalization / Feature Scaling Audit

Current code exposes `conditioned_normalization` and `cond_norm_hidden_size` in YAML/model config and the model classes support them. However, the v2 runner command builder and runner CLI do not currently pass these fields from YAML into `_model_config_from_args`; the runner inherits the `MODEL_CONFIG` defaults, where `conditioned_normalization=false`. For current v2 controlled runs this is config-only / no-op unless runner wiring is added and smoked. No formal conditioned-normalization training should be prepared from YAML alone.

Feature scaling is currently dominated by the dataset bridge and train-only normalization: coordinates are normalized, target deltaT uses train-only normalization, and condition features come from `relative_bc_features`. No new feature scaling policy is introduced here.

## Entry Criteria For Full Runs

Do not start full runs until the corresponding smoke has passed with finite loss/grad, no OOM, and a clear reason to prefer the change over LR/checkpoint-origin controls.
