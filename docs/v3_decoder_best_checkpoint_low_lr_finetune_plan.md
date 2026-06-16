# Heat3D v3 D1/D2 Best-Checkpoint Low-LR Fine-Tune Plan

This is configuration preparation only. No training was started, no final-probe
labels are trained, no 4096 probe data is generated, and ignored output/data are
not committed.

## Motivation

D1-S5R and D2-S5R both selected best checkpoints early:

- D1-S5R: best epoch 596, best `valid_iid/base=0.0236985`, final
  `valid_iid/base=0.0244051`.
- D2-S5R: best epoch 673, best `valid_iid/base=0.0246530`, final
  `valid_iid/base=0.0261158`.

Their final checkpoints have some raw/shape improvements, but scalar validation
regresses after the best epoch. This suggests a possible scalar-selection and
late-schedule mismatch rather than a need for more high-LR training.

## Prepared Tests

The new tests start from the exact best checkpoint of each completed S5R run and
use a conservative constant low learning rate:

- D1-S5R-best-FT200:
  - init checkpoint:
    `output/heat3d_v2_runs/latent96_s6_mlp3_B88_sample_shuffle_nearest_repair_D1S5R_S5basebest_mse_e1600_s5schedule_wd1e-4/params_best.pkl`
  - `mlp_hidden_layers=3`
  - `epochs=200`
  - `lr=1e-5`
  - `lr_schedule=constant`
  - `checkpoint_load_strict=true`

- D2-S5R-best-FT200:
  - init checkpoint:
    `output/heat3d_v2_runs/latent96_s6_mlp4_B88_sample_shuffle_nearest_repair_D2S5R_S5basebest_mse_e1600_s5schedule_wd1e-4/params_best.pkl`
  - `mlp_hidden_layers=4`
  - same optimizer, loss, graph, batch, and strict-load policy.

Strict checkpoint load is intentional because each fine-tune config has the same
architecture as its source S5R run. If strict load fails in an actual smoke, that
should be treated as a checkpoint compatibility bug before falling back to
`checkpoint_load_strict=false`.

## Scalar-First Boundary

These configs are scalar-first diagnostics:

- `selection_metric=valid_base_mse`
- `save_final_predictions=false`
- `save_best_predictions=false`
- `post_training_diagnostics=false`
- `final_probe_eval_after_training=false`

If either run materially improves `valid_base_mse`, run posthoc prediction export
and diagnostics from the saved checkpoints. Final probe remains inference/eval
only and is not a training target.

## Configs

- `configs/heat3d_v2/v3_D1S5RbestFT_e200_adamw_latent96_s6_mlp3_B88_sample_shuffle_base_mse_constant_lr1e-5_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`
- `configs/heat3d_v2/v3_D2S5RbestFT_e200_adamw_latent96_s6_mlp4_B88_sample_shuffle_base_mse_constant_lr1e-5_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`

Both are indexed in:

`configs/heat3d_v2/v3_sample_weighting_and_decoder_test_runs.json`
