# Heat3D v3 D1/D2 S5-Schedule Long Plan

This is configuration preparation only. No training was started, no final-probe
labels are trained, no 4096 probe data is generated, and ignored output/data are
not committed.

## S5 Schedule Source

The D1-S5R and D2-S5R configs copy the S5 baseline schedule from:

`configs/heat3d_v2/frozen_v1_e1600_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_warmup_cosine_lr5e-4_minlr5e-5_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`

The local ignored S5 `run_config.json` was not present in this worktree, so the
repo-tracked S5 YAML is the authoritative source for this preparation. The
copied schedule fields are:

- `epochs=1600`
- `optimizer.name=adamw`
- `lr=5e-4`
- `lr_schedule=warmup_cosine`
- `warmup_epochs=10`
- `min_lr=5e-5`
- `weight_decay=1e-4`
- `gradient_clip_norm=1.0`
- B88 `sample_shuffle`, `batch_build_seed=0`, `batch_order_seed=0`, `graph_seed=0`
- nearest-repair graph policy
- MSE objective

## Why D1-S5R / D2-S5R Exist

D1-L400 failed with `mlp_hidden_layers=3`, `lr=3e-5`, `min_lr=1e-6`, and
`e400`, but that does not fully rule out D1/D2 capacity. D1/D2 introduce new
decoder MLP parameters, and the weaker fine-tune schedule may not be enough to
recover the S5 checkpoint behavior after matching partial-load.

The new configs are S5-schedule retraining-style continuations from S5 best:

- D1-S5R: `mlp_hidden_layers=3`, S5 schedule, S5 best checkpoint,
  `checkpoint_load_strict=false`, `partial_load_policy=matching`.
- D2-S5R: `mlp_hidden_layers=4`, same setup.

These are not from-scratch runs. They are params-only warm-start configs from:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5_seed0_e1600_warmupcosine_lr5e-4_minlr5e-5_wd1e-4/params_best.pkl`

## Export / Diagnostics

Both configs enable:

- `save_final_predictions=true`
- `save_best_predictions=true`
- `post_training_diagnostics=true`
- `final_probe_eval_after_training=true`
- `final_probe_checkpoint_kind=both`
- `selection_metric=valid_base_mse`

Final probe remains checkpoint inference/evaluation only. It is not trained and
is not used as a training target.

## Prepared Configs

- `configs/heat3d_v2/v3_D1S5R_s5basebest_e1600_adamw_latent96_s6_mlp3_B88_sample_shuffle_base_mse_s5schedule_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`
- `configs/heat3d_v2/v3_D2S5R_s5basebest_e1600_adamw_latent96_s6_mlp4_B88_sample_shuffle_base_mse_s5schedule_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`

Both are indexed in:

`configs/heat3d_v2/v3_sample_weighting_and_decoder_test_runs.json`
