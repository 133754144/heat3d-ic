# Heat3D V4 Starting Defaults (Historical)

These are the V4 entry defaults, not the frozen final baseline. The closed V4
clean baseline is `V4P5_02_clean_baseline_raw_B28_e600`; see
`docs/v4_closeout.md` for its P5 dataset, selected checkpoint, and metrics.

## Decision

V4 started from the V3 settings that were consistently useful across the
long-run and checkpoint fine-tune audits:

| Area | v4 default |
| --- | --- |
| Batch plan | `B88 sample_shuffle` |
| Optimizer | `adamw` |
| LR schedule | `warmup_cosine`, `lr=5e-4`, `warmup_epochs=10`, `min_lr=5e-5` |
| Main model | `node_latent_size=96`, `edge_latent_size=96`, `processor_steps=6`, `mlp_hidden_layers=2` |
| Graph radius | `discrete_physical_coverage` |
| Graph repair | `coverage_repair_policy=none` |
| Loss | plain `mse` unless a targeted-loss experiment explicitly opts in |
| Best selection | `valid_base_mse` |
| Long-run exports | save final/best checkpoints and predictions |
| Long-run diagnostics | post-training diagnostics and final-probe inference enabled |

The corresponding v4 starter config is:

`configs/heat3d_v2/v4_default_e600_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_warmupcosine_lr5e-4_minlr5e-5_discrete_radius_model_seed0_batchbuild0_batchorder0_graphseed0.yaml`

The retained v3 best checkpoint for reference is the completed S4 discrete
fine-tune path:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_S4discretebestFT2_e400_constant_lr5e-6_wd1e-4/params_best.pkl`

## Rationale

`B88 sample_shuffle` removed the graph-shape batch concentration issue while
keeping training compatible with the current runner. AdamW with warmup-cosine
was the stable path after the manual-GD and low-LR confusion was corrected.
The `latent96/s6/mlp2` model remained the strongest default capacity/VRAM
tradeoff; deeper decoder and processor variants are still useful diagnostics
but did not justify becoming defaults.

`discrete_physical_coverage` becomes the default graph radius policy because
v3 showed that explicit physical-node coverage is the right design direction
for 3D point clouds. `nearest_rnode` repair remains available as an explicit
control, but the default v4 path should not depend on a post-hoc repair edge
unless an experiment opts into it.

`valid_base_mse` is the default selection metric so experiments with targeted
or auxiliary loss terms do not select checkpoints by the augmented loss total.
For pure MSE runs, it is equivalent to the base training objective.

## Non-Defaults Kept For Controls

- `legacy_kdtree_mean4`: retained for regression and historical comparisons.
- `coverage_repair_policy=nearest_rnode`: retained for nearest-repair A/B and
  checkpoint comparisons.
- targeted losses, sample weighting, decoder-capacity variants, and processor
  depth variants: retained as explicit experiments only.

## Smoke Policy

Long-run defaults intentionally save predictions and run diagnostics. Smoke
configs should explicitly set:

- `save_final_predictions: false`
- `save_best_predictions: false`
- `final_probe_eval_after_training: false`
- `post_training_diagnostics: false`

This keeps smoke tests cheap while preserving the long-run audit surface.
