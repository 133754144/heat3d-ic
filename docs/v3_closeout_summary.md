# Heat3D v3 Closeout Summary

## Closeout Decision

v3 is closed as a graph/path/optimizer stabilization phase. The v4 starting
point is not a new model family; it is the strongest v3 default path made
explicit and reproducible:

- `B88 sample_shuffle`
- `adamw + warmup_cosine`
- `latent96 / edge96 / processor_steps=6 / mlp_hidden_layers=2`
- `discrete_physical_coverage` graph radius, with no nearest-repair post pass
- plain `mse`
- `valid_base_mse` checkpoint selection
- final/best checkpoints, predictions, post-training diagnostics, and final
  probe inference enabled for long runs

The corresponding v4 starter config is documented in
`docs/v4_starting_defaults.md`.

## Retained v3 Best Checkpoint

The retained v3 best checkpoint family is the S4 discrete-radius fine-tune
line. The current completed best checkpoint is:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_S4discretebestFT3_e400_constant_lr2p5e-6_wd1e-4/params_best.pkl`

This checkpoint is retained as the v3 scalar reference because it is the best
completed run observed in the S4 discrete-radius family and has both
`params_best.pkl` and `params_final.pkl` available on devbox.

## Completed S4 Family Results

Read-only audit source: `loss_summary.json` under ignored remote `output/`
directories on devbox / WSL2, checked on 2026-06-18. These are diagnostic
results, not publication-ready benchmarks.

| Run | Graph | Model | Best epoch | Best valid_base_mse | Best raw DeltaT MSE | Final valid_base_mse | Final stress_base_mse | Checkpoint status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S4 nearest e600 | nearest repair | latent96/s6/mlp2 | 597 | 0.0197146 | 3.67828e-05 | 0.0200590 | 0.0313623 | no params in original run |
| S4 nearest e600 checkpointed rerun | nearest repair | latent96/s6/mlp2 | 597 | 0.0198978 | 3.71352e-05 | 0.0203435 | 0.0270604 | best/final params available |
| S4 discrete e600 | pure discrete | latent96/s6/mlp2 | 587 | 0.0194904 | 3.63726e-05 | 0.0195491 | 0.0267553 | best/final params available |
| S4discretebestFT e800 | pure discrete | latent96/s6/mlp2 | 758 | 0.0184043 | 3.43421e-05 | 0.0184191 | 0.0256371 | best/final params available |
| S4discretebestFT2 e400 | pure discrete | latent96/s6/mlp2 | 392 | 0.0182252 | 3.40085e-05 | 0.0183174 | 0.0255240 | best/final params available |
| S4discretebestFT3 e400 | pure discrete | latent96/s6/mlp2 | 375 | 0.0179973 | 3.35874e-05 | 0.0180411 | 0.0254450 | best/final params available |
| S4 mlp3 discrete e600 | pure discrete | latent96/s6/mlp3 | 541 | 0.0220820 | 4.12078e-05 | 0.0223509 | 0.0292710 | best/final params available |
| S4mlp3discretebestFT e400 | pure discrete | latent96/s6/mlp3 | 362 | 0.0216821 | 4.04529e-05 | 0.0217946 | 0.0289013 | best/final params available |
| S4mlp3discretebestFT2 e400 | pure discrete | latent96/s6/mlp3 | 376 | 0.0214054 | 3.99450e-05 | 0.0215054 | 0.0287033 | best/final params available |

## Latest Follow-Up Status

Two runs that were pending during the previous closeout audit completed later
on 2026-06-18:

- `S4discretebestFT3_e400_constant_lr2p5e-6` improved the retained scalar
  reference from `0.0182252` to `0.0179973`.
- `S4mlp3discretebestFT2_e400_constant_lr5e-6` improved the mlp3 discrete
  branch from `0.0216821` to `0.0214054`, but still remained clearly weaker
  than the mlp2 discrete branch.

This update strengthens the v4 default choice: pure discrete radius remains
preferred, but `mlp_hidden_layers=2` remains the default because the mlp3
branch is still behind on scalar, stress, and final-probe metrics.

## What v3 Established

1. Graph coverage was a real failure mode. The legacy KDTree mean-4 radius can
   produce zero p2r/r2p physical-node coverage; v3 made this measurable and
   repaired it.
2. Discrete physical-node coverage is the cleanest graph default. It directly
   implements a coverage guarantee without relying on post-hoc nearest repair.
3. B88 `sample_shuffle` is the safer batch plan. It avoids the earlier graph
   shape / batch composition concentration issue while staying trainable.
4. AdamW with warmup-cosine is the stable optimizer path. Manual GD and
   low-lr early experiments are retained only as historical diagnostics.
5. Saving best/final params is required. Earlier runs without params could not
   support later checkpoint inference or final-probe audits.
6. Targeted hotspot / strong-q losses did not beat the best S4-family MSE
   checkpoints. They remain explicit experiments, not defaults.
7. Deeper decoder variants did not become the default. `mlp_hidden_layers=3`
   improved with additional low-LR fine-tuning but remained worse than
   `mlp_hidden_layers=2` on scalar validation and stress metrics.

## Merge Readiness

The v3 branch is ready to merge to `main` with the following interpretation:

- `main` should inherit v4 defaults through the runner and graph builder.
- historical legacy / nearest-repair configs remain explicit and reproducible.
- ignored `output/`, `data/`, checkpoints, predictions, and logs remain
  excluded from Git.
- v4 begins from the discrete-radius default config, while the retained v3
  best checkpoint remains an ignored runtime artifact.
