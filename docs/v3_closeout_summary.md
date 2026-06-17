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

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_S4discretebestFT2_e400_constant_lr5e-6_wd1e-4/params_best.pkl`

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
| S4 mlp3 discrete e600 | pure discrete | latent96/s6/mlp3 | 541 | 0.0220820 | 4.12078e-05 | 0.0223509 | 0.0292710 | best/final params available |
| S4mlp3discretebestFT e400 | pure discrete | latent96/s6/mlp3 | 362 | 0.0216821 | 4.04529e-05 | 0.0217946 | 0.0289013 | best/final params available |

## Pending Runs Excluded From Closeout

The following runs were observed running or incomplete at closeout time and
are not used to define the v3 best checkpoint or v4 defaults:

- `S4discretebestFT3_e400_constant_lr2p5e-6` on devbox: running at closeout
  audit time; no completed `loss_summary.json` or params were available.
- `S4mlp3discretebestFT2_e400_constant_lr5e-6` on WSL2: running at closeout
  audit time; no completed `loss_summary.json` or params were available.

If either later improves on S4discretebestFT2, it should be recorded as a
post-closeout v3 artifact or a v4 initialization candidate, but it does not
block merging the v3 code and defaults.

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
   improved neither scalar validation nor stress metrics enough to justify
   defaulting away from `mlp_hidden_layers=2`.

## Merge Readiness

The v3 branch is ready to merge to `main` with the following interpretation:

- `main` should inherit v4 defaults through the runner and graph builder.
- historical legacy / nearest-repair configs remain explicit and reproducible.
- ignored `output/`, `data/`, checkpoints, predictions, and logs remain
  excluded from Git.
- v4 begins from the discrete-radius default config, while the retained v3
  best checkpoint remains an ignored runtime artifact.
