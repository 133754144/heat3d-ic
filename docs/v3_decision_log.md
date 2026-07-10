# Heat3D v3 Decision Log

## Current State

- Best scalar validation reference: B6 best.
- Best raw mechanism reference: S3 final.
- Main unresolved question: whether B6 e400 is undertrained or whether the
  scalar/raw metric mismatch reflects objective or decoder-path behavior.

## S4 / B6-e600

S4 extends B6 from e400 to e600 with the same model, graph policy, seed,
batch plan, optimizer, learning rate, warmup cosine schedule, and min LR.
The only intended experimental variable is the epoch count.

Purpose:

- test whether B6 e400 stops before its scalar and raw mechanism metrics have
  converged;
- compare B6-e600 against B6-e400, S2, and S3 before launching new schedule
  exploration;
- keep this as a diagnostic run, not a formal benchmark.

## Hold Decisions

- Do not start P5 pointwise/local decoder work yet.
- Do not start P7 loss/objective changes yet.
- Wait for S4 plus paired per-sample mismatch evidence before deciding whether
  the next move is more training, decoder/path audit, or objective alignment.

## Closeout Update

v3 closeout promotes the S4 discrete-radius line to the retained v3 best
checkpoint family. The strongest completed checkpoint observed at closeout is:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_S4discretebestFT2_e400_constant_lr5e-6_wd1e-4/params_best.pkl`

This supersedes the older B6/S4-nearest reference for v4 initialization and
default-setting discussions. `S4discretebestFT3` and
`S4mlp3discretebestFT2` were still incomplete during the closeout audit, so
they are tracked as pending artifacts rather than merge blockers.

## Latest Result Update

`S4discretebestFT3` completed after the initial closeout and is now the
strongest retained v3 scalar checkpoint:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_discrete_radius_S4discretebestFT3_e400_constant_lr2p5e-6_wd1e-4/params_best.pkl`

It improves best `valid_base_mse` to `0.0179973`. The corresponding mlp3
follow-up, `S4mlp3discretebestFT2`, improved the mlp3 branch to `0.0214054`
but remains weaker than mlp2, so v4 keeps `mlp_hidden_layers=2` as default.
