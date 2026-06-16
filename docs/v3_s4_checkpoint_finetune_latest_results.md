# Heat3D v3 S4 Checkpoint Fine-Tune Latest Results

Scope: read-only scan of devbox and WSL2 outputs on 2026-06-17. No training was
started by this audit, and no `output/`, `data/`, checkpoint, prediction, log, or
`AGENTS.md` files are committed.

## Machine State

| machine | branch | head | status | active Heat3D task |
| --- | --- | --- | --- | --- |
| devbox | `research/v3-startup-supervision` | `0b983b8` | clean | `discrete_radius S4 e600` completed |
| WSL2 | `research/v3-startup-supervision` | `ed75457` | clean | `S4mlp3bestFT2 e400` completed; stale tmux shell remains but no Heat3D training process |

Running jobs were not interrupted. No active Heat3D training process was
observed after the WSL2 smoke completed.

## Completed S4-Family Scalar Results

| run | model | best epoch | final epoch | best valid/base | final valid/base | best stress/base | final stress/base | best raw DeltaT RMSE | final raw DeltaT RMSE | checkpoint | predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| S4 original no-checkpoint | lat96/s6/mlp2 | 597 | 600 | 0.0197146 | 0.0200590 | 0.0313683 | 0.0313623 | 0.0060649 | 0.0061184 | no | best/final |
| S4 checkpointed rerun | lat96/s6/mlp2 | 597 | 600 | 0.0198978 | 0.0203435 | 0.0269293 | 0.0270604 | 0.0060939 | 0.0061611 | best/final | best/final |
| S4 discrete radius | lat96/s6/mlp2 | 587 | 600 | 0.0194904 | 0.0195491 | 0.0268483 | 0.0267553 | 0.0060310 | 0.0060395 | best/final | best/final |
| S4bestFT | lat96/s6/mlp2 | 397 | 400 | 0.0192674 | 0.0193596 | 0.0258727 | 0.0260044 | 0.0059957 | 0.0060115 | best/final | best/final |
| S4bestFT2 | lat96/s6/mlp2 | 307 | 400 | 0.0189541 | 0.0190830 | 0.0253618 | 0.0252795 | 0.0059465 | 0.0059672 | best/final | best/final |
| S4 mlp3 | lat96/s6/mlp3 | 599 | 600 | 0.0202975 | 0.0204926 | 0.0349134 | 0.0350933 | 0.0061542 | 0.0061839 | best/final | best/final |
| S4mlp3bestFT | lat96/s6/mlp3 | 368 | 400 | 0.0196004 | 0.0196983 | 0.0339995 | 0.0340384 | 0.0060480 | 0.0060636 | best/final | best/final |
| S4mlp3bestFT2 | lat96/s6/mlp3 | 397 | 400 | 0.0191695 | 0.0192581 | 0.0333086 | 0.0333961 | 0.0059806 | 0.0059939 | best/final | best/final |
| D1S5RbestFT | lat96/s6/mlp3 | 187 | 200 | 0.0240029 | 0.0241774 | 0.0368855 | 0.0370016 | 0.0066929 | 0.0067170 | best/final | no |
| D2S5RbestFT | lat96/s6/mlp4 | 3 | 200 | 0.0250621 | 0.0253261 | 0.0425532 | 0.0420662 | 0.0068386 | 0.0068739 | best/final | no |
| S5 base reference | lat96/s6/mlp2 | 1527 | 1600 | 0.0210238 | 0.0212054 | 0.0291828 | 0.0289898 | 0.0062634 | 0.0062906 | best/final | best/final |

## Mechanism And Final-Probe Summary

| run | label | mechanism RMSE | zRMSE | centered corr | top-k | peak rel | final-probe RMSE | final-probe relRMSE | P02 RMSE | P03 RMSE | P09 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4bestFT | best | 0.0031736 | 0.0810549 | 0.992133 | 0.944336 | 0.0353884 | 0.369451 | 0.797214 | 0.616422 | 0.974591 | 0.544407 |
| S4bestFT2 | best | 0.0030750 | 0.0782173 | 0.992409 | 0.948047 | 0.0344240 | 0.369570 | 0.797545 | 0.616724 | 0.974963 | 0.544109 |
| S4mlp3bestFT | best | 0.0031292 | 0.0813607 | 0.991190 | 0.942773 | 0.0353193 | 0.363882 | 0.780073 | 0.605598 | 0.971199 | 0.534423 |

P10 remains an unsupported schema gap for final-probe interpretation:
localized top contact and side asymmetry are not represented in the current
probe generator/solver schema.

## MLP Depth: Fit Quality Versus Time

| run | model | epochs | final train/base | final valid/base | epoch loop | seconds/epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S4 checkpointed rerun | s6/mlp2 | 600 | 0.0013228 | 0.0203435 | 10297.35s | 17.16 |
| S4bestFT | s6/mlp2 | 400 | 0.0010030 | 0.0193596 | 6955.78s | 17.39 |
| S4bestFT2 | s6/mlp2 | 400 | 0.0007843 | 0.0190830 | 6581.20s | 16.45 |
| S4 mlp3 | s6/mlp3 | 600 | 0.0009799 | 0.0204926 | 13677.12s | 22.80 |
| S4mlp3bestFT | s6/mlp3 | 400 | 0.0006971 | 0.0196983 | 9128.82s | 22.82 |
| S4mlp3bestFT2 | s6/mlp3 | 400 | 0.0005688 | 0.0192581 | 9155.47s | 22.89 |

The data supports the hypothesis that increasing MLP depth lowers training loss
but slows training. In the comparable low-lr fine-tune chain, mlp3 reaches lower
final train/base than mlp2 (`0.0005688` versus `0.0007843`), while each epoch is
about 39% slower on the observed machines (`22.9s` versus `16.5s`). Validation
does not improve proportionally: mlp3 FT2 (`0.0191695` best valid/base) remains
slightly weaker than mlp2 S4bestFT2 (`0.0189541`).

## Discrete-Radius Smoke

The requested `mlp_hidden_layers=3` plus pure `discrete_physical_coverage`
smoke was run on WSL2 for 5 epochs. It completed without OOM:

| run | graph policy | model | epochs | best/final valid/base | best/final stress/base | checkpoints | predictions | status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| S4 mlp3 discrete-radius smoke | `discrete_physical_coverage`, repair `none` | lat96/s6/mlp3 | 5 | 0.488275 / 0.488275 | 0.536169 / 0.536169 | best/final | disabled | finite, no OOM |

This is only an OOM/compatibility result. The high loss is expected for a
5-epoch from-scratch smoke and should not be used as a model-quality conclusion.
The smoke does show that the B88 mlp3 discrete-radius graph can build,
initialize, run forward/backward/update, and save checkpoints in the current
environment.

## Conclusions

1. `S4bestFT2` is the strongest completed scalar/stress checkpoint so far:
   best valid/base `0.0189541`, best stress/base `0.0253618`, and saved
   best/final params.
2. The S4 checkpointed rerun is slightly worse than the original no-checkpoint
   S4 on scalar, but substantially better on stress and is usable as a
   checkpoint source.
3. The two-stage S4 low-LR continuation is effective: S4 checkpointed rerun
   best `0.0198978` -> S4bestFT best `0.0192674` -> S4bestFT2 best
   `0.0189541`.
4. `S4mlp3bestFT2` improves the mlp3 scalar path further (`0.0196004` ->
   `0.0191695`) and selects best at epoch 397/400, so a lower-lr continuation is
   justified. A prepared `S4mlp3bestFT3` config uses `lr=5e-6` from the FT2 best
   checkpoint.
5. D1/D2 S5R best-checkpoint low-LR fine-tunes do not recover competitiveness.
   D1 improves only to `0.0240029`; D2 best is early at epoch 3 and then
   degrades. Do not prioritize wider decoder MLP continuation.
6. S4 discrete radius e600 completed and is strong (`0.0194904` best valid/base,
   `0.0268483` best stress/base), but still trails S4bestFT2. Keep it as a
   serious graph-policy control, not the current scalar leader.
7. The `mlp_hidden_layers=3` plus pure `discrete_physical_coverage` 5-epoch
   smoke passed without OOM, so a longer run is feasible from a compatibility
   standpoint. It should still wait for an explicit long-run decision because
   discrete radius is costlier and the mlp3 nearest-repair chain is not yet a
   scalar leader.

## Recommended Next Use

- Promote `S4bestFT2` best checkpoint as the current scalar/stress checkpoint
  candidate for downstream checkpoint-based experiments.
- Keep `S5 base`, `S5final FT`, and `D3-L200` as comparison baselines because
  they exercise different model-path hypotheses.
- Run `S4mlp3bestFT3` only if continuing the mlp3 scalar path is still useful
  after reviewing S4bestFT2 as the scalar/stress checkpoint.
- If testing mlp3 plus discrete radius further, start with an e50 or e100
  diagnostic rather than jumping directly to e600.
- Do not treat the 5-epoch smoke as performance evidence; use it only as the
  compatibility/OOM gate.
- Prepare `S4discretebestFT` as a direct e800 low-lr continuation from the
  completed S4 discrete-radius best checkpoint. This mirrors the S4bestFT
  pattern but uses one e800 run to match the earlier two-by-e400 fine-tune
  budget: strict params load, constant `lr=1e-5`, pure discrete radius, full
  prediction export, post-training diagnostics, and final-probe inference. It is
  prepared as config only in this audit and was not started.
