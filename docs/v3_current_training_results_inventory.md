# Heat3D v3 Current Training Results Inventory

Scope: read-only inventory of current training results found on devbox and WSL2
under `output/heat3d_v2_runs/`. No training was started for this inventory, no
output artifacts are committed, and `data/`, `output/`, and `AGENTS.md` remain
ignored.

Scan summary:

- devbox: 175 `loss_summary.json` files found.
- WSL2: 16 `loss_summary.json` files found.
- Active Heat3D training processes in the latest 2026-06-17 follow-up scan:
  devbox `discrete_radius S4 e600` and WSL2 `S4mlp3bestFT2 e400`.
- Table ranking key metric: best `valid_iid/base` where available.
- `ckpt=BF` means both `params_best.pkl` and `params_final.pkl` exist.
- `pred=BF` means both `best_predictions.npz` and `predictions.npz` exist.

Latest S4 checkpoint/fine-tune update:

- The S4 no-checkpoint row below is no longer the best usable checkpoint path.
- `S4bestFT2` is now the strongest completed scalar/stress checkpoint:
  best valid/base `0.0189541`, best stress/base `0.0253618`, final valid/base
  `0.0190830`, final stress/base `0.0252795`.
- Detailed S4-family results are in
  `docs/v3_s4_checkpoint_finetune_latest_results.md`.

## Primary V3 Comparison

| rank | run | server | best iid/base | best stress/base | final iid/base | final stress/base | best/final epoch | schedule | model | ckpt | pred | verdict |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | S4 `seed0 e600 warmupcosine lr5e-4` | devbox/wsl2 | 0.0197146 | 0.0313683 | 0.0200590 | 0.0313623 | 597/600 | warmup_cosine | lat96/s6/mlp2 | no | BF | Best scalar row, but no params checkpoint. Use as scalar reference only. |
| 2 | B6 `seed0 e400 warmupcosine lr5e-4` | devbox | 0.0203209 | 0.0306349 | 0.0206497 | 0.0308075 | 397/400 | warmup_cosine | lat96/s6/mlp2 | no | BF | Strong pre-checkpoint baseline. |
| 3 | S5 base `seed0 e1600 warmupcosine lr5e-4` | devbox/wsl2 | 0.0210238 | 0.0291828 | 0.0212054 | 0.0289898 | 1527/1600 | warmup_cosine | lat96/s6/mlp2 | BF | BF | Best complete checkpointed baseline. |
| 4 | D3-L200 `steps8 e200` | WSL2 | 0.0210456 | 0.0282934 | 0.0210686 | 0.0282471 | 129/200 | warmup_cosine lr3e-5 | lat96/s8/mlp2 | BF | BF | Best model-path variant; processor depth is plausible. |
| 5 | E2 `S5finalFT final hotspot-only lr3e-5 e100` | devbox | 0.0211202 | 0.0289176 | 0.0211641 | 0.0289022 | 3/100 | constant | lat96/s6/mlp2 | BF | BF | Good LR-escape targeted run; no breakthrough over S5. |
| 6 | A hotspot-only `lr1e-5 e100` | devbox/wsl2 | 0.0211280 | 0.0288827 | 0.0212803 | 0.0288583 | 86/100 | constant | lat96/s6/mlp2 | BF | BF | Cleanest targeted-loss diagnostic. |
| 7 | P4 hotspot+strong-q `lr1e-5 e100` | devbox | 0.0211334 | 0.0288591 | 0.0212424 | 0.0289112 | 83/100 | constant | lat96/s6/mlp2 | BF | BF | Useful targeted-loss evidence, not a new baseline. |
| 8 | W2 sample-weight + hotspot-only `e50` | WSL2 | 0.0211385 | 0.0289665 | 0.0212949 | 0.0288870 | 1/50 | constant | lat96/s6/mlp2 | BF | no | Sample weighting did not beat S5/P4. |
| 9 | B hotspot0.05 strongq0.025 `e100` | devbox | 0.0211461 | 0.0289132 | 0.0212645 | 0.0288992 | 71/100 | constant | lat96/s6/mlp2 | BF | BF | Strong-q is not a clear gain source. |
| 10 | S5final FT no-mask `e100` | devbox/wsl2 | 0.0211927 | 0.0289011 | 0.0213502 | 0.0289516 | 4/100 | constant | lat96/s6/mlp2 | BF | BF | Best no-mask fine-tune fallback checkpoint. |
| 11 | S5final EM `edge-mask0.02 e100` | devbox | 0.0211989 | 0.0289025 | 0.0213441 | 0.0289441 | 4/100 | constant | lat96/s6/mlp2 | BF | BF | Edge mask is neutral/slightly positive, not decisive. |
| 12 | S5best FT no-mask `e100` | devbox/wsl2 | 0.0212501 | 0.0291303 | 0.0213915 | 0.0290771 | 32/100 | constant | lat96/s6/mlp2 | BF | BF | Starting from S5 best is weaker than S5final FT. |
| 13 | W1 sample-weight `S5basebest e50` | devbox | 0.0212604 | 0.0291750 | 0.0213829 | 0.0291066 | 3/50 | constant | lat96/s6/mlp2 | BF | no | Hard-sample weighting at 1.25 is not a main path. |
| 14 | E6 true rapid-decay `S5basebest e100` | devbox | 0.0212710 | 0.0292713 | 0.0213901 | 0.0291213 | 5/100 | rapid_decay | lat96/s6/mlp2 | BF | BF | No evidence that rapid decay helps. |
| 15 | S3 `seed0 e1200 warmupcosine lr1e-3` | WSL2 | 0.0218745 | 0.0386520 | 0.0231263 | 0.0383088 | 596/1200 | warmup_cosine | lat96/s6/mlp2 | no | BF | Better than S2, but stress is weak. |
| 16 | discrete-radius seed0 `e400` | devbox | 0.0230111 | 0.0325364 | 0.0230614 | 0.0325327 | 374/400 | warmup_cosine | lat96/s6/mlp2 | no | BF | Useful graph-policy control; not better than S5. |
| 17 | nearest-repair seed0 `e400` | devbox | 0.0230306 | 0.0339808 | 0.0230806 | 0.0339621 | 361/400 | warmup_cosine | lat96/s6/mlp2 | no | BF | Earlier graph-policy run before S5 schedule. |
| 18 | D1-S5R `mlp3 e1600 S5 schedule` | WSL2 | 0.0236985 | 0.0376156 | 0.0244051 | 0.0374426 | 596/1600 | warmup_cosine | lat96/s6/mlp3 | BF | BF | S5 schedule rescues D1-L400 but still not competitive. |
| 19 | D2-S5R `mlp4 e1600 S5 schedule` | devbox | 0.0246530 | 0.0430413 | 0.0261158 | 0.0422316 | 673/1600 | warmup_cosine | lat96/s6/mlp4 | BF | BF | Worse than D1-S5R; do not continue wider decoder MLP. |
| 20 | S2 `seed0 e1200 constant lr1e-3` | WSL2 | 0.0259946 | 0.0454090 | 0.0270849 | 0.0459213 | 1018/1200 | constant | lat96/s6/mlp2 | no | BF | Constant lr=1e-3 is weak for seed0. |
| 21 | W1 `seed1 e1200 warmup-flat lr1e-3` | WSL2 | 0.109367 | 0.114138 | 0.114535 | 0.117784 | 1199/1200 | upstream_onecycle | lat96/s6/mlp2 | no | BF | Seed1 warmup-flat improves early path but fails final scalar quality. |
| 22 | D1-L400 `mlp3 e400 weak lr` | devbox | 0.101528 | 0.132198 | 0.101952 | 0.133271 | 394/400 | warmup_cosine lr3e-5 | lat96/s6/mlp3 | BF | BF | Negative control for weak partial-load schedule. |
| 23 | legacy graph `B88 e400` | devbox | 0.209451 | 0.286436 | 0.210072 | 0.282725 | 312/400 | warmup_cosine | lat96/s6/mlp2 | no | BF | Confirms graph repair is essential. |
| 24 | D1 e50 `mlp3` | devbox | 0.281075 | 0.364172 | 0.281075 | 0.364172 | 50/50 | constant lr1e-5 | lat96/s6/mlp3 | BF | no | Under-converged decoder-depth smoke. |
| 25 | D2 e50 `mlp4` | devbox | 0.296554 | 0.380429 | 0.296554 | 0.380429 | 50/50 | constant lr1e-5 | lat96/s6/mlp4 | BF | no | Under-converged decoder-depth smoke. |

## Older V1/V2 And Negative-Control Groups

| group | representative best iid/base | interpretation |
| --- | ---: | --- |
| v2 B48/B64/B96 stratified baselines | about 0.20-0.24 | Older v2 model/input path is far behind v3 repaired graph + S5 schedule. |
| B96 latent96 seed-instability controls | about 0.036 to 1.00 | Confirms strong seed/path sensitivity before B88 sample-shuffle/S5 schedule. |
| Seed1 LR sweep failures A/B/C/D/G/U | about 0.30-0.98 for many runs | Most seed1 optimizer/schedule variants remain poor; do not treat them as contenders. |
| single/4/16 memorization and timing smokes | often 0.45-2.37 | Smoke/profiling evidence only, not comparable model-quality results. |

## Current Ranking Conclusions

1. Best completed scalar/stress checkpoint: S4bestFT2 best (`0.0189541`
   valid_iid/base, `0.0253618` stress/base), with saved best/final params and
   predictions.
2. Scalar reference only: original S4 best (`0.0197146` valid_iid/base), but it
   lacks saved params and should not be used as a checkpoint source.
3. Best older complete checkpoint baseline before the S4 checkpoint/fine-tune
   chain: S5 base best (`0.0210238`) with saved best/final params and
   predictions.
4. Best model-path variant: D3-L200 (`processor_steps=8`) is essentially tied
   with S5 on scalar and has the best stress value among complete checkpointed
   model-path runs (`0.0282471` final stress/base).
5. Targeted-loss and LR-escape runs (P4/A/B/E2/W2/E6) produce small local
   movements around S5 but no clear breakthrough. Keep P4/A/E2 as diagnostics,
   not new baselines.
6. Decoder MLP-depth expansion is not promising: D1-S5R/D2-S5R improve over
   weak low-lr D1/D2 tests, but remain worse than S5/D3 in scalar, stress, and
   final-probe means.
7. Graph repair remains mandatory: legacy graph is around `0.209` best
   valid_iid/base, an order of magnitude worse than S5-family runs.

## Recommended Comparison Set

Use this compact set when comparing future configurations:

- Current scalar/stress checkpoint: S4bestFT2 best.
- Scalar reference only: original S4 best.
- Older complete checkpoint baselines: S5 base best/final, S5final FT no-mask
  final, P4 best, A hotspot-only best/final.
- Model-path candidate: D3-L200 best/final.
- Negative controls: legacy graph, D1-L400, D1-S5R, D2-S5R, W1 seed1,
  S2 constant lr, and v2 B48/B96 baselines.
