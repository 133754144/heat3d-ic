# V4 P1 Semantic Normalization Failure Audit

Read this file only for V4 P1 semantic-normalization result review, provenance
field review, or model-lab merge decisions.

## Scope

This is a read-only audit note. It does not change training defaults, model
structure, solver, loss, loader behavior, or run artifacts.

Evidence used:

- completed `run_registry.csv` result rows for `V4Test00_baseline_seed_0` and
  `V4P1_01_baseline_normalization`;
- code facts from `rigno/heat3d_v1_normalization.py`;
- existing ignored full medium1024 audit output under
  `output/heat3d_v4_p1_full_medium1024_audit/`;
- local 16-sample proxy data only for a small transform-range sanity check,
  because the full medium1024 dataset is not present in this local worktree.

No SSH, training, tmux, evaluation launch, or new experiment queue was used.

## Result Comparison

| metric | legacy_zscore | semantic_normalization_v1 | change |
| --- | ---: | ---: | ---: |
| best `valid_base_mse` | 0.0202488992363 | 0.0319627597928 | +57.85% |
| best RMSE | 0.142298626966 | 0.178781318355 | +25.64% |
| best stress | 0.025624351576 | 0.0337334573269 | +31.65% |
| final stress | 0.0257141031325 | 0.0339051559567 | +31.85% |
| iid centered corr | 0.990555825453 | 0.988288762302 | -0.23% |
| iid amplitude ratio | 0.9984843037 | 0.993197109782 | -0.53% |
| iid top-k overlap | 0.9365234375 | 0.9279296875 | -0.00859 |
| bin0 over-ratio | 0.449183146159 | 0.583811442057 | +29.97% |
| final-probe RMSE | 0.354494850671 | 0.398472952051 | +12.41% |
| final-probe relRMSE | 0.749158090941 | 0.903179087291 | +20.56% |
| final-probe Tmax error | -3.1038021053 K | -3.39650840413 K | peak underprediction worse |

Interpretation: semantic v1 did not improve the main validation objective or
final-probe robustness. Shape correlation changed little, while stress,
low-DeltaT overprediction, and final-probe amplitude errors worsened.

## Transform Facts

`semantic_normalization_v1` is condition semantic normalization plus coordinate
provenance. It is not a coordinate-scale fix.

| feature | legacy_zscore | semantic_normalization_v1 | audit implication |
| --- | --- | --- | --- |
| coords | train min/max to `[-1, 1]` | unchanged; records extent/aspect provenance only | physical size is still not a model input |
| k | linear z-score | `log_k_zscore` | compresses material contrast; proxy max normalized k fell from about 1.41-1.82 to 1.06-1.16 |
| q | linear z-score | `signed_log1p_q_zscore` | compresses high-power magnitude cues; proxy q max fell from 13.09 sigma to 9.39 sigma |
| BC flags | z-scored binary masks | exact 0/1 passthrough | semantically cleaner, but not enough to offset k/q/top_h and target-amplitude issues |
| top_h | shared linear z-score path | independent top_h z-score | numerically unchanged in the proxy check; does not fix top_h OOD |
| target | normalized DeltaT | unchanged normalized DeltaT | raw K recovery policy is unchanged |

For medium1024 final-probe, the existing full audit already shows OOD pressure
that semantic v1 does not remove:

- k range leaves train support: final-probe min/max 0.668/423.366 vs train
  2.125/290.989;
- q max leaves train support: 194M vs train 158M;
- top_h leaves train support: 3400 vs train 1719;
- BC masks shift strongly: final-probe is all interior, while train has top
  0.1667, bottom 0.1667, side 0.4375, interior 0.375;
- geometry extent/aspect ratio is not the main OOD source;
- raw DeltaT max is far outside train: 7.7087 K vs train 0.9153 K.

## Failure Hypothesis

The weaker semantic result is most likely caused by changing magnitude-bearing
condition channels before the model/loss/target-scale plan was retuned. `log_k`
and signed `log1p(q)` make inputs numerically better behaved, but they also
compress absolute material and source-power cues that drive DeltaT amplitude.
Binary BC passthrough is probably directionally useful, yet final-probe remains
OOD in k, q, top_h, BC-mask distribution, and DeltaT amplitude.

Keep `legacy_zscore` as the V4 baseline. Treat `semantic_normalization_v1` as an
opt-in failed/diagnostic profile until ablations separate: BC flags only, k
transform only, q transform only, and target/amplitude calibration.
