# Heat3D v3 Graph Policy Hardening and P2-a Smoke

## Scope

This note records the v3 P1 hardening / compatibility review and P2-a small
training smoke. It does not change model, decoder, loss, optimizer semantics, or
training objective. It does not run 16-sample or full-dataset training.

Inputs used locally:

- subset: `/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small`
- hardening output: `output/heat3d_v3_p2_policy_smoke/hardening.json`
- P2-a output: `output/heat3d_v3_p2_policy_smoke/p2_policy_small_training_smoke.json`

## Policies Checked

| label | radius_policy | coverage_repair_policy | repair_p2r | repair_r2p |
| --- | --- | --- | --- | --- |
| legacy | `legacy_kdtree_mean4` | `none` | true | true |
| nearest_repair | `legacy_kdtree_mean4` | `nearest_rnode` | true | true |
| discrete_radius | `discrete_physical_coverage` | `none` | true | true |

The explicit legacy configuration is equivalent to the default
`Heat3DGraphBuilder()` path.

## P1 Hardening Result

Command:

```bash
python3 scripts/check_heat3d_v3_graph_policy_hardening.py \
  --subset "/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small"
```

Checks passed:

- default builder and explicit legacy builder metadata / graph leaves are equal.
- legacy P0 current synthetic summary still matches the previous fixed baseline.
- legacy, nearest repair, and discrete radius all build metadata, build graphs,
  and complete model forward on real supervised-small batches.
- output shape and finite checks pass.
- edge indices stay in bounds.
- dummy pnode/rnode are not used by real graph edges.
- nearest repair and discrete radius keep r2r topology stable against legacy.
- existing v1 supervised batch forward smoke still passes on `sample_000` and
  `sample_005`.

Hardening real-data coverage summary on two train samples:

| policy | p2r_zero | r2p_zero | p2r_edges | r2p_edges | edge_ratio p2r/r2p |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 116 | 116 | 248 | 244 | 1.000 / 1.000 |
| nearest_repair | 0 | 0 | 364 | 360 | 1.468 / 1.475 |
| discrete_radius | 0 | 0 | 650 | 650 | 2.621 / 2.664 |

## P2-a Small Training Smoke

Command:

```bash
python3 scripts/run_heat3d_v3_p2_policy_small_training_smoke.py \
  --subset "/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small" \
  --output-dir output/heat3d_v3_p2_policy_smoke
```

Settings:

- 1-sample: `sample_000`, 20 epochs.
- 4-sample: `sample_000` to `sample_003`, 10 epochs.
- learning rate: `1e-5`.
- train-only smoke, no checkpoint.

| policy | sample_count | initial_loss | final_loss | loss_drop | raw DeltaT RMSE | raw DeltaT MAE | edge_ratio p2r/r2p | finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 1 | 1.215097e+00 | 1.168791e+00 | 4.630554e-02 | 2.124636e-01 | 1.640642e-01 | 1.000 / 1.000 | true |
| nearest_repair | 1 | 1.269997e+00 | 1.224875e+00 | 4.512191e-02 | 2.175013e-01 | 1.706150e-01 | 1.468 / 1.475 | true |
| discrete_radius | 1 | 1.287800e+00 | 1.266819e+00 | 2.098167e-02 | 2.211940e-01 | 1.735060e-01 | 2.621 / 2.664 | true |
| legacy | 4 | 1.200144e+00 | 1.182209e+00 | 1.793504e-02 | 2.585070e-01 | 1.949293e-01 | 1.000 / 1.000 | true |
| nearest_repair | 4 | 1.269032e+00 | 1.250916e+00 | 1.811600e-02 | 2.658380e-01 | 2.051855e-01 | 1.500 / 1.507 | true |
| discrete_radius | 4 | 1.269994e+00 | 1.263574e+00 | 6.420612e-03 | 2.672567e-01 | 2.076024e-01 | 2.768 / 2.806 | true |

Coverage during P2-a:

| sample_count | policy | p2r_zero | r2p_zero | p2r_edges | r2p_edges |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | legacy | 58 | 58 | 124 | 122 |
| 1 | nearest_repair | 0 | 0 | 182 | 180 |
| 1 | discrete_radius | 0 | 0 | 325 | 325 |
| 4 | legacy | 222 | 222 | 444 | 438 |
| 4 | nearest_repair | 0 | 0 | 666 | 660 |
| 4 | discrete_radius | 0 | 0 | 1229 | 1229 |

## Interpretation

P1 implementation is compatible with the legacy path and satisfies the formal
coverage gate for the tested small samples. The nearest repair policy gives the
lowest added edge cost among the two repair candidates and is therefore the
safer P2 candidate from a graph-budget perspective.

The P2-a smoke does not show an immediate loss/RMSE advantage for nearest repair
or discrete radius. All policies learn downward, but legacy has the lowest final
loss and RMSE in this short train-only smoke, while nearest repair has a very
similar loss drop with higher absolute loss. This means zero coverage remains a
clear graph correctness problem, but this smoke alone is not enough evidence
that repairing zero coverage is the dominant one-sample or four-sample learning
bottleneck.

Discrete radius also passes zero-coverage and forward gates, but its edge cost
is roughly 2.6x to 2.8x legacy in these checks and its short-smoke loss drop is
smaller. It should remain a research candidate, not the default P2 path.

## Recommendation

For P2-b, use nearest repair as the primary candidate because it guarantees no
p2r/r2p zero coverage with a smaller edge increase than discrete radius. Keep
legacy as the control and keep discrete radius as a secondary candidate. Do not
promote any candidate to default before a 16-sample controlled P2-b run confirms
shape stability, finite gradients, loss decline, and no unexpected graph-budget
regression.

If 16-sample P2-b still fails to separate nearest repair from legacy, move to P3
model path / decoder audit before changing loss or adding pointwise skip.

## P2-b Longer 16-sample Smoke

V2 source check:

- `docs/v2_closeout_summary.md` and `docs/v2_training_results_overview.md`
  explicitly record one-sample RIGNO memorization around 42% error.
- The same files record that a pointwise MLP fits the 1/4-sample cases below
  20%.
- A local search across `docs/`, `configs/`, current ignored `output/`, and
  sibling worktree v2 docs/output/config indexes did not find a clear source
  for MLP single-sample IID error below 2%. Treat `<2%` as user-reported,
  source pending.

V3 near-term fitting target: reduce RIGNO small-sample relative RMSE to
`<=20%` before any full-dataset benchmark.

Command:

```bash
python3 scripts/run_heat3d_v3_p2_policy_16sample_longer_smoke.py \
  --subset "/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small" \
  --epochs 100 \
  --lr 1e-5 \
  --output-dir output/heat3d_v3_p2_policy_smoke
```

Settings:

- 16 supervised-small samples, all treated as train-only fitting smoke.
- Original split composition: 10 train, 3 valid, 1 test_smoke, 1 test_ood_bc,
  1 test_ood_stack.
- Model/bridge/loss/optimizer semantics unchanged from P2-a smoke.
- No checkpoint and no full-dataset run.
- The optional 200-epoch run was skipped because the 100-epoch smoke took
  several minutes on the non-JIT manual-GD path.

Output: `output/heat3d_v3_p2_policy_smoke/p2_policy_16sample_longer_e100.json`.

| policy | final_loss | best_loss | loss_drop | raw DeltaT RMSE | raw DeltaT MAE | relative RMSE | p2r/r2p zero | p2r/r2p edges | edge_ratio p2r/r2p | <=20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 1.085127e+00 | 1.085127e+00 | 4.939425e-02 | 3.033689e-01 | 2.299350e-01 | 73.07% | 908 / 908 | 1796 / 1750 | 1.000 / 1.000 | false |
| nearest_repair | 1.127428e+00 | 1.127428e+00 | 4.747283e-02 | 3.091313e-01 | 2.371595e-01 | 74.46% | 0 / 0 | 2704 / 2658 | 1.506 / 1.519 | false |
| discrete_radius | 1.142690e+00 | 1.142690e+00 | 3.371632e-02 | 3.112457e-01 | 2.414942e-01 | 74.97% | 0 / 0 | 4923 / 4923 | 2.741 / 2.813 | false |

Additional run diagnostics:

| policy | loss_drop_ratio | relative RMSE gap to 20% | grad_norm min/median/max/final | graph_build_s | train_step_s |
| --- | ---: | ---: | --- | ---: | ---: |
| legacy | 4.35% | 53.07 pp | 3.741 / 5.419 / 18.700 / 3.741 | 0.154 | 1.620 |
| nearest_repair | 4.04% | 54.46 pp | 3.908 / 5.207 / 18.458 / 3.908 | 8.544 | 1.517 |
| discrete_radius | 2.87% | 54.97 pp | 4.088 / 4.730 / 14.389 / 4.088 | 2.292 | 1.486 |

Interpretation:

- P2-b confirms graph repair safety at 16 samples: nearest repair and discrete
  radius both remove p2r/r2p zero coverage while staying finite and shape-stable.
- P2-b does not support coverage repair as sufficient to reduce fitting error:
  legacy has the best final/best loss and lowest relative RMSE in this smoke.
- Nearest repair remains the lower-cost coverage guarantee, but it does not
  outperform legacy on fitting error here.
- Discrete radius remains useful as a coverage-oriented research candidate, but
  its edge cost is much higher and its 100-epoch fitting result is weaker.

Recommendation after P2-b:

- Do not run full-dataset controlled training yet.
- Do not change objective/loss or add pointwise skip from this evidence.
- Move next to P3 model path / decoder audit, because graph coverage repair
  alone did not bring RIGNO small-sample fitting close to the `<=20%` target.
