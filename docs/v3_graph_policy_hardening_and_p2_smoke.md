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
