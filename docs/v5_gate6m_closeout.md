# Gate 6M no-training closeout

Gate 6M completed valid-only branch swapping and shared-backbone gradient
diagnostics. It did not train either prepared candidate, modify a checkpoint,
or access `test/hard/sealed`.

## Branch swapping

| Field | Point-global | Sample-first | Raw CV RMSE | Shape CV-RMSE | Scale log-RMSE |
|---|---:|---:|---:|---:|---:|
| V32 e474 | 22.4084% | 21.0348% | 0.160067 K | 0.145697 | 0.197482 |
| O075 e280 | 23.4299% | 19.8446% | 0.167186 K | 0.150128 | 0.157300 |
| shape_V32 + scale_O075 | 23.1424% | 19.4796% | 0.165309 K | 0.145697 | 0.157300 |
| shape_O075 + scale_V32 | 22.7412% | 21.5050% | 0.162101 K | 0.150128 | 0.197482 |

O075 scale improves sample-first behavior when attached to V32 shape, but its
Q4 point-SSE delta against V32 is `+277.115 K²`; Q1–Q3 together improve by
`-96.128 K²`. O075 shape with V32 scale regresses against V32 in all four
DeltaT quartiles, with Q4 contributing `+65.540 K²`.

Thus V32 has the stronger shape branch. O075 has a useful low/mid-energy scale
branch but its high-energy tail prevents a point-global improvement. Neither
swap is a new checkpoint or advancement candidate.

## Shared-backbone gradient audit

The most relevant off-diagonal cosine values are:

| Checkpoint | Loss pair | Shared-backbone cosine |
|---|---|---:|
| V32 | relative vs raw | -0.1458 |
| V32 | shape vs relative | +0.4160 |
| V32 | scale vs relative | +0.3345 |
| O075 | shape vs raw | -0.0833 |
| O075 | shape vs relative | +0.3628 |
| O075 | scale vs raw | +0.2452 |

The four objectives are not uniformly antagonistic, but V32 retains a
relative/raw conflict on the shared backbone. Full matrices and norms are in
the JSON/CSV artifacts.

## Train-independent physical attribution

The point-SSE deltas were compared against direct inference-time context:
total power, source concentration, q-weighted inverse conductivity,
anisotropy, and top-h. No target, train-fitted feature, or learned
representation was used for these distances/correlations.

Across all four swap/reference comparisons, absolute Pearson and Spearman
correlations are below approximately `0.18`. No single audited physical
condition explains the branch-swap regression. The robust attribution is
instead the target-scale Q4 concentration: Q4 contributes about 71%–74% of
absolute point-SSE movement.

## Prepared configs

| Candidate | Config | Host | Contract | Status |
|---|---|---|---|---|
| A | `V4P5_35_gate6m_v32_scale_head_only_e100` | devbox | V32 e474 params-only, fresh optimizer, scale head only, e100 | started_user_managed |
| B | `V4P5_36_gate6m_v32_epoch_regroup_e600` | WSL2 | V32 random-init, epoch-wise batch regrouping only, e600 | prepared_not_started |

Both retain V32 B28, validation/prediction B32, architecture, loss, optimizer,
LR schedule, seeds, split, and all four checkpoint classes except for the
explicitly declared single-variable contract. B also retains V32's 600 epochs,
so its complete warmup-cosine LR trajectory is identical to V32. Commands are in
`docs/v5_gate6m_launch_commands.md`; Gate 6M did not execute them.

A's YAML and run contract were not changed by this revision. Only the V5
registry and this closeout record the user-reported started state. B remains
not started.

## Remote synchronization

The closeout content commit `73181153ad2d6c4c20b6eb942157e1c2b6cd23dc`
and V32 point-global e474 checkpoint were verified on both servers before the
sync-manifest commit:

| Host | Assignment | Branch | Content HEAD | Checkpoint SHA256 | Dry-run |
|---|---|---|---|---|---|
| devbox | A | research/v5 | `7318115` | `f3063b53…f045d24` | passed |
| WSL2 | B | research/v5 | `7318115` | `f3063b53…f045d24` | passed |

Both worktrees were clean. The checkpoint uses the same relative input path
on both hosts. After committing this manifest, both servers must be
fast-forwarded once more to the final Git HEAD; that refresh does not alter
the ignored checkpoint artifact.

## A completed-run recovery (valid_iid only)

The user-reported completed run was found on WSL2 at
`output/heat3d_v5_gate6m_a_runs/V4P5_35_gate6m_v32_scale_head_only_e100`.
The plan assignment remains `devbox`; the result registry records the actual
source as `wsl2` and the result columns as completed valid-only. The original
user-managed lifecycle label is preserved. The evaluator and compact
result artifacts are:

- `configs/heat3d_v5/gate6m/gate6m_a_valid_only_metrics.json`
- `configs/heat3d_v5/gate6m/gate6m_a_valid_only_metrics.csv`
- `docs/v5_gate6m_a_valid_only_closeout.md`
- `scripts/evaluate_heat3d_v5_gate6m_valid_only.py`

The four checkpoint artifacts are complete and reloadable (point-global/legacy
e18, sample-first e25, final e100; 893736 parameters each). The best
point-global result is 22.390066%, sample-first is 20.936297%, and raw
CV-weighted RMSE is 0.159923 K, so the `<20%` clean threshold is not met.
Scale log-RMSE improves over V32 e474 while shape is unchanged to numerical
precision; the final checkpoint regresses relative to e18. This is a
scale-only diagnostic result, not a promotion.

`V4P5_36_gate6m_v32_epoch_regroup_e600` remains `not_started`: no run config,
loss summary, checkpoint, or prediction artifact was present on either host.
No test, hard, or sealed role was opened.
