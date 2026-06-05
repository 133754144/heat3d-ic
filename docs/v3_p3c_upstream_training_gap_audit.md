# Heat3D v3 P3-c Upstream Training Gap And Optimizer Sanity

## Purpose

P3-a showed a pointwise MLP can fit `sample_000` to `0.583%` relative RMSE, but
P3-b RIGNO stayed near `68%` to `70%` after 300 epochs. P3-c compares Heat3D's
one-sample RIGNO smoke against the original RIGNO example training style, then
checks whether optimizer/lr or decoder routing explains the fitting bottleneck.

No model, decoder, loss, objective, graph semantics, checkpoint behavior, data,
or full-dataset run is changed.

## Upstream `example.py` Training Flow

Audited file: `/Users/xuyihua/Desktop/学习相关/myCode/rigno-main/example.py`
and expected devbox mirror `~/myCode/rigno-main/example.py`.

Key settings:

- Dataset: `unstructured/Heat-L-Sines`, `N_TRAIN = 32 * 16`, `N_VALID = 8 * 8`,
  `N_TEST = 16`, `BATCH_SIZE = 8`, `TRAINING_EPOCHS = 50 * 4`.
- Model: `processor_steps=8`, `node_latent_size=64`, `edge_latent_size=64`,
  `mlp_hidden_layers=3`, time-conditioned flags enabled only for
  time-dependent data, `p_edge_masking=0.5`.
- Graph builder: upstream `RegionInteractionGraphBuilder` with
  `rmesh_levels=4`, `subsample_factor=4`, `overlap_factor_p2r=1.0`,
  `overlap_factor_r2p=2.0`, `node_coordinate_freqs=4`.
- Graphs: `dataset.build_graphs(builder=graph_builder)` before training, and
  each epoch rebuilds graphs with a PRNG key for randomized regional nodes.
- Stats/normalization: `dataset.compute_stats(residual_steps=TAU_MAX_TRAINING)`;
  the stepper receives dataset stats and computes loss inputs in normalized
  model space.
- Loss: `mse_loss(*_loss_inputs)` from upstream metrics, through
  `TimeDerivativeStepper`.
- Optimizer: Optax AdamW via
  `optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-08)`.
  The example learning rate is `optax.exponential_decay` with `init_value=1e-2`,
  `decay_rate=0.1`, and transition steps spanning all planned updates.
- Train step: `_compute_loss` is `jax.jit`; the outer loop iterates batches and
  trajectory subbatches, then calls `jax.value_and_grad(_compute_loss)` and
  `state.apply_gradients(grads=grads)`.
- Checkpoint/eval: the example keeps an in-memory `best` params/loss pair, then
  autoregressively unrolls on test data and reports relative L1 errors. It is
  not a Heat3D-style checkpoint/export runner.

## Heat3D P2/P3 Smoke Differences

Current Heat3D one-sample smoke differs in several material ways:

| Area | upstream example | Heat3D P2/P3 smoke |
| --- | --- | --- |
| task | time-dependent operator / derivative stepping | steady DeltaT one-step regression |
| target | stepper-normalized trajectory target | train-only normalized DeltaT |
| optimizer | AdamW with decaying lr, `weight_decay=1e-08` | P2/P3-b mainly manual GD at `1e-5`; P3-c adds manual GD/Adam sanity |
| model size | latent 64/64, processor 8, MLP 3 | smoke `MODEL_CONFIG`: latent 16/16, processor 2, MLP 1 |
| graph | randomized upstream regional graphs each epoch | fixed Heat3D graph for the selected sample/policy |
| decoder input | upstream 2D/unstructured task path | Heat3D 3D point cloud path with repaired/candidate coverage policies |
| training budget | many batches and trajectory subbatches | one sample, one full-batch update per epoch |
| regularization | `p_edge_masking=0.5` when key is passed | Heat3D smoke config sets `p_edge_masking=0.0` |

## Likely Explanatory Gaps

Most plausible gaps to validate first:

1. Optimizer/lr: manual GD at `1e-5` may be too weak; upstream relies on AdamW
   with a much larger scheduled lr.
2. Capacity: Heat3D smoke model is much smaller than upstream example and v2
   stable-anchor configs.
3. Decoder/regional routing: P3-b showed active processor gradients and rnode
   latent changes, but decoder ablation remained pnode-dominant.
4. Fixed regional graph: upstream randomizes regional nodes during training for
   invariance; for one-sample fitting this may be less important than optimizer
   and decoder path, but it is still a behavioral difference.
5. Task mismatch: upstream solves temporal operator learning; Heat3D uses
   static 3D IC material/source/BC conditioning. Directly copying upstream
   training is not guaranteed to fix the Heat3D path.

## P3-c Commands

Optimizer/lr sanity:

```bash
python3 scripts/run_heat3d_v3_p3c_rigno_1sample_optimizer_sanity.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --policy legacy \
  --epochs 300 \
  --output-json output/heat3d_v3_p3c/optimizer_sanity.json
```

Decoder/path audit:

```bash
python3 scripts/audit_heat3d_v3_p3c_decoder_path.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --policy legacy \
  --use-best-from output/heat3d_v3_p3c/optimizer_sanity.json \
  --output-json output/heat3d_v3_p3c/decoder_path_audit.json
```

## Local Short Check

Local 2-epoch smoke passed for `adam`, lr `1e-4`. It only verifies script
compatibility:

- optimizer sanity: best relative RMSE `70.694%`.
- decoder audit: finite output, pnode-dominant judgment, rnode routing weak,
  no q scaling issue by normalized channel scale.

## Devbox Results

Devbox confirmed `~/myCode/rigno-main/example.py` exists and matches the
audited upstream example structure: `TRAINING_EPOCHS = 50 * 4`, `BATCH_SIZE = 8`,
`dataset.build_graphs(...)`, `dataset.compute_stats(...)`, Optax AdamW,
`mse_loss`, `@jax.jit`, and `state.apply_gradients(...)`.

Optimizer/lr sanity on `sample_000`, policy `legacy`, 300 epochs:

| optimizer | lr | best epoch | best loss | relative RMSE | raw RMSE | raw MAE | <=20% | <=2% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| manual_gd | 1e-5 | 300 | 1.092983e+00 | 68.30% | 2.054579e-01 | 1.565214e-01 | false | false |
| manual_gd | 1e-4 | 300 | 9.020259e-01 | 62.04% | 1.866490e-01 | 1.340776e-01 | false | false |
| manual_gd | 1e-3 | 300 | 7.303751e-01 | 55.83% | 1.679535e-01 | 1.182901e-01 | false | false |
| adam | 1e-4 | 300 | 3.131301e-01 | 36.55% | 1.099711e-01 | 8.206196e-02 | false | false |
| adam | 1e-3 | 300 | 7.435127e-02 | 17.81% | 5.358711e-02 | 2.844086e-02 | true | false |

Because Adam `1e-3` clearly improved, an optional 1000-epoch single run was
also executed:

| optimizer | lr | best epoch | best loss | relative RMSE | raw RMSE | raw MAE | <=20% | <=2% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| adam | 1e-3 | 1000 | 1.912380e-02 | 9.03% | 2.717711e-02 | 1.317779e-02 | true | false |

Optimizer conclusion: manual GD and too-low lr explain much of the P3-b failure.
Adam `1e-3` crosses the `<=20%` one-sample fitting gate without model/loss
changes. It still does not approach the pointwise MLP's `0.583%` result or the
`<=2%` target, so optimizer/lr is a major blocker but not the entire gap.

Decoder/path audit with the 1000-epoch Adam `1e-3` params:

| metric | value |
| --- | ---: |
| trained normalized loss | 1.912158e-02 |
| trained relative RMSE | 9.03% |
| latent_pnodes std / norm | 1.319 / 59.69 |
| processed_rnodes std / norm | 1.982 / 44.85 |
| r2p edge feature std / norm | 1.945e-02 / 4.364e-01 |
| decoded output std / norm | 0.962 / 10.89 |

Decoder ablations:

| ablation | relative RMSE | output change RMSE |
| --- | ---: | ---: |
| original | 9.03% | 0.000000e+00 |
| only_rnode / zero_pnode | 55.68% | 7.981235e-01 |
| only_pnode / zero_rnode | 58.12% | 8.802444e-01 |
| shuffle_rnode | 57.28% | 8.621439e-01 |
| shuffle_pnode | 67.49% | 9.968632e-01 |

Decoder/path conclusion: after a sane optimizer is used, the path is `mixed`
rather than strongly pnode-dominant. rnode routing is not weak: zeroing or
shuffling either side causes large output and error changes. The earlier
pnode-dominant P3-b result was partly confounded by undertraining.

q scaling check:

- q raw safe std: `8.804222e+06`
- q normalized std: `1.000002`
- q/target correlation: `0.340701`
- judgment: no q scaling issue detected by this audit.

## Next Validation Priority

P3-c changes the near-term diagnosis:

1. Optimizer/lr sanity should be promoted into P2/P3 smoke defaults for further
   audits: use Adam or AdamW, not manual GD, when judging RIGNO fitting ability.
2. Do not treat nearest graph repair as failed solely from manual-GD P2/P3-b
   results; rerun small graph-policy comparisons with Adam `1e-3` or an
   upstream-style AdamW schedule before drawing policy conclusions.
3. Since 1000 epochs reaches `9.03%` but not `<=2%`, next isolate capacity vs
   decoder/locality: run the same one-sample sanity with larger RIGNO capacity
   matching v2 stable anchors before adding P5 pointwise/local decoder.
4. P5 pointwise/local decoder remains plausible, but P3-c says optimizer must be
   fixed first; otherwise decoder conclusions are confounded by undertraining.
