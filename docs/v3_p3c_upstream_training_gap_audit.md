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

Pending.

## Next Validation Priority

Run the P3-c optimizer matrix first. If Adam/lr brings RIGNO below `20%`, the
primary blocker is optimizer/update configuration. If Adam improves but remains
far above `20%`, inspect decoder-path and channel-scale results before moving to
capacity or P5 local decoder / pointwise skip work.
