# Heat3D v2 Upstream RIGNO Training Gap Audit

本文审计 upstream RIGNO 原作者训练框架与当前 Heat3D v2 runner 的差距。审计对象为 `/tmp/rigno-upstream-audit`，remote `https://github.com/camlab-ethz/rigno.git`，commit `3e4b307`。本文只用于 research-stage diagnostic，不是 formal benchmark 结论。

## Upstream Training Loop Summary

upstream `rigno/train.py` 的训练入口将 `TrainState`、normalization stats 和 graph data replicate 到设备，然后用 `jax.pmap` 编译 `_train_one_batch`。每个 epoch：

1. optionally rebuild graphs with a new PRNG key when `dataset.metadata.fix_x`;
2. iterate `dataset.batches(mode="train", batch_size=FLAGS.batch_size, key=subkey)`;
3. shard each batch across devices;
4. compute loss and grads inside compiled `pmap`;
5. average loss/grads with `jax.lax.pmean`;
6. call `state.apply_gradients(grads=grads)`;
7. evaluate only every `evaluation_frequency` or final epoch;
8. checkpoint with Orbax using validation final relative-L1 metric as best criterion.

## Optimizer, LR Schedule, And Clipping

upstream optimizer path uses Optax AdamW with `optax.inject_hyperparams(optax.adamw)`. The default schedule is a joined schedule:

- warmup/cosine one-cycle phase;
- final exponential decay phase;
- weight_decay is `1e-08`.

The inspected upstream train path does not show explicit `clip_by_global_norm` in the main AdamW chain. Heat3D v2 currently uses AdamW with constant or simple schedule, `weight_decay=1e-4`, and optional `optax.clip_by_global_norm(1.0)`.

## Loss Input Space And Normalization

upstream stepper normalizes model inputs and usually computes the loss on normalized target/prediction pairs. The default `mse_loss` is simple mean squared error on normalized arrays. Evaluation metrics include relative Lp errors after metadata normalization.

Heat3D v2 uses `normalized_deltaT` as base training target but its main `background_pseudo_negative` loss combines:

- normalized DeltaT base MSE;
- raw DeltaT background L1;
- raw DeltaT background signed bias;
- raw DeltaT overprediction penalty;
- raw DeltaT relative background penalty;
- pseudo-negative relative overprediction;
- normalized hotspot retention.

This is substantially more shaped than upstream default MSE.

## Batching, Graphs, And Compiled Step

upstream `Dataset.build_graphs` prebuilds graph metadata for dataset samples, pads graph edge sets to stable shapes, concatenates graph tensors, and stores them on the dataset. `Dataset.batches(...)` then returns batches with cached graphs. This reduces graph rebuild inside the training loop and helps avoid shape-driven recompilation.

Heat3D v2 currently builds grouped arrays and graphs at startup for train/valid/all groups, then reuses those groups during epochs. It does not use upstream `Dataset` graph padding/cache directly, but B192 startup graph build is a one-time cost and not the main epoch-time issue. Current train step is Python-looped over groups with `jax.value_and_grad(loss_fn)` per batch, not a dedicated `jit`/`pmap` compiled train_step.

## Validation And Checkpoint Selection

upstream evaluates at a low frequency derived from total epochs, plus the final epoch. It checkpoints by validation final relative-L1. Heat3D v2 validates every epoch and selects by `valid_loss` by default. In B192 e50 this gives enough evidence that best_epoch is epoch 1, but it also means `valid_loss` is tied to the chosen training loss, which may not align with field-shape or hotspot metrics.

## Difference Table

| Area | upstream RIGNO | Heat3D v2 runner | Risk |
|---|---|---|---|
| train step | `jax.pmap` compiled `_train_one_batch` | Python loop, per-batch `jax.value_and_grad` | slower, less stable profiling; possible repeated compile/sync overhead |
| LR schedule | warmup/cosine one-cycle + final decay | mostly constant / simple second_stage | constant lr may overshoot after epoch 1 |
| weight decay | `1e-08` | `1e-4` | much stronger decay may affect amplitude/field scale |
| grad clipping | no obvious clip in inspected AdamW chain | clip_by_global_norm `1.0` | clipping may interact with B192 large gradients |
| loss | simple normalized MSE by default | composite normalized/raw mixed loss | competing objectives and metric mismatch |
| graphs | dataset prebuild/pad/cache | startup grouped graph build/reuse | engineering difference; less likely main B192 degradation cause |
| batch count | full-batch count depends on dataset, no remainder assumption except initial assert | B192 has only 4 updates/epoch | low update count, poor e50 update budget |
| validation | low-frequency | every epoch | more diagnostics but not likely cause |
| selection | validation final relative-L1 | `valid_loss` | may select loss-optimal but not field-optimal checkpoints |

## Hypotheses For best_epoch=1 / Later Degradation

1. `lr=3e-4` is too large for B192. B192 has smoother but larger effective-batch gradients; constant lr can overshoot after the first few updates.
2. B192 has too few updates: only 4 updates/epoch and 200 total updates over e50. This is not update-count equivalent to B4 e50.
3. Heat3D v2 `weight_decay=1e-4` is much larger than upstream `1e-8`, and may contribute to amplitude drift or late degradation.
4. Current full composite loss mixes normalized and raw-space objectives; simplified B192 results improved best_valid_loss but did not fix early-best, so optimizer/LR/update count remain likely causes.
5. Current runner lacks compact train mini-batch component monitoring, making it hard to separate base MSE learning from background/pseudo-negative/hotspot tradeoffs.
6. `valid_loss` is not fully aligned with field-shape, peak, or low-temperature metrics; selection may hide better field checkpoints.

## Engineering Differences Less Likely To Explain B192 Degradation

- Startup graph build is expensive but one-time and B192 e50 wall-clock is already short.
- Validation every epoch increases runtime but should not cause training degradation.
- Prediction export happens after training and does not affect optimizer updates.

## Top 5 Recommendations

1. Test lower B192 LR first: `1e-4`, then `3e-5`, before broader loss changes.
2. Add an update-count-aware design: compare B192 e50 against B4 by total updates, or add gradient accumulation for larger effective batch with more updates.
3. Test optimizer knobs after LR: `weight_decay=0` and `gradient_clip_norm=0.5`.
4. Add a lightweight runner monitor for updates_per_epoch, total_update_count, per-epoch train mini-batch loss/component means, and final/best ratio.
5. Consider upstream-style schedule and compiled train_step only after the simpler LR/update-count evidence is clear.
