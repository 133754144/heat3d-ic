# Heat3D v3 B96 nearest_repair OOM audit

## Scope

This audit only covers the `nearest_repair` B96 e400 configuration:

`configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml`

It does not check B196, does not change model/decoder/loss/objective semantics, and does not run the full e400 training.

## OOM location

The user-provided devbox traceback places the OOM inside the training step, not validation, train metrics, prediction export, or diagnostics export:

- Script: `scripts/run_heat3d_v1_medium_controlled_training_export.py`
- Function: `_fit_once`
- Failing line in the old code path: `grad_norm = _global_norm(grads)`
- Downstream helper: `scripts/check_heat3d_v1_small_train_valid_smoke.py::_global_norm`
- Allocation reported by JAX: 57.33 MiB

For B96, the training split has 704 samples and 8 mini-batch groups. The e400 command used `--grad-norm-report-every 10`, so no B96 batch should have reported grad norm. The old runner still materialized a full diagnostic grad norm for every batch, creating extra GPU allocation pressure unrelated to the optimizer update.

## Memory audit method

The audit uses:

- process RSS from Python `resource`
- JAX device `memory_stats()` from `jax.devices()`
- memory audit JSONL stages around graph building, train batches, validation, train metrics, prediction export, and save paths

The visible JAX CUDA device reports a pool limit of about 9169.9 MiB.

## Short probe results

All probes used nearest_repair, B96/B64 M2-width config, AdamW warmup-cosine, no full e400 run, and ignored `output/`.

| Probe | Result | Train groups | All groups | Grad norm reported/skipped | Max CPU RSS | JAX peak bytes in use |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B96 e5 default, save predictions + best predictions, train_metrics half/final | passed | 8 | 11 | 0 / 40 | 4336.7 MiB | 8958.2 MiB |
| B96 e5 train+valid-only flags, no prediction save, train_metrics none | passed | 8 | 11 | 0 / 40 | 4140.5 MiB | 8843.5 MiB |
| B64 e3 train+valid-only flags, no prediction save, train_metrics none | passed | 11 | 16 | 3 / 30 | 3441.1 MiB | 6154.2 MiB |

Notes:

- B96 memory headroom is thin: peak JAX bytes in use is close to the 9169.9 MiB pool limit.
- B64 is materially safer in GPU memory.
- The runner still builds `all` groups even when `save_predictions=False`; prediction arrays are skipped later, but all-graph preparation remains a memory surface.
- B96 group shapes are stable across epochs. The two B96 shape signatures come from the final partial group, not from epoch-to-epoch graph drift.
- CPU RSS rises slightly during graph/model setup and then mostly plateaus across epoch 1-5. No clear cross-epoch CPU leak was observed in the short probes.
- JAX peak bytes in use rises after compilation/first heavy batches and then stays below the pool limit in the short probes. No repeated shape-cache growth was observed.

## Memory hygiene fix

The runner now computes diagnostic grad/update/param norms only when one of these is true:

- the batch is selected by `--grad-norm-report-every`
- profiling is enabled

This preserves explicit debug/profile behavior but avoids unreported per-batch `_global_norm(grads)` allocations. Training math is unchanged: optimizer, gradient clipping, weight decay, loss, model, decoder, and objective paths are not modified.

With B96 and `--grad-norm-report-every 10`, there are 8 train batches per epoch, so the fixed runner reports 0 grad norms and skips 40 over 5 epochs. With B64, there are 11 batches per epoch, so batch 10 is still reported as requested.

Temporary `grads`, `updates`, and `loss_value` references are released after each batch. Optional memory audit hooks can also run `gc.collect()` at epoch boundaries, but the successful probes above did not require GC to pass.

## Recommendation

B96 nearest_repair can continue after this fix, but it should be treated as near the memory ceiling on the RTX 5070 WSL setup. For the manual B96 e400 run:

- keep profiling disabled
- keep `--grad-norm-report-every 10` or set it to `0`
- use memory audit JSONL for the first few epochs if the run is retried
- avoid extra every-batch diagnostics

Fallback:

- Use B64 if B96 OOM recurs; the B64 probe had much more GPU headroom.
- Use B48 only if B64 also fails or if additional diagnostics must be enabled.
- Do not infer anything about B196 from this audit; B196 was not checked.
