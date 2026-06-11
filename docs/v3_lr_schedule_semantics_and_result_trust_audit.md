# Heat3D v3 LR Schedule Semantics And Result Trust Audit

Purpose: document and fix a learning-rate schedule semantics mismatch in the
controlled runner. No training was run for this audit.

## Bug

The CLI/log meaning of `two_stage` and `second_stage` is epoch based:

- `--second-stage-epoch 400` should mean epoch 1-400 use base LR.
- Epoch 401 and later should use `--second-stage-lr`.

The previous Optax schedule computed the current epoch but did not use it for
the `two_stage` / `second_stage` boundary. It compared `update_count + 1`
directly against `second_stage_epoch`. For B88 runs with 8 updates per epoch,
`second_stage_epoch=400` therefore switched LR after update 400, which is about
epoch 50, not after epoch 400.

The fix changes Optax `two_stage` and `second_stage` boundary checks to use:

`epoch = floor(update_count / updates_per_epoch) + 1`

This makes `_lr_for_epoch` and the actual Optax schedule agree.

## Affected Trust

`T3_seed1_e1200_twostage_lr1e-3_to3e-4_at400` is not trustworthy as evidence
for "e400 delayed decay failed." Under the old implementation on B88, it should
be reinterpreted as:

- approximately first 50 epochs at `lr=1e-3`
- remaining epochs at `lr=3e-4`

So T3 is only a negative example for "overly early LR drop after about e50";
it does not test or disprove a true delayed decay after epoch 400.

## Results Not Severely Affected

- `L2 constant lr=1e-3` remains trustworthy because `constant` did not use the
  stage boundary.
- `A5/A6 constant` remain trustworthy for the same reason.
- `B6 warmup_cosine` remains trustworthy; it already used epoch derived from
  `update_count / updates_per_epoch`.
- `L3 warmup_cosine` remains trustworthy as a cosine decay negative result.
- `U1/U2/U3 upstream_onecycle` are not affected by the same stage-boundary bug.
  They are per-update continuous schedules. The runner's printed epoch LR is
  an epoch-level approximation for logging, while Optax changes continuously
  per update.

## Post-Fix Check

Added `scripts/check_heat3d_v3_lr_schedule_semantics.py`.

The check verifies:

- with `updates_per_epoch=8`, `epochs=1200`, `base_lr=1e-3`,
  `second_stage_epoch=400`, and `second_stage_lr=3e-4`:
  - steps 0, 399, 400, and 3199 use `1e-3`
  - step 3200 and later use `3e-4`
- `constant` remains unchanged.
- `warmup_cosine` keeps the existing epoch-derived behavior.
- `upstream_onecycle` remains a per-update continuous schedule.

## Next Use

If delayed decay remains worth testing, rerun a true epoch-based `two_stage`
after this fix. Do not compare the old T3 run as if it had delayed LR decay
until epoch 400.
