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

## Post-Fix Trusted Schedule Diagnostics

| run | trust classification | schedule | scalar result | interpretation |
| --- | --- | --- | --- | --- |
| W1 seed1 e1200 warmup-flat | trusted completed diagnostic run; not formal benchmark; not publication-ready result | `upstream_onecycle`, `lr_init=1e-4`, `lr_peak=lr_base=lr_lowr=1e-3`, `pct_start=10/1200`, `pct_final=0` | final `valid_loss=0.114535391`; best epoch 1199, best `valid_loss=0.109367043` | early-positive, final-negative versus L2 |

W1 was trained after the two-stage schedule semantics fix and is not affected
by the old stage-boundary bug. The synced WSL2 run was diagnosed read-only on
the main SSH server from:

`output/_from_wsl2/DESKTOP-2GE35DV/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_W1_seed1_e1200_upstream_warmup_flat_lr1e-3_wd1e-4`

W1 scalar and diagnostics summary:

- LR history: count 1200, first `1e-4`, last/min/max after warmup `1e-3`.
- Final: `valid_iid_loss=0.114535391`, `valid_stress_loss=0.117784038`.
- Best: epoch 1199, `valid_iid_loss=0.109367043`,
  `valid_stress_loss=0.114138097`, `final_best_ratio=1.04725691`.
- Final prediction diagnostics: mean DeltaT RMSE `0.009326461`, MAE
  `0.004474867`, centered spatial correlation `0.978120523`,
  top-k overlap `0.920703125`, bin0 signed bias `+0.002938616`,
  bin0 overprediction ratio `0.990051270`.
- Best prediction diagnostics: mean DeltaT RMSE `0.008499829`, MAE
  `0.003310936`, centered spatial correlation `0.979145478`,
  top-k overlap `0.927343750`, bin0 signed bias `-0.001069163`,
  bin0 overprediction ratio `0.062642415`.
- `per_sample_zscore_rmse` was not emitted by the current field-shape
  diagnostics JSON; field-shape correlation and top-k overlap were available.

W1 condition summary:

- Loss summary records separate `valid_iid` and `valid_stress` losses. The
  prediction diagnostics split grouping reports `valid` rather than separate
  `valid_iid` / `valid_stress` labels for the exported prediction archive.
- Best prediction weakest split-level DeltaT RMSE groups were
  `test_ood_bc_candidate` (`0.027501449`), `test_ood_combined_candidate`
  (`0.022440546`), `test_id` (`0.022070082`), and
  `test_ood_stack_candidate` (`0.019846467`).
- Best prediction weakest condition groups included
  `source_category=multi_block_power` (`rmse=0.024381868`),
  `source_category=high_dynamic_range_power_cases` (`0.023508902`),
  `k_mode=diag3` (`0.020046640`),
  `k_region_mode=high_contrast_interface_k` (`0.018967655`),
  `k_region_mode=low_k_barrier_or_TIM_variation` (`0.018708248`),
  `bc_category=very_low_top_h_candidate` (`0.030075804`),
  `bc_category=held_out_top_h_candidate` (`0.023261133`), and
  `bc_category=very_high_top_h_candidate` (`0.022996380`).

Research interpretation:

- W1 is early-positive: its e20/e50/e100/e200/e400 trajectory is consistent
  with the earlier L3 early advantage and is clearly better than the L2 early
  path.
- W1 is final-negative relative to L2: its e1200 best `valid_loss=0.109367043`
  is weaker than the L2 seed1 e1200 constant-lr reference
  (`best/final valid_loss approximately 0.069704`).
- Current conclusion: warmup-flat can repair the early seed1 trajectory, but
  it does not reproduce L2 late-stage convergence.
- Next emphasis should move toward checkpoint-level mechanism diagnostics and
  P3 model-path audit instead of blindly adding more LR schedules.
- The old T3 result remains an early-decay negative control, not a true
  epoch-400 delayed-decay result.

## Next Use

If delayed decay remains worth testing, rerun a true epoch-based `two_stage`
after this fix. Do not compare the old T3 run as if it had delayed LR decay
until epoch 400.
