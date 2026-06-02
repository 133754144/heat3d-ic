# Heat3D v2 B48 capacity training results review

本记录是 research-stage diagnostic review，不是 formal benchmark。`valid_iid`
仍是主验证 split；`valid_stress` 只作为 diagnostic split。

## Queue status

- Queue log: `output/heat3d_v2_runs/queue_logs/queue_b48_series_after_current.log`
- 当前训练进程: 无。
- Queue watcher: 无。
- 队列状态: 7/7 completed, no pending.
- 已完成队列顺序:
  1. `m2width_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  2. `m25width_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  3. `m35width_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  4. `m3depth_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  5. `m25depth_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  6. `m2depth_B48_base_mse_warmup_cosine_e400_stratified_seed0`
  7. `m3width_B48_base_mse_warmup_cosine_e400_stratified_seed0`
- 另有已完成 run: `m3width_B48_base_mse_warmup_cosine_mlp3_e400_stratified_seed0`。
- 重新尝试的 `192/192/s8/mlp3/B48` 2epoch retry 仍然 OOM，未生成
  `loss_summary.json` 或 prediction export。

## Diagnostics status

对已完成的 B48 e400 runs 补齐了已有 post-hoc diagnostics:

- baseline comparison: final/best
- error bins: final/best
- condition diagnostics: final/best
- field-shape diagnostics: final/best
- run summary: final/best
- split-aware diagnostics: `valid_iid` / `valid_stress` x final/best

这些文件都写在 ignored `output/` run directory 中，不应提交。

## Loss summary

| run | model | B | best_ep | best_iid | final_iid | final_stress | final/best | raw_iid | raw_stress | ok | grad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 B192 replay | 64/64/s4/m2 | 192 | 200 | 0.3047 | 0.3047 | 0.4535 | 1.000 | 0.0005686 | 0.0008462 | Y | Y |
| M1.5 B96 | 96/96/s6/m2 | 96 | 305 | 0.2291 | 0.2299 | 0.3364 | 1.004 | 0.0004290 | 0.0006277 | Y | Y |
| M2 B96 | 128/128/s6/m2 | 96 | 351 | 0.2217 | 0.2223 | 0.3278 | 1.003 | 0.0004148 | 0.0006116 | Y | Y |
| M2 B48 | 128/128/s6/m2 | 48 | 353 | 0.2117 | 0.2119 | 0.3270 | 1.001 | 0.0003954 | 0.0006102 | Y | Y |
| M2.5 B48 | 160/160/s6/m2 | 48 | 326 | 0.2079 | 0.2089 | 0.3203 | 1.005 | 0.0003897 | 0.0005977 | Y | Y |
| M3.5 B48 | 224/224/s6/m2 | 48 | 395 | 0.2163 | 0.2164 | 0.3352 | 1.000 | 0.0004037 | 0.0006256 | Y | Y |
| M3-depth B48 | 192/192/s8/m2 | 48 | 399 | 0.9817 | 0.9817 | 0.9756 | 1.000 | 0.001832 | 0.001820 | Y | Y |
| M2.5-depth B48 | 160/160/s8/m2 | 48 | 395 | 0.3927 | 0.3927 | 0.5249 | 1.000 | 0.0007328 | 0.0009794 | Y | Y |
| M2-depth B48 | 128/128/s8/m2 | 48 | 396 | 0.2141 | 0.2142 | 0.3275 | 1.000 | 0.0003996 | 0.0006112 | Y | Y |
| M3-width B48 | 192/192/s6/m2 | 48 | 391 | 0.9830 | 0.9830 | 0.9752 | 1.000 | 0.001834 | 0.001820 | Y | Y |
| M3-mlp3 B48 | 192/192/s6/m3 | 48 | 400 | 0.2230 | 0.2230 | 0.3333 | 1.000 | 0.0004161 | 0.0006219 | Y | Y |

## Split-aware diagnostics

Key rows below use best predictions. Final predictions were very close for the
B48 e400 runs because `final/best` was near 1.0.

| run | split | raw_mse | var_ratio | corr | amp | topk | p95 | p99 | peak | bin0_bias | bin0_over | le005_bias | le005_over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 B192 replay | valid_iid | 0.02148 | 1.576 | 0.8906 | 1.024 | 0.7327 | 0.04118 | 0.07322 | 0.05688 | 0.01280 | 1.000 | 0.008158 | 0.7939 |
| M1 B192 replay | valid_stress | 0.02751 | 4.918 | 0.8457 | 1.306 | 0.6523 | 0.05459 | 0.09666 | 0.09210 | 0.01600 | 0.9934 | 0.009462 | 0.7453 |
| M1.5 B96 | valid_iid | 0.01811 | 2.024 | 0.9177 | 1.140 | 0.7481 | 0.03716 | 0.06714 | 0.05257 | 0.006529 | 0.9350 | 0.003857 | 0.7399 |
| M1.5 B96 | valid_stress | 0.02386 | 5.362 | 0.8712 | 1.544 | 0.6614 | 0.04871 | 0.08809 | 0.09630 | 0.009082 | 0.8902 | 0.005837 | 0.7145 |
| M2 B96 | valid_iid | 0.01778 | 2.026 | 0.9157 | 1.203 | 0.7192 | 0.03618 | 0.06673 | 0.05253 | 0.006226 | 0.9264 | 0.003629 | 0.7298 |
| M2 B96 | valid_stress | 0.02349 | 5.296 | 0.8675 | 1.626 | 0.6477 | 0.04766 | 0.08635 | 0.09750 | 0.008492 | 0.8704 | 0.004752 | 0.6811 |
| M2 B48 | valid_iid | 0.01725 | 2.253 | 0.9200 | 1.171 | 0.7404 | 0.03571 | 0.06325 | 0.04883 | 0.005439 | 0.9025 | 0.003512 | 0.7309 |
| M2 B48 | valid_stress | 0.02309 | 6.096 | 0.8768 | 1.591 | 0.6409 | 0.04705 | 0.08443 | 0.08845 | 0.008689 | 0.8695 | 0.005527 | 0.6991 |
| M2.5 B48 | valid_iid | 0.01714 | 2.271 | 0.9196 | 1.189 | 0.7327 | 0.03549 | 0.06316 | 0.05088 | 0.005533 | 0.8948 | 0.003543 | 0.7250 |
| M2.5 B48 | valid_stress | 0.02304 | 5.863 | 0.8697 | 1.599 | 0.6364 | 0.04707 | 0.08382 | 0.09226 | 0.008680 | 0.8646 | 0.005392 | 0.6938 |
| M3.5 B48 | valid_iid | 0.01760 | 2.320 | 0.9155 | 1.212 | 0.7346 | 0.03643 | 0.06380 | 0.05245 | 0.006521 | 0.9232 | 0.004255 | 0.7447 |
| M3.5 B48 | valid_stress | 0.02352 | 5.897 | 0.8600 | 1.637 | 0.6227 | 0.04790 | 0.08521 | 0.09578 | 0.009342 | 0.8703 | 0.005725 | 0.6958 |
| M2-depth B48 | valid_iid | 0.01731 | 2.293 | 0.9194 | 1.181 | 0.7365 | 0.03591 | 0.06372 | 0.04934 | 0.005480 | 0.9001 | 0.003612 | 0.7327 |
| M2-depth B48 | valid_stress | 0.02319 | 6.127 | 0.8749 | 1.599 | 0.6386 | 0.04742 | 0.08517 | 0.08828 | 0.008658 | 0.8681 | 0.005533 | 0.7006 |
| M2.5-depth B48 | valid_iid | 0.02156 | 0.9705 | 0.8591 | 1.040 | 0.7442 | 0.04199 | 0.07963 | 0.04315 | 0.008947 | 0.9851 | 0.004649 | 0.7544 |
| M2.5-depth B48 | valid_stress | 0.02755 | 2.232 | 0.7676 | 1.178 | 0.6909 | 0.05351 | 0.1053 | 0.07167 | 0.009669 | 0.9488 | 0.003491 | 0.6858 |
| M3-width B48 | valid_iid | 0.03652 | 0.2311 | 0.5872 | 0.2678 | 0.09231 | 0.05212 | 0.1439 | 0.2539 | 0.01447 | 1.000 | 0.008343 | 0.8008 |
| M3-width B48 | valid_stress | 0.03826 | 0.2615 | 0.5703 | 0.2263 | 0.03636 | 0.06619 | 0.1550 | 0.2443 | 0.01494 | 1.000 | 0.007717 | 0.7539 |
| M3-depth B48 | valid_iid | 0.03651 | 0.2337 | 0.5871 | 0.2664 | 0.08654 | 0.05210 | 0.1440 | 0.2547 | 0.01449 | 1.000 | 0.008371 | 0.8016 |
| M3-depth B48 | valid_stress | 0.03828 | 0.2601 | 0.5718 | 0.2103 | 0.02727 | 0.06651 | 0.1550 | 0.2486 | 0.01497 | 1.000 | 0.007695 | 0.7537 |
| M3-mlp3 B48 | valid_iid | 0.01770 | 2.417 | 0.9237 | 1.228 | 0.7462 | 0.03650 | 0.06397 | 0.05281 | 0.006607 | 0.9077 | 0.004358 | 0.7337 |
| M3-mlp3 B48 | valid_stress | 0.02327 | 6.016 | 0.8809 | 1.610 | 0.6500 | 0.04755 | 0.08398 | 0.09346 | 0.009760 | 0.8691 | 0.006278 | 0.6927 |

## Interpretation

1. B48 control vs B96 M2-width:
   - B48 improves scalar loss for the same 128/128/s6/m2 model:
     `best_iid` 0.2117 vs 0.2217.
   - Stress loss is slightly better: 0.3270 vs 0.3278.
   - Stress field variance ratio rises from about 5.30 to about 6.10, so the
     scalar improvement does not remove overshoot risk.

2. Width scaling:
   - 128 -> 160 gives the best scalar result in this batch:
     `M2.5 B48` has `best_iid=0.2079` and `final_stress=0.3203`.
   - 224 is still trainable but worse than 160.
   - 192/s6/m2 collapsed to a low-amplitude solution despite finite gradients.
     Treat this as a configuration-specific failure case and do not use it as
     evidence that wider is always better or worse.

3. Depth scaling:
   - 128/s8 is feasible but does not beat 128/s6.
   - 160/s8 degrades sharply.
   - 192/s8 collapses, and 192/s8/mlp3/B48 retry OOMs.
   - Current evidence says stop steps8 long runs for this branch.

4. MLP depth:
   - 192/s6/mlp3 is trainable and avoids the 192/s6/mlp2 collapse.
   - It improves stress correlation/top-k relative to M2/M2.5 B48, but scalar
     loss is worse than the 160/s6/m2 run.
   - It also keeps high stress variance/amplitude overshoot, so it is not the
     current scalar candidate.

5. Current strongest scalar model:
   - `m25width_B48_base_mse_warmup_cosine_e400_stratified_seed0`
   - model `160/160/s6/mlp2`, batch 48.

6. Current best tradeoff:
   - `M2.5 B48` is the best scalar/stress tradeoff among these runs.
   - `M3-mlp3 B48` is only interesting if the next objective prioritizes
     stress correlation/top-k over scalar loss.

7. Main bottleneck:
   - Capacity helps up to about 160 width, and B48 update dynamics helped.
   - Blindly increasing width/depth is not reliable: 192/s6/m2 and 192/s8/m2
     collapsed, while 224/s6/m2 underperformed 160.
   - The remaining bottleneck is a mix of objective/overshoot control and
     stable optimization under larger capacity. MSE-only training still leaves
     high stress variance/amplitude and low-DeltaT overprediction diagnostics.

## Recommended next training

Priority next steps:

1. Prepare or run `M2.5 B48 e400` seed1/seed2 to check whether the current
   best scalar candidate is stable.
2. Prepare or run conservative overshoot-control variants around M2.5 B48:
   stronger weight decay or light background/overprediction loss, while keeping
   `valid_stress` diagnostic-only.
3. Do not schedule more steps8 or 192/s8/mlp3 runs in this line unless the
   objective changes to pure feasibility testing.
