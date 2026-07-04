# V4P3 Next Config Plan

Read this file only for V4P3_09/10 OOM status or V4P3_11-14 launch planning.

`V4P3_09` and `V4P3_10` are fixed as `status=oom_failed` with
`reason=processor_steps8_oom`. The `processor_steps=8` direction is stopped for
the current B32 / latent96-edge96 / AdamW memory budget.

| config_id | base | delta | purpose | expected diagnostic signal |
| --- | --- | --- | --- | --- |
| V4P3_11 | V4P3_08 | `epochs=200`; keep `processor_steps=6`, B32, formal split, `prediction_split=valid_iid` | E200 short/base control | Check whether early-stop-length control preserves V4P3_08 split gains without 600-epoch overfit. |
| V4P3_12 | V4P3_11 | `condition_feature_transform=semantic_v1_logk_signedlog1p_q_binary_bcflags_independent_bc_scalars` | Verify q/k semantic transform | Compare scalar/shape metrics and strong-q bins against V4P3_11. |
| V4P3_13 | V4P3_12 | `background_relative_weight=0.05`, `background_over_weight=0.02` | Light background calibration | Reduce bin0/le0.05 overprediction without hurting RMSE or top-k. |
| V4P3_14 | V4P3_12 | `strong_q_weight=0.05`, `hotspot_weight=0.05` | Light strong-q/hotspot emphasis | Improve q_power_bin_2, strong-q, top-DeltaT, and top-k metrics without large background regression. |

Do not register `raw_plus_fourier` or combined loss/transform configs until
V4P3_11-14 results are reviewed.
