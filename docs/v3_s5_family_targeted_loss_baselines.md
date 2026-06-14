# Heat3D v3 S5-Family Targeted-Loss Baselines

This table freezes the S5-family diagnostic baselines used before preparing
the hotspot/strong-q targeted-loss fine-tune. It records existing output
metrics only. No output files are committed.

| run name | checkpoint path | checkpoint exists | valid_iid | valid_stress | raw RMSE | zRMSE | top-k | peak_rel | role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B6 best | not saved in legacy B6 run | no | 0.0203209 | 0.0306349 | 0.00377562 | 0.100659 | 0.930469 | 0.041156 | stronger seed0 scalar reference before S4/S5 |
| S4 best | not saved in pre-checkpoint S4 run | no | 0.0197146 | 0.0313683 | 0.00335738 | 0.0869186 | 0.938086 | 0.0365489 | B6 e600 extension, best scalar in this local family |
| S5 base best | `output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5_seed0_e1600_warmupcosine_lr5e-4_minlr5e-5_wd1e-4/params_best.pkl` | yes on devbox | 0.0210238 | 0.0291828 | 0.00282356 | 0.0662843 | 0.948828 | 0.0320597 | scalar-oriented S5 checkpoint baseline |
| S5 base final | `output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5_seed0_e1600_warmupcosine_lr5e-4_minlr5e-5_wd1e-4/params_final.pkl` | yes on devbox | 0.0212054 | 0.0289898 | 0.0027924 | 0.0656913 | 0.950586 | 0.0327246 | raw/stress-oriented S5 checkpoint baseline |
| S5final FT no-mask final | `output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5final_FT_e100_lr1e-5_nomask_wd1e-4/params_final.pkl` | yes on devbox | 0.0213502 | 0.0289516 | 0.0027353 | 0.0646257 | 0.951758 | 0.0312176 | best no-mask raw mechanism checkpoint for targeted-loss start |
| S5final EM final | `output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S5final_EM_e100_lr1e-5_edgemask0p02_wd1e-4/params_final.pkl` | yes on devbox | 0.0213441 | 0.0289441 | 0.00273608 | 0.0646229 | 0.952344 | 0.0312305 | edge-mask raw mechanism checkpoint; close to no-mask FT |

Interpretation:

- S4 best remains the strongest scalar `valid_iid` row in this table, but it
  predates default checkpoint saving.
- S5final FT no-mask final and S5final EM final are the strongest raw mechanism
  rows available with params checkpoints.
- The first targeted-loss config uses S5final FT no-mask final as the
  conservative no-mask starting point.
- For `hotspot_strong_q` runs, `valid_loss` is the targeted total loss.
  Cross-run model quality should be compared using `valid_base_mse`,
  `valid_raw_deltaT_mse`, `iid_err`/`stress_err`, and downstream diagnostics.
