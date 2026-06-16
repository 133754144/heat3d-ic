# Heat3D v3 D1/D2 S5-Schedule Long Results

Scope: read-only review of completed D1-S5R and D2-S5R runs on WSL2/devbox.
No training was started by this review, no final-probe labels were trained, and
no output artifacts are committed.

## Server Check

| server | branch | head | active Heat3D training | latest relevant run |
| --- | --- | --- | --- | --- |
| WSL2 | `research/v3-startup-supervision` | `ff330f3` | none | D1-S5R e1600 completed |
| devbox | `research/v3-startup-supervision` | `ff330f3` | none | D2-S5R e1600 completed |

## Completed Runs

| run | server | model change | best epoch | final epoch | checkpoints | predictions | diagnostics | final probe |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| D1-S5R | WSL2 | `mlp_hidden_layers=3` | 596 | 1600 | best/final saved | best/final saved | complete | complete |
| D2-S5R | devbox | `mlp_hidden_layers=4` | 673 | 1600 | best/final saved | best/final saved | complete | complete |

Ignored output locations:

- `output/heat3d_v2_runs/latent96_s6_mlp3_B88_sample_shuffle_nearest_repair_D1S5R_S5basebest_mse_e1600_s5schedule_wd1e-4/`
- `output/heat3d_v3_decoder_s5schedule_audit/D1S5R/`
- `output/heat3d_v3_final_probe_D1S5R/`
- `output/heat3d_v2_runs/latent96_s6_mlp4_B88_sample_shuffle_nearest_repair_D2S5R_S5basebest_mse_e1600_s5schedule_wd1e-4/`
- `output/heat3d_v3_decoder_s5schedule_audit/D2S5R/`
- `output/heat3d_v3_final_probe_D2S5R/`

## Main Comparison

| run | label | valid_iid/base | valid_stress/base | raw RMSE | raw MAE | zRMSE | top-k | peak_rel | final-probe RMSE | P02 RMSE | P03 RMSE | P09 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S5 base best | best | 0.0210238 | 0.0291828 | 0.00282356 | n/a | 0.0662843 | 0.948828 | 0.0320597 | 0.372716 | 0.631598 | 0.980921 | 0.547795 |
| S5 base final | final | 0.0212054 | 0.0289898 | 0.0027924 | n/a | 0.0656913 | 0.950586 | 0.0327246 | 0.372552 | 0.631385 | 0.980932 | 0.547593 |
| D3-L200 `s8_mlp2` | final | 0.0210686 | 0.0282471 | 0.00288647 | 0.00163844 | 0.0696691 | 0.949219 | 0.0324199 | 0.372393 | 0.630046 | 0.97959 | 0.546957 |
| D1-L400 `s6_mlp3` | best | 0.101528 | 0.132198 | 0.013766 | 0.0110989 | 0.346773 | 0.827734 | 0.0772344 | 0.38429 | 0.626469 | 0.988155 | 0.568773 |
| D1-S5R `s6_mlp3` | best | 0.0236985 | 0.0376156 | 0.00362398 | 0.00214137 | 0.0872893 | 0.937109 | 0.0420732 | 0.37743 | 0.636631 | 0.984623 | 0.552151 |
| D1-S5R `s6_mlp3` | final | 0.0244051 | 0.0374426 | 0.00294756 | 0.00169064 | 0.0689539 | 0.950000 | 0.0359907 | 0.378001 | 0.638348 | 0.984897 | 0.553135 |
| D2-S5R `s6_mlp4` | best | 0.0246530 | 0.0430413 | 0.00338012 | 0.00195067 | 0.0822911 | 0.939453 | 0.0379606 | 0.378744 | 0.641880 | 0.976967 | 0.559638 |
| D2-S5R `s6_mlp4` | final | 0.0261158 | 0.0422316 | 0.00296694 | 0.00166688 | 0.0685561 | 0.948633 | 0.0353340 | 0.380036 | 0.641741 | 0.979153 | 0.561208 |

## Interpretation

- S5-schedule retraining-style configs are a fairer D1/D2 test than the earlier
  low-lr D1-L400. D1-S5R improves D1-L400 scalar loss by roughly 4x:
  `0.0237` best valid_iid/base vs `0.1015`.
- D1-S5R still does not beat S5 or D3-L200. Its best scalar loss is worse than
  S5 base best (`0.0237` vs `0.0210`), and its stress loss is much worse
  (`0.0376` vs S5 best `0.0292` and D3 final `0.0282`).
- D2-S5R is worse than D1-S5R on scalar and stress. The larger decoder MLP
  depth does not show useful upside in this configuration.
- D1-S5R/D2-S5R final checkpoints have decent raw shape metrics after long
  training, but those gains do not compensate for worse scalar/stress and
  weaker final-probe means.
- Best epochs are early relative to e1600: D1-S5R best at 596, D2-S5R best at
  673. The later epochs improve some raw mechanism fields but regress scalar
  validation and final-probe mean RMSE.

## Decision

- Do not promote D1-S5R or D2-S5R as new baselines.
- Do not continue MLP-depth expansion as the main v3 path. D1/D2 now have both
  weak low-lr and S5-schedule evidence, and neither beats S5/D3.
- Keep D3/processor-depth as the more plausible model-path direction if another
  capacity experiment is needed.
- Keep S5final FT / S5final EM / P4 targeted-loss family as stronger checkpoint
  candidates for final-probe-oriented work.
- P10 remains an unsupported final-probe schema gap: localized top contact and
  side asymmetry are still not represented by the current probe generator.
