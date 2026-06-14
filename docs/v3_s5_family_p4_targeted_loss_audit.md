# Heat3D v3 S5-Family + P4 Targeted-Loss Audit

Scope: read-only diagnostics over completed S5-family and P4 predictions/checkpoints.
No training was started, no final-probe labels were regenerated, and no output
artifacts are committed.

## Unified Table

| run | role | checkpoint | epoch | selection_metric | valid_base_mse | iid_err_pct | stress_base_mse | stress_err_pct | raw RMSE | zRMSE | top-k | peak_rel | top5 RMSE | top10 RMSE | strong-q RMSE | final-probe RMSE | final-probe relRMSE | P02 RMSE | P03 RMSE | P09 RMSE |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4 best | best scalar pre-checkpoint baseline | no | 597 | valid_loss | 0.0197146 | 21.7018 | 0.0313683 | 24.6866 | 0.00335738 | 0.0869186 | 0.938086 | 0.0365489 | 0.00943539 | 0.00785359 | 0.0111704 | n/a | n/a | n/a | n/a | n/a |
| S5 base best | S5 scalar-oriented checkpoint | yes | 1527 | valid_loss | 0.0210238 | 22.4119 | 0.0291828 | 23.8142 | 0.00282356 | 0.0662843 | 0.948828 | 0.0320597 | 0.0084009 | 0.00684465 | 0.0108136 | 0.372716 | 0.805463 | 0.631598 | 0.980921 | 0.547795 |
| S5 base final | S5 raw/stress-oriented checkpoint | yes | 1600 | valid_loss | 0.0212054 | 22.5093 | 0.0289898 | 23.7346 | 0.0027924 | 0.0656913 | 0.950586 | 0.0327246 | 0.00842671 | 0.00682534 | 0.0110382 | 0.372552 | 0.80497 | 0.631385 | 0.980932 | 0.547593 |
| S5final FT no-mask final | no-mask fine-tune fallback candidate | yes | 100 | valid_loss | 0.0213502 | 22.5854 | 0.0289516 | 23.7225 | 0.0027353 | 0.0646257 | 0.951758 | 0.0312176 | 0.00813673 | 0.00661579 | 0.0106236 | 0.372366 | 0.804419 | 0.631203 | 0.980769 | 0.54755 |
| S5final EM final | edge-mask fine-tune candidate | yes | 100 | valid_loss | 0.0213441 | 22.5831 | 0.0289441 | 23.7154 | 0.00273608 | 0.0646229 | 0.952344 | 0.0312305 | 0.008137 | 0.00661607 | 0.0106285 | 0.372353 | 0.804389 | 0.631186 | 0.980766 | 0.547533 |
| P4 mse-selected best | targeted-loss best selected by valid_base_mse | yes | 83 | valid_base_mse | 0.0211334 | 22.4717 | 0.0288591 | 23.6816 | 0.00273058 | 0.0644722 | 0.952539 | 0.0309734 | 0.0081395 | 0.00661375 | 0.0105845 | 0.372598 | 0.805305 | 0.631104 | 0.980627 | 0.547066 |
| P4 mse-selected final | targeted-loss final checkpoint | yes | 100 | valid_base_mse | 0.0212424 | 22.5264 | 0.0289112 | 23.7029 | 0.00272915 | 0.0644472 | 0.952148 | 0.0306853 | 0.00811697 | 0.00659719 | 0.0105311 | 0.372563 | 0.805188 | 0.63109 | 0.980628 | 0.547071 |

## P4 Loss Components

| label | valid_loss | valid_base_mse | hotspot_mse | strong_q_mse | hotspot_mask_fraction | strong_q_mask_fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P4 mse-selected best | 0.039735 | 0.0211334 | 0.143407 | 0.228624 | 0.101713 | 0.00954027 |
| P4 mse-selected final | 0.0399992 | 0.0212424 | 0.143877 | 0.23126 | 0.101713 | 0.00954027 |

## Conclusions

- P4 best was selected by `valid_base_mse` at epoch 83. Targeted total
  `valid_loss` is not the cross-run quality metric.
- P4 best is slightly better than P4 final on scalar base MSE:
  `0.0211334` vs `0.0212424`.
- Compared with S5final FT no-mask final, P4 best improves in-distribution
  scalar/base and raw mechanism metrics slightly: `valid_base_mse`
  `0.0211334` vs `0.0213502`, raw RMSE `0.00273058` vs `0.0027353`,
  zRMSE `0.0644722` vs `0.0646257`.
- P4 best/final still do not beat S5 base best or S4 best on `valid_base_mse`:
  P4 best `0.0211334`, S5 base best `0.0210238`, S4 best `0.0197146`.
- Final-probe means regress slightly versus S5final FT no-mask final:
  P4 best RMSE `0.372598`, relRMSE `0.805305`; FT final RMSE `0.372366`,
  relRMSE `0.804419`.
- Keep P4 best as targeted-loss diagnostic evidence. Do not promote P4 final.
  Keep S5final FT no-mask final as the fallback checkpoint for
  final-probe-oriented use.
- A second targeted-loss ablation is only worth doing if it explicitly protects
  final-probe robustness. This conservative hotspot/strong-q pass improved
  in-distribution scalar/raw metrics but did not improve final-probe metrics.

Ignored diagnostic outputs:

- `output/heat3d_v3_targeted_loss_audit/p4_msebest_full_diagnostics/`
- `output/heat3d_v3_final_probe_p4_msebest/`
- `output/heat3d_v3_targeted_loss_audit/s5_family_p4_comparison.json`
- `output/heat3d_v3_targeted_loss_audit/s5_family_p4_comparison.md`
