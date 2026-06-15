# Heat3D v3 Decoder / Processor Long-Run Audit

Scope: read-only checkpoint export and diagnostics for completed decoder/model-path
long tests. No training was started in this audit, no final-probe labels were
generated, and no output artifacts are committed.

## Offline Completion

| run | server | completed offline outputs |
| --- | --- | --- |
| D1-L400 `latent96_s6_mlp3` | devbox | `predictions.npz`, `best_predictions.npz`, full final/best diagnostics, region decomposition, final-probe checkpoint comparison |
| D3-L200 `latent96_s8_mlp2` | WSL2 | `predictions.npz`, `best_predictions.npz`, full final/best diagnostics, region decomposition, final-probe checkpoint comparison |

Ignored output locations:

- `output/heat3d_v3_decoder_long_audit/d1l_e400_post_training_diagnostics/`
- `output/heat3d_v3_final_probe_d1l_e400/`
- `output/heat3d_v3_decoder_long_audit/d3l_e200_post_training_diagnostics/`
- `output/heat3d_v3_final_probe_d3l_e200/`

## Main Metrics

| run | label | epoch | valid_iid/base | valid_stress/base | raw RMSE | raw MAE | zRMSE | top-k | peak_rel | final-probe RMSE | P02 RMSE | P03 RMSE | P09 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S5 base best | best | 1527 | 0.0210238 | 0.0291828 | 0.00282356 | n/a | 0.0662843 | 0.948828 | 0.0320597 | 0.372716 | 0.631598 | 0.980921 | 0.547795 |
| S5 base final | final | 1600 | 0.0212054 | 0.0289898 | 0.0027924 | n/a | 0.0656913 | 0.950586 | 0.0327246 | 0.372552 | 0.631385 | 0.980932 | 0.547593 |
| S5final FT no-mask final | final | 100 | 0.0213502 | 0.0289516 | 0.0027353 | n/a | 0.0646257 | 0.951758 | 0.0312176 | 0.372366 | 0.631203 | 0.980769 | 0.54755 |
| D3-L200 `s8_mlp2` | best | 129 | 0.0210456 | 0.0282934 | 0.0029171 | 0.00165816 | 0.0706544 | 0.949609 | 0.0328254 | 0.372372 | 0.629985 | 0.979456 | 0.547013 |
| D3-L200 `s8_mlp2` | final | 200 | 0.0210686 | 0.0282471 | 0.00288647 | 0.00163844 | 0.0696691 | 0.949219 | 0.0324199 | 0.372393 | 0.630046 | 0.97959 | 0.546957 |
| D1-L400 `s6_mlp3` | best | 394 | 0.101528 | 0.132198 | 0.013766 | 0.0110989 | 0.346773 | 0.827734 | 0.0772344 | 0.38429 | 0.626469 | 0.988155 | 0.568773 |
| D1-L400 `s6_mlp3` | final | 400 | 0.101952 | 0.133271 | 0.0137787 | 0.0111169 | 0.343085 | 0.826172 | 0.0799021 | 0.384137 | 0.626375 | 0.988114 | 0.568616 |

## Interpretation

- D3-L200 is a useful processor-depth result. It improved over the earlier D3
  e50 trajectory and reaches S5-family scalar scale: best `valid_iid/base`
  `0.0210456`, close to S5 base best `0.0210238`.
- D3-L200 does not beat the strongest S5-family raw mechanism rows. Its final
  raw RMSE/zRMSE/top-k are `0.002886` / `0.06967` / `0.9492`, while S5final
  FT no-mask final is `0.002735` / `0.06463` / `0.9518`.
- D3-L200 is slightly better on some final-probe focus metrics such as P02/P09,
  but the mean final-probe RMSE remains effectively tied with S5-family runs.
  This is diagnostic evidence for processor depth, not a new default model.
- D1-L400 is not a promising continuation path. Even after 400 epochs, its
  best `valid_iid/base` is `0.1015` and raw shape metrics remain far behind
  S5/D3. The added decoder MLP layer plus params-only partial load did not
  recover the S5 checkpoint behavior.
- Keep D3-L200 as evidence that processor depth can preserve S5-level scalar
  behavior with slightly different final-probe tradeoffs. Do not extend D1-L400
  without a separate initialization or architecture-transfer plan.

## Next Decision

Prioritize processor-path analysis over wider decoder MLP depth:

- If another model-path long test is run, prefer a controlled `processor_steps=8`
  continuation or a checkpoint/replay diagnostic over more `mlp_hidden_layers`
  depth.
- Before promoting D3, compare condition-wise weak groups against S5final FT
  no-mask and P4 targeted-loss best; the aggregate metrics are close enough that
  condition-level tradeoffs should decide.
- Keep P10 marked as unsupported final-probe schema gap: localized top contact
  and side asymmetry remain unsupported.
