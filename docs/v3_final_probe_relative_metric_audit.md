# Heat3D v3 Final Probe Relative Metric Audit

Scope: S4 checkpointed rerun, `params_best.pkl`, evaluated on the 10-sample
v3 final-target probe. This is a diagnostic readout only, not a formal
benchmark.

Checkpoint:

`output/heat3d_v2_runs/latent96_s6_mlp2_B88_sample_shuffle_nearest_repair_S4_seed0_e600_warmupcosine_lr5e-4_minlr5e-5_wd1e-4_checkpointed_rerun/params_best.pkl`

Metrics source:

`final_probe_eval/best/metrics/s5_probe_metrics.json`

## Main Conclusion

Absolute RMSE is misleading across final-probe samples because each probe has a
different target DeltaT scale. P10 is the clearest example: its absolute RMSE
is small (`0.147 K`), but its diagnostic `relRMSE_DeltaT` is high (`83.5%`).
Therefore final-probe tables should include at least:

- `relRMSE_DeltaT`
- `Tmax_error`
- `Probe Family`
- heat-source descriptor such as `source_category` / `q_power_range`
- P10 unsupported-gap flag

`relRMSE_DeltaT` is the preferred ranking signal for cross-probe comparison.
It should not be described as simply `RMSE / (Tmax - Tmin)`: the diagnostics
normalizes against the target DeltaT field scale, not just the max-min range.

## Relative Ranking

| Rank | Probe | Test focus | RMSE | relRMSE | Tmax err | Family |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | P03 | low-k hotspot confinement | 0.974 | 91.8% | -7.05K | random block |
| 2 | P09 | diag3 anisotropic spreading | 0.545 | 86.5% | -5.75K | anisotropic |
| 3 | P02 | sparse high-k bridge + high dynamic heat | 0.616 | 85.6% | -5.84K | random block |
| 4 | P10 | extreme top-h boundary extrapolation | 0.147 | 83.5% | -1.70K | extreme BC |
| 5 | P04 | multi-scale high-contrast interfaces | 0.394 | 82.6% | -3.06K | random block |
| 6 | P08 | IC hotspot motif in random material background | 0.206 | 76.7% | -2.67K | IC motif |
| 7 | P05 | random volumetric heat blobs | 0.237 | 76.7% | -1.16K | volumetric |
| 8 | P01 | non-layered high/low-k conductivity routing | 0.251 | 76.5% | -2.04K | random block |
| 9 | P07 | TSV-like vertical high-k heat path | 0.120 | 70.0% | -1.67K | IC motif |
| 10 | P06 | elongated source plus weak background | 0.202 | 66.7% | -1.14K | volumetric |

## Interpretation

- No final-probe sample is genuinely solved: all `relRMSE_DeltaT` values remain
  high, roughly `66%` to `92%`.
- P03, P02, and P09 remain the strongest failure modes: local hotspot
  confinement, high-dynamic compact heat sources, and anisotropic/diag3
  spreading.
- P10 must remain caveated. The current v0 probe only implements global top
  Robin very-high-h behavior; localized top contact and side asymmetry remain
  generator/schema gaps and must not be claimed as tested.
- Future final-probe summary tables should not sort only by absolute RMSE.
  Use `relRMSE_DeltaT` plus `Tmax_error`, and keep `q_region_RMSE` /
  `strong_q_RMSE` available for hotspot-region analysis.
