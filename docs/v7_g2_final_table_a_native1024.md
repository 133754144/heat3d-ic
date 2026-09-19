# Table A — P1i native1024 valid-only comparison

Scope: the frozen `valid_iid=128` cases, 1,024 native query points per case,
and the exact per-seed checkpoint selections recorded in
`docs/v7_g2_final_evidence_freeze.json`.  All `±` values are **SD across
training seeds** (three independent seeds), not a case-level confidence
interval.

| Model | Params | sample-first rel. RMSE % | point-global rel. RMSE % | RMSE [K] | MAE [K] | Corr_point [%] | Amp_range [%] | Corr_CV [%] | Amp_CVRMS [%] | peak MAE [K] | hotspot RMSE [K] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D V7 P1i-e200 | 892,776 | 1.6919 ± 0.0478 | 2.0577 ± 0.0806 | 1.6045 ± 0.0628 | 1.0250 ± 0.0254 | 99.1041 ± 0.0868 | 97.9125 ± 0.3784 | 99.2716 ± 0.0548 | 99.8791 ± 0.2151 | 2.4725 ± 0.0530 | 4.0528 ± 0.1708 |
| GINO | 13,673,988 | 15.0139 ± 0.9130 | 17.7315 ± 1.6551 | 13.8267 ± 1.2906 | 10.1083 ± 0.7615 | 96.5395 ± 1.7286 | 118.8116 ± 3.3883 | 96.2965 ± 1.6137 | 102.8707 ± 0.1361 | 12.2843 ± 0.7759 | 16.1072 ± 1.1701 |
| Transolver | 716,737 | 16.0028 ± 1.0310 | 18.5565 ± 0.9266 | 14.4701 ± 0.7225 | 10.6927 ± 0.5580 | 96.9161 ± 0.2167 | 95.0353 ± 0.5106 | 97.2029 ± 0.2197 | 101.9345 ± 1.7185 | 11.8688 ± 0.5546 | 15.8520 ± 0.7962 |

The complete per-seed rows and additional diagnostics (`scale_log_error`,
signed bias, field standard-deviation ratio, top-5 overlap and true-hotspot
top-1% overlap) are in
`docs/results/v7_g2_final_evidence_aggregates.json`.  This table is not a
full-field comparison; Table B uses the separate 240,825-node domain.
