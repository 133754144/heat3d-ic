# Table B — P1i full-field240825 valid-only comparison

Scope: the same frozen `valid_iid=128` physical cases and the same
`65×65×57=240,825` node truth archive.  Heat3D uses the V7 P1i-e200 frozen
checkpoint family plus the retained `U_v2_direct240825` sidecars; Therm-FM is
the loss-selected Poseidon-T transfer cohort.  All `±` values are **SD across
training seeds**.

| Model | Params | Representation / pretraining | sample-first rel. RMSE % | point-global rel. RMSE % | CV-weighted point-global % | RMSE [K] | MAE [K] | Corr_point [%] | Amp_range [%] | Corr_CV [%] | Amp_CVRMS [%] | peak MAE [K] | hotspot RMSE [K] |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat3D V7 P1i-e200 + U-v2 direct | 892,776 | sparse physical conditioning → direct full-field query; no external pretraining | 3.1262 ± 0.1772 | 3.4786 ± 0.1728 | 3.3854 ± 0.1846 | 2.7625 ± 0.1372 | 2.0976 ± 0.1171 | 98.1022 ± 0.2124 | 96.0984 ± 0.3790 | 98.9563 ± 0.0819 | 97.3367 ± 0.2102 | 3.8028 ± 0.0827 | 5.7183 ± 0.3287 |
| Therm-FM / Poseidon-T | 21,435,546 | dense 741-channel physical input; external Poseidon-T pretraining | 2.5972 ± 0.2399 | 3.1490 ± 0.4603 | 3.1972 ± 0.4514 | 2.5007 ± 0.3656 | 1.6363 ± 0.1963 | 97.4701 ± 0.1930 | 123.0210 ± 2.6190 | 98.2852 ± 0.1678 | 100.0987 ± 0.0140 | 3.9503 ± 0.1629 | 3.9860 ± 0.1704 |

Therm-FM is an **external-pretrained dense-input capability reference**, not a
same-information-budget baseline.  The checkpoint rule is asymmetric by
design: Heat3D uses its frozen valid-iid selection; Therm-FM reports
`metrics_at_loss_selected_checkpoint` (minimum valid normalized loss), not an
independent optimum for each physical metric.  Per-seed diagnostics include
`mean |Amp_range-1|`, `mean |Amp_CVRMS-1|`, `scale_log_error` mean/RMSE,
signed temperature bias, field standard-deviation ratio, legacy top-5 overlap,
Metric meanings are fixed: peak MAE/RMSE are the MAE/RMSE of the per-sample
scalar `abs(max(T_pred)-max(T_true))`; hotspot RMSE is field RMSE on the truth
top-1% (2409 nodes) region; true-hotspot top-1% overlap is a localization
metric. They must not be conflated.

The reconciled conclusion is descriptive: Therm-FM is lower on sample-first,
MAE, and hotspot-region RMSE; Heat3D has slightly lower scalar peak error and
higher correlation/true-hotspot overlap; point-global/RMSE and peak-error paired
intervals cross zero. There is no overall winner and no same-information-budget
claim.

The retained Heat3D sidecars are bound through immutable per-seed receipts and
are not re-inferred in this phase.  The historical checkpoint→P1i replay
`FAIL_CLOSED` receipt is preserved; this table therefore claims frozen
sidecar identity/common-evaluator evidence, not a fresh checkpoint-inference
reproduction.
