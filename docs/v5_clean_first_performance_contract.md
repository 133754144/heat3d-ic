# V5 Clean-First Performance Contract

This contract freezes the evaluation language for V5 native shape-scale and
Global FiLM experiments before model changes or warm-start runs begin.

## Frozen Reference And Isolation

- Frozen reference: `V4P5_02_clean_baseline_raw_B28_e600`, best checkpoint
  `params_best.pkl`, epoch 405.
- Fit only `train=672`; fit every data/global-context standardizer only there.
- Select V5 candidates only on `valid_iid=128`.
- `test_iid=128` is report-only. `hard_train_holdout`,
  `hard_challenge_valid`, and `hard_challenge_test` are frozen descriptive
  reports: they cannot affect fitting, normalization, feature/schema choice,
  thresholds, early stopping, or checkpoint selection.

The primary V5 checkpoint is the minimum valid sample-first CV-relative RMSE;
raw CV-weighted RMSE is its deterministic tie-break. A second control
checkpoint is selected by legacy normalized `valid_base_mse` so that V4-style
selection remains directly visible rather than being silently replaced.

## Required Clean Metrics

Every V5 prediction evaluation emits the following for each reported split:

| family | required output |
| --- | --- |
| Relative | point-global relative RMSE; sample-first CV-relative RMSE |
| Raw field | CV-weighted RMSE K |
| Field fidelity | amplitude ratio; CV-weighted centered spatial correlation |
| High temperature | hotspot top-5-percent CV-RMSE; top-five-node CV-RMSE |
| Source-sensitive | strong-positive-q (sample q90) CV-RMSE |
| Background | true-DeltaT-p50 background signed bias, CV-RMSE, and over-ratio |
| Shape/scale | shape CV-RMSE; scale log-RMSE |
| Compatibility | legacy normalized `valid_base_mse` |

`point-global relative RMSE` preserves the V4 denominator: global raw-DeltaT
RMSE divided by global mean absolute true DeltaT. `sample-first CV-relative
RMSE` instead gives each sample one equal vote after evaluating its own
control-volume-weighted error ratio. They must never be collapsed into a
single unnamed "relative RMSE".

The final clean target is less than 20 percent point-global relative RMSE on
both valid and test. This threshold is not a license to select on test.

## V5 Native Semantics

`DeltaT` is raw temperature minus the reference Dirichlet temperature.
`s_true` is control-volume RMS of DeltaT and `phi_true = DeltaT / s_true`.
V5 native predictions reconstruct `DeltaT_hat = s_hat * phi_hat`, recover raw
temperature, and only then project Dirichlet nodes in raw-temperature space.

The V5 metric implementation lives in
`rigno/heat3d_v5_metrics.py`. It is a read-only NumPy evaluator and requires
that all supplied scales remain positive. At inference, learned context may
use only physical inputs, BC, control volumes, and an optional frozen V4
pooled latent; target scale, target shape, residuals, oracle metrics, and any
other label-derived input are forbidden.

See the machine-readable source of truth at
`configs/heat3d_v5/v5_clean_first_performance_contract.json`.
