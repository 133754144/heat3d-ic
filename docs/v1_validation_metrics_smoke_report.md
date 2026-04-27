# Heat3D v1 Validation Metrics Smoke Report

## Stage Purpose

This stage establishes a validation metrics smoke loop for the 16-sample
`v1_multilayer_bc_eq_supervised_small` dataset.

The goal is to compare:

- a `zero_delta` baseline
- a tiny trained prediction

This is smoke diagnostics only. It is not a formal benchmark, not model
performance evidence, and not an OOD generalization result.

## Current Pipeline

The current smoke pipeline is:

1. manifest
2. ignored local samples under `data/`
3. smoke-only temperature labels
4. supervised target / batch / zero_delta checks
5. tiny train / valid smoke
6. validation metrics smoke

The default route remains:

- relative BC feature view
- `zero_delta_u_bridge`
- normalized `DeltaT = T - T_ref` target
- recovery by `T_pred = T_ref + DeltaT_pred`

## Metrics Covered

The validation metrics smoke currently reports:

- `raw_deltaT_mse`
- `recovered_T_rmse`
- `recovered_T_mae`
- `recovered_T_max_abs_err`
- `true_peak_T`
- `pred_peak_T`
- `peak_T_abs_err`
- `true_hotspot_index`
- `pred_hotspot_index`
- `hotspot_coord_distance`
- split summary
- repeatability

## Key Smoke Output

The current smoke run completed with:

- train loss: `1.15936840 -> 1.12858367`
- valid loss: `0.85722756 -> 0.83369905`
- train mean recovered T RMSE:
  - zero_delta baseline: `4.111654e-01`
  - tiny trained prediction: `3.251936e-01`
- valid mean recovered T RMSE:
  - zero_delta baseline: `3.590730e-01`
  - tiny trained prediction: `2.943906e-01`
- repeatability: pass
- per-sample metric rows: `26`
- split summaries: `4`

## Valid Conclusion

In the current smoke dataset and tiny training setting, the trained prediction
shows an observable recovered-temperature RMSE decrease relative to the
zero_delta baseline. The code now has a minimal train-to-evaluation loop.

## Non-Claims

This stage does not establish:

- a formal benchmark
- a model performance conclusion
- OOD generalization
- high-fidelity solver labels
- an industrial 3D IC thermal simulation dataset

The `test_ood_bc` and `test_ood_stack` samples are diagnostic smoke candidates
only. They must not be used as evidence of OOD generalization at this stage.

## Next Recommended Step

The next stage should move toward solver fidelity / physics label v2 planning.

The priority should not be more training. The priority should be improving the
physical credibility of the label generator and planning checks for:

- residual estimates
- heat-flux consistency
- boundary-condition consistency
- interface consistency

Those checks should be designed before any stronger benchmark or model
performance claims are made.
