# Heat3D v2 Next e300 / B96 Plan

- Configs prepared: M1 B192 base-MSE e300 and M1 B96 base-MSE e100 control.
- M1 e300 purpose: extend the current M1/B192 scalar-loss baseline to test whether longer controlled training improves the e200 plateau.
- M1 B96 e100 purpose: separate B96/update-dynamics effects from the earlier M1.5 capacity effect; this is not the default baseline.
- Stdout simplification: compact training logs now keep only loss, one relative DeltaT error percentage, and best-valid tracking; full metrics remain in `loss_summary.json` and post-hoc diagnostics.
- No training executed in this preparation step.
