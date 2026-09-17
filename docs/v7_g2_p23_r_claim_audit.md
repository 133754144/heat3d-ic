# P23-R publication claim audit

The valid-only replay does not authorize an overall winner claim.  Therm-FM's
three-seed replay is internally reproduced, but the Heat3D V6 seed1/seed2
frozen checkpoints are missing, so no six-archive common comparison exists.

## Supported now

- Therm-FM P22 physical metrics can be reproduced on the frozen valid128 route
  (`REPLAY_REPRODUCTION_PASS`).
- The available Heat3D V6 seed0 full-field replay reproduces its historical
  seed0 diagnostic metrics using the frozen V6 mapping.
- The P23 evaluator uses truth-defined top-1% hotspot nodes and correct raw
  sufficient-statistic point-global aggregation.

## Not supported / withheld

- Heat3D-versus-Therm-FM average full-field ranking or paired superiority.
- Hotspot, peak, or condition-specific winner claims.
- Any same-information-budget claim (Therm-FM uses external Poseidon-T
  pretraining and a dense representation).
- Any test/generalization claim involving `test_iid`, sealed IID, or
  DeepOHeat official100.

The paired bootstrap, condition analysis, and Table B remain fail-closed until
the exact frozen V6 seed1 and seed2 checkpoint artifacts are recovered.  They
must be replayed under the already frozen manifest/evaluator; retraining or
checkpoint substitution is not permitted.
