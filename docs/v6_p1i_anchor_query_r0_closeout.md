# P1i three-seed 1024 R0 equivalence gate

Status: **passed_three_seed_dual_backend_prediction_level_equivalence**.

The hard gate is prediction-level. Aggregate metric agreement is only a secondary consistency check.

| seed | epoch | adapter-reference max K | archived RMSE K | full-field archived RMSE K | support PG % | full PG % |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 559 | 0 | 0.00299622551 | 0.00143458716 | 2.109864 | 3.459794 |
| 1 | 455 | 0 | 0.00231727673 | 0.00174927461 | 2.048143 | 3.490425 |
| 2 | 587 | 0 | 0.00223781286 | 0.00158587997 | 1.923985 | 3.377540 |

All original support/order/features, real-edge graph semantics, anchor-derived context/native scale/QK inputs, predictions, and 240825-node reconstructions passed the frozen checks.

Roles: train inputs were used only to replay the frozen train-only standardizer; valid_iid was evaluated. test/sealed were not accessed. No training, tuning, checkpoint mutation, or high-N inference occurred.
