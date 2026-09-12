# G2-E2 Heat3D epoch accounting

Status: `PASS_EPOCH_ACCOUNTING_FINITE` (runtime-only, no publication accuracy).

The deterministic-XLA run used the frozen legacy Heat3D path: 768 train
samples, B24 (32 optimizer updates), 128 valid samples in four B32 batches,
and one model forward per validation batch.  The raw remote JSON is retained
outside Git at `/tmp/g2_e2_heat3d_epoch_accounting_det.json` with SHA256
`978dd7835548113f2fc466b75c05dc4f34bae573392de297a0f3f2110c11b055`.

| component | seconds |
|---|---:|
| preparation (outside epoch) | 428.348777 |
| train total (32 batches) | 2442.249287 |
| validation total (4 batches) | 318.853197 |
| checkpoint serialization | 0.766620 |
| metric aggregation + logging | 0.000382 |
| misc residual | 0.083115 |
| epoch total | 2761.952601 |

The accounting identity closes exactly at the recorded precision:
`T_train + T_valid + T_checkpoint + T_logging + T_misc = T_epoch`, residual
`0.0 s`.  The first train batch included JIT compilation (83.407990 s); 32
batch-specific executables were compiled.  Train batch times were
`[83.407990, 75.750415, 80.743109, 76.759506, 76.929236, 75.534335,
74.993421, 73.378960, 73.335324, 78.555974, 76.525013, 77.965352,
75.209257, 74.403281, 76.657442, 72.547136, 76.806002, 76.743127,
77.782753, 77.155444, 77.096145, 79.228579, 76.087724, 75.991630,
77.163241, 77.343550, 75.727314, 76.046396, 72.818484, 73.342573,
74.589916, 75.547681] s`; validation batch times were
`[85.386932, 77.397342, 77.831096, 78.234116] s`.

The RTX 5070 snapshot was 11,552 MiB used of 12,227 MiB after validation;
JAX peak bytes in use were 3,523,862,528.  `XLA_FLAGS` was explicitly
`--xla_gpu_deterministic_ops=true`.  No test/sealed file or accuracy metric was
opened.  Multiplying the measured epoch by the frozen 200-epoch budget gives
153.44 h per seed, so Heat3D remains `BLOCKED_RUNTIME_OVER_TARGET`; this is an
execution-stack observation, not a license to change B24, architecture, or
the epoch budget.

The earlier non-deterministic probe is superseded because its shell had an
unset XLA flag; it is not used for projection.
