# V7 G2-P19：valid-only inference profiling aggregation

仅使用冻结 valid128 输入；不读取 target/truth，不训练，不访问 test_iid、sealed 或 DeepOHeat official100。
U-v2 统一称 `direct-query dense inference`。所有时间均在同一 devbox GPU、显式 `jax.block_until_ready` 边界下记录。

| checkpoint | cold E2E (s) | steady/model-only (s) | cached E2E median (s) | peak live GiB |
|---|---:|---:|---:|---:|
| Heat3D e600 fixed endpoint | 18.386050 | 8.873593 steady E2E | 0.473031 | 0.461 |
| DeepOHeat best | 3.096993 | 0.001014 model-only | 0.004822 | 0.035 |
| DeepOHeat final | 1.467455 | 0.001835 model-only | 0.004656 | 0.035 |

Heat3D phases: query-graph construction, model forward, postprocess and steady/cached E2E are retained in the JSON receipt. DeepOHeat reports model-only and E2E separately. The table uses each phase's live row maximum; JAX cumulative process peak is retained separately.
The IDW utility is documented separately as a historical Heat3D V6/P1h diagnostic and is not used to rename or interpret U-v2 timing.

Raw receipt SHA256: `baea3fa13f51529bcce36f6594e7632d18c44e62b47654c82704e5cacbcb6594`; runner commit: `f9cd8b59512b42b609bd28fb88b8cd4fb0111e73`.
