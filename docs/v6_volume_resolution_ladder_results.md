# V6 volume resolution ladder formal CPU evaluation

Status: **completed**. Evaluation used only `valid_iid`, batch size 1, `JAX_PLATFORMS=cpu`, and `CUDA_VISIBLE_DEVICES=""`. No training, checkpoint mutation, test, hard, or GPU execution occurred.

The table uses the frozen V6_03 seed0 and V6_04 point-global checkpoints.

| nodes | model | point-global % | sample-first % | raw RMSE K | peak K | source K | layer mean K | layer drop K | top K | bottom K | graph s | inference s | peak RSS GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | V6_03_V5best_P1h | 144.9835 | 148.3902 | 58.5645 | 61.7607 | 62.1612 | 61.5051 | 3.7501 | 62.3613 | 46.5299 | 4.85 | 22.39 | 0.855 |
| 1024 | V6_04_V5best_P1h_DualAttention | 142.2949 | 145.8894 | 57.4784 | 61.0353 | 61.2621 | 60.3125 | 3.6715 | 60.8058 | 45.8609 | 3.06 | 20.18 | 0.935 |
| 2048 | V6_03_V5best_P1h | 138.3550 | 141.5417 | 56.0111 | 59.1662 | 59.3841 | 58.6042 | 3.5902 | 59.2184 | 44.3357 | 9.04 | 24.48 | 1.526 |
| 2048 | V6_04_V5best_P1h_DualAttention | 137.2586 | 140.6195 | 55.5672 | 59.2027 | 59.2864 | 58.1056 | 3.5387 | 58.6889 | 44.1899 | 7.28 | 22.84 | 1.736 |
| 4096 | V6_03_V5best_P1h | 138.3893 | 141.5969 | 56.0618 | 59.0772 | 59.4219 | 58.5809 | 3.6183 | 59.0312 | 44.3660 | 24.85 | 29.56 | 2.035 |
| 4096 | V6_04_V5best_P1h_DualAttention | 137.7815 | 141.2573 | 55.8156 | 58.9935 | 59.4093 | 58.3267 | 3.5471 | 58.7768 | 44.3301 | 23.14 | 27.23 | 2.035 |
| 8192 | V6_03_V5best_P1h | 137.7321 | 140.8902 | 55.7985 | 59.2454 | 59.4347 | 58.4381 | 3.6478 | 58.6131 | 43.8669 | 89.18 | 39.17 | 2.374 |
| 8192 | V6_04_V5best_P1h_DualAttention | 136.8738 | 140.3656 | 55.4508 | 59.0068 | 59.3010 | 58.0465 | 3.5502 | 58.2198 | 43.8321 | 87.79 | 36.41 | 2.534 |

## Stability

- `V6_03_V5best_P1h`: maximum absolute relative aggregate-metric change from 4096 to 8192 is 1.125%; descriptive 2% stability check = `true`.
- `V6_04_V5best_P1h_DualAttention`: maximum absolute relative aggregate-metric change from 4096 to 8192 is 1.123%; descriptive 2% stability check = `true`.

At 8192 nodes, V6_04−V6_03 is -0.8583 percentage points for point-global, -0.5247 percentage points for sample-first, and -0.3477 K for raw CV RMSE.

## Feasibility and interpretation

- 8192 completed on local CPU with maximum observed process peak RSS 2.534 GiB. GPU memory is N/A.
- 8192 is feasible on this host, but graph construction dominates runtime; it is suitable for bounded validation rather than frequent per-epoch evaluation.
- The ladder is stable from 4096 to 8192, but absolute errors remain very large. The result shows strong dependence on the historical source-dense operator support, not a useful model-quality gain.
- V6_04 is modestly better at 8192, but the difference is small relative to the shared failure on the volume-representative support.
- The 2% stability statement is descriptive, not a preregistered promotion threshold, and no checkpoint/model selection was changed.

## Provenance

- evaluator SHA256: `5cc0fc2a03f6b121105f2d2fc768f8309ec9df7cd5296d60e3c618072d4a3ceb`
- base Git HEAD: `7d30b78896c4ca724df40a500191c72686b19070`
- checkpoints were hash-verified; train-only Global Context fit count was 768 for every run.
- raw per-sample valid-only payloads are embedded in the unified JSON.
