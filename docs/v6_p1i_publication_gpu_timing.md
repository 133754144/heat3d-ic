# P1i publication GPU timing

Production states exclude qualification, fresh graph/map construction, hashing, metrics, and serialization. Every GPU interval ends after device synchronization.

| N | new-case median/p95 (ms) | warm-cache median/p95 (ms) | neural-forward median/p95 (ms) | CPU-GPU recon max (K) |
|---:|---:|---:|---:|---:|
| 4096 | 326.045/338.719 | 3.219/3.490 | 3.216/3.665 | 3.071e-05 |
| 8192 | 346.083/359.568 | 4.327/4.638 | 4.397/4.715 | 3.190e-05 |
| 16384 | 363.402/372.084 | 8.008/8.663 | 8.020/8.547 | 3.203e-05 |
| 32768 | 412.983/423.625 | 18.057/18.399 | 18.047/18.477 | 3.081e-05 |
| 65536 | 470.207/493.953 | 36.221/36.678 | 36.095/36.456 | 3.097e-05 |
