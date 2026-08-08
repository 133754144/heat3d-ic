# P1i Anchor-derived High-N valid32 failure closeout

N=1024 passed. N=4096 was stopped by the frozen cross-backend real-edge topology gate; N=8192/16384 were not started.

| N | status | support PG % | support SF % | support raw K | full PG % | full raw K | oracle floor PG % | GPU replay RMSE K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | passed | 1.754447 | 1.410423 | 1.294950 | 3.071968 | 2.930196 | 2.489856 | 0.001699529 |
| 4096 | failed | 2.966938 | 2.539822 | 2.107667 | 2.823751 | 2.638147 | 1.488771 | 0.001415467 |

## Root cause

CPU and GPU agree on the 1024 real-edge topology. At 4096, platform float normalization changes regional radii enough to alter p2r/r2p/r2r edge sets. Same-backend cache checks pass, so this is neither cache corruption nor a model-accuracy failure.

The frozen binding, sampler, checkpoint, graph parameters and metrics were not changed. Test/sealed remained closed.

## Provenance

- execution commit: `91131e482c3f8ed0e2d621c74d86e7007006e08d`
- closeout code commit: `7c0ee10c63a8deb2576f34319d1d4296cb9f73d4`
- frozen binding SHA256: `80872389351eda59c15386bc205dd0ee1067ebf8c1c0be63b4085100360b8ede`
- frozen checkpoint SHA256: `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`
- failure bundle SHA256: `b090cf9fbbe3ab05adaef8dc8dee2b06e0069891687fb9b9954ab3408eda341e`
