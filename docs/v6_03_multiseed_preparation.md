# V6_03 P1h multi-seed preparation

Status: **passed**. Seed1 and seed2 resolve from the frozen V6_03 seed0
configuration. The only scientific differences are the four registered
seed fields; dataset, model, graph, loss, optimizer hyperparameters, LR,
B24/micro24, e600, and checkpoint selection remain unchanged.

| config | scientific diff | training started |
|---|---|---|
| `V6_03_V5best_P1h_seed1` | `optimizer.batch_order_seed, optimizer.graph_seed, optimizer.model_seed, optimizer.seed` | false |
| `V6_03_V5best_P1h_seed2` | `optimizer.batch_order_seed, optimizer.graph_seed, optimizer.model_seed, optimizer.seed` | false |

## Manual launch commands

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_03_V5best_P1h_seed1.yaml
```

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_03_V5best_P1h_seed2.yaml
```

No training, optimizer update, test/hard access, or formal inference
was performed by this preparation.
