# Gate 6G low-memory attention preflight

Gate 6G lives on the independent `research/v5-gate6g` branch. It does not
modify the completed V13 output directory. All registered candidates are
scratch e200 plans; this preflight and its e1 smokes use only `train` and
`valid_iid`. Test, hard roles, and sealed IID are forbidden.

## Constant-LR contract

The completed WSL2 V13 `run_config.json` records `lr_peak=2e-4`. Gate 6G does
not inherit V13's warmup/cosine trajectory: every candidate uses the same
independent constant rate

`constant_lr = 0.5 * V13 lr_peak = 1e-4`.

V13's first 200 epochs therefore are not a matched control for these runs.

## Registered e200 candidates

| order | config | isolated path change |
|---:|---|---|
| 1 | `V4P5_22_gate6g_control_constlr` | V13 architecture, mean pooling, stop-gradient off |
| 2 | `V4P5_23_gate6g_stopgrad_constlr` | pooled latent stop-gradient only |
| 3 | `V4P5_24_gate6g_shape_attention_constlr` | shape attention only |
| 4 | `V4P5_25_gate6g_scale_attention_constlr` | independent scale attention plus required stop-gradient |
| 5 | `V4P5_26_gate6g_shape_attention_stopgrad_constlr` | shape attention plus stop-gradient |
| 6 | `V4P5_27_gate6g_deep_scale_head_constlr` | three-layer mean-pooled scale head only |

All six keep V13's dataset, split, B28, graph, model/batch/graph seeds,
optimizer family, weight decay, and `1.5|0.5|1|1` loss. Primary selection is
point-global true-RMS relative RMSE. Point-global, sample-first CV-relative,
legacy valid-base-MSE, and final checkpoints are all enabled with prediction
reload audit metadata.

## Attention paths

Shape attention consumes LayerNorm regional latents after FiLM, input-only
regional raw `coords/k/q/BC` features, and the frozen 24-D global context. Its
zero-initialized residual output acts only on decoder input. Scale attention
has independent parameters and modifies only the scale pooling route as
`mean + zero-initialized attention residual`. Neither path uses targets, and
both remain linear in regional-node count rather than constructing regional
self-attention.

The checker verifies exact default replay, shared control parameters, zero
initial output deltas, gradient routing, input provenance, and resolved-config
diffs. E1 smoke reporting separates process RSS, live device bytes,
reserved-device bytes, and allocator pool bytes.

## Manual e200 launch order

Run only after reviewing the e1 closeout:

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_22_gate6g_control_constlr.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_23_gate6g_stopgrad_constlr.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_24_gate6g_shape_attention_constlr.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_25_gate6g_scale_attention_constlr.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_26_gate6g_shape_attention_stopgrad_constlr.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_27_gate6g_deep_scale_head_constlr.yaml
```

These commands are plans only. Gate 6G preparation leaves
`long_training_started=false`.
