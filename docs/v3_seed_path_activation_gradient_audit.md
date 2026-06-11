# Heat3D v3 Seed Path Activation/Gradient Audit

Purpose: short B88 sample_shuffle nearest_repair audit for model_seed
`0,1,4,6`. This was a 20 epoch diagnostic run only; no e400 training was run.

Ignored devbox outputs:

- `output/heat3d_v3_seed_path_instrumented/seed_path_instrumented_smoke.json`
- `output/heat3d_v3_seed_path_instrumented/seed_path_instrumented_smoke.md`

| seed | e20 valid_iid | e20 rel RMSE | e20 output amp ratio | e20 processor rel update | clip ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2885 | 0.8302 | 0.8079 | 2.0330 | 0.9625 |
| 1 | 0.9629 | 1.5167 | 0.2743 | 1.8535 | 0.4875 |
| 4 | 1.1245 | 1.6391 | 0.02369 | 1.8654 | 0.0000 |
| 6 | 0.9688 | 1.5214 | 0.1840 | 1.8115 | 0.1875 |

## Key Observations

- The split appears immediately. seed0 drops from valid_iid `1.1436` to
  `0.3653` by epoch 10, while seed1/4/6 remain near `0.99-1.13`.
- seed0 output amplitude grows from `0.2123` at init to `0.8079` by epoch 20.
  Failed seeds remain much lower: seed1 `0.2743`, seed6 `0.1840`, seed4 only
  `0.02369`.
- seed4 is the clearest amplitude/gradient collapse case. Its epoch-20
  encoder/processor/decoder/output grad norms are `0.0335/0.0058/0.0066/0.1917`;
  most learning pressure is confined to the output path.
- Processor relative updates are nonzero for all seeds, so the processor is not
  a pure no-op. The failure looks more like low-amplitude decoder/output
  trajectory plus weak encoder/processor gradient flow than missing graph
  connectivity.
- Gradient clipping is frequent for seed0 and partial for seed1/6, but absent
  for seed4. That supports an initialization/path-scale problem rather than a
  simple high-gradient instability.

## Next Judgment

The evidence supports moving to the dummy-init audit before more e400 sweeps.
If dummy init improves seed1 early amplitude and gradients, prepare a guarded
dummy-init e400 YAML. If it does not, the next higher-value path is P5-style
decoder/local path work or a deeper initialization/capacity audit.
