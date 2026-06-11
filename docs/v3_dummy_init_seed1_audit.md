# Heat3D v3 Dummy Init Seed1 Audit

Purpose: compare `real_first_batch` vs `upstream_dummy` initialization for
B88 sample_shuffle nearest_repair model_seed1. This was a 20 epoch diagnostic
only; no e400 run was started.

Ignored devbox outputs:

- `output/heat3d_v3_dummy_init_audit/real_first_batch/seed_path_instrumented_smoke.json`
- `output/heat3d_v3_dummy_init_audit/upstream_dummy/seed_path_instrumented_smoke.json`

| init_mode | e20 valid_iid | e20 rel RMSE | e20 output amp ratio | e20 processor rel update | clip ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| real_first_batch | 0.9639 | 1.5176 | 0.2755 | 1.8481 | 0.5125 |
| upstream_dummy | 0.9629 | 1.5167 | 0.2742 | 1.8537 | 0.4875 |

## Checkpoint Comparison

At initialization both modes are effectively identical:

- valid_iid: `1.212438` vs `1.212440`
- relative RMSE: both `1.701985`
- output amplitude ratio: `0.226605` vs `0.226601`
- encoder/processor/decoder/output grad norms are also numerically equivalent.

By epoch 20, `upstream_dummy` is only marginally different from
`real_first_batch` and remains on the same failed trajectory. The e20 valid_iid
loss differs by about `0.001`, and the output amplitude ratio remains near
`0.27`, far below successful seed0's stage-4 e20 value `0.8079`.

## Judgment

Dummy initialization does not rescue seed1 early trajectory under the current
canonical first-shape dummy implementation. It is not worth preparing a
dummy-init e400 YAML from this result.

The stronger signal remains the stage-4 activation/gradient result: failed
seeds have low output amplitude and weak encoder/processor gradient flow early
in training. Next work should focus on initialization/scale or decoder/local
path behavior rather than the model.init input batch choice alone.
