# Gate 6R V45/V46 training preparation

Status: `prepared_not_started`. No e600 or multi-seed run was started.

## Frozen contract

Both candidates inherit V38 and keep its clean dataset/split, model capacity,
graph, AdamW warmup-cosine optimizer, B28 training, B32 validation/prediction,
seed0, epoch-wise regrouping, `r2r_only` masking at `p=0.05`, e600 budget,
checkpoint policy, and random initialization with `init_checkpoint: null`.
Training is restricted to `train`; selection is restricted to `valid_iid`.
Test, hard roles, and sealed IID are forbidden.

- V45 adds only `model.scale_deepsets_mode=source_volume_residual` to V38.
  Its objective remains V38's objective and XY scale context remains disabled.
- V46 adds only V42's two objective modes to V45. Its architecture and
  DeepSets path are otherwise identical to V45.

Resolved scientific differences are exactly:

- V45−V38: `model.scale_deepsets_mode`.
- V46−V45: `loss.native_raw_loss_mode` and
  `loss.native_log_scale_weight_mode`.

The DeepSets implementation is the same path used by V44: arithmetic mean,
source-power-weighted, and volume-weighted latent aggregation; source/volume
is divided across each physical node's valid P2R degree. The output projection
is zero initialized and the module adds 28,896 parameters relative to V38.

## Real B28 single-update smoke

The smoke ran on WSL2 at `67df3e2ce0bcef1835f36b2fd24eced3878932a7`
with `MEM_FRACTION=0.85`. It loaded only `train=672`, used random
initialization, executed one forward/backward/AdamW update, and wrote no
checkpoint or output artifact.

| candidate | finite loss/grad/update | parameters | peak memory |
|---|:---:|---:|---:|
| V45 | yes | 922,632 | 5.680900 GiB |
| V46 | yes | 922,632 | 5.680985 GiB |

For both candidates: zero-degree physical nodes = 0, partition-of-unity error
= 0, source conservation error = 0, and volume conservation error =
`2.65e-23`. No test/hard/sealed role was loaded. No training process remained
after the smoke.

## Manual e600 launch commands

Prepared only; not executed:

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_45_gate6r_deepsets_only_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_46_gate6r_objective_deepsets_e600.yaml
```
