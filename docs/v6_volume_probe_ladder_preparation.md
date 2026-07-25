# V6 volume-representative probe ladder preparation

Status: **prepared, not evaluated**. No training, optimizer update, checkpoint
selection, or formal model inference was executed.

## Sampling contract

The frozen 1024/2048/4096/8192 solver-node probes use one label-independent
nested circular systematic PPS design. Solver nodes are first placed under a
fixed-seed permutation to prevent structured-grid aliasing; systematic
thresholds then sample in proportion to the true solver control volume.

For resolution `n`:

- inclusion probability: `pi_i = n * CV_i / sum_all_solver(CV)`;
- expansion weight: `w_i = CV_i / pi_i = sum_all_solver(CV) / n`;
- metric integration: Horvitz--Thompson expansion weights estimate full solver
  control-volume integrals;
- source-dense quota: `0`;
- forbidden selection inputs: q, temperature, source layout/metadata, split
  labels, predictions, and errors.

The 4096 probe is therefore independent of the earlier diagnostic support that
reserved 50% of its nodes for source-allowed layers.

| nodes | top | bottom | all 9 layers | all 8 interfaces | max layer-volume fraction error |
|---:|---:|---:|:---:|:---:|---:|
| 1024 | 9 | 25 | yes | yes | 0.021578 |
| 2048 | 16 | 55 | yes | yes | 0.010061 |
| 4096 | 29 | 115 | yes | yes | 0.011526 |
| 8192 | 55 | 223 | yes | yes | 0.006441 |

A post-freeze, valid-only q-coverage audit did not influence support selection.
The minimum source-node counts per valid sample were 4, 8, and 18 at
1024/2048/4096; because supports are nested, 8192 cannot have lower coverage.
No test/hard q or target was read.

Frozen artifacts:

- `configs/heat3d_v6/v6_volume_representative_probe4096.json`
- `configs/heat3d_v6/v6_volume_representative_probe_ladder.json`
- `scripts/check_heat3d_v6_volume_probe_ladder.py`
- `scripts/evaluate_heat3d_v6_volume_probe_ladder.py`

## Evaluator outputs

The evaluator is restricted to V6_03/V6_04 point-global checkpoints and
`valid_iid`. For each model it reports:

- CV-weighted point-global and sample-first relative RMSE;
- raw CV-weighted RMSE K;
- peak and source-region error;
- layer mean/drop error;
- top/bottom surface error;
- graph construction time, inference time, and peak device memory when the
  backend exposes memory statistics.

The 1024/2048/4096/8192 evaluation commands below are manual commands only and
were not executed in this preparation. Replace `<RUN_INPUT_ROOT>` with a
directory containing the frozen V6_03/V6_04 run artifacts and
`<DATASET_ROOT>` with the canonical P1h directory.

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
MPLCONFIGDIR=/tmp/heat3d-mpl XDG_CACHE_HOME=/tmp/heat3d-cache \
python scripts/evaluate_heat3d_v6_volume_probe_ladder.py \
  --dataset <DATASET_ROOT> --input-root <RUN_INPUT_ROOT> \
  --resolution 1024 --batch-size 1 \
  --output-json v6_volume_probe1024_results.json --write
```

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
MPLCONFIGDIR=/tmp/heat3d-mpl XDG_CACHE_HOME=/tmp/heat3d-cache \
python scripts/evaluate_heat3d_v6_volume_probe_ladder.py \
  --dataset <DATASET_ROOT> --input-root <RUN_INPUT_ROOT> \
  --resolution 2048 --batch-size 1 \
  --output-json v6_volume_probe2048_results.json --write
```

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
MPLCONFIGDIR=/tmp/heat3d-mpl XDG_CACHE_HOME=/tmp/heat3d-cache \
python scripts/evaluate_heat3d_v6_volume_probe_ladder.py \
  --dataset <DATASET_ROOT> --input-root <RUN_INPUT_ROOT> \
  --resolution 4096 --batch-size 1 \
  --output-json v6_volume_probe4096_results.json --write
```

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
MPLCONFIGDIR=/tmp/heat3d-mpl XDG_CACHE_HOME=/tmp/heat3d-cache \
python scripts/evaluate_heat3d_v6_volume_probe_ladder.py \
  --dataset <DATASET_ROOT> --input-root <RUN_INPUT_ROOT> \
  --resolution 8192 --batch-size 1 \
  --output-json v6_volume_probe8192_results.json --write
```

To evaluate only one model, append either:

```text
--model V6_03_V5best_P1h
--model V6_04_V5best_P1h_DualAttention
```
