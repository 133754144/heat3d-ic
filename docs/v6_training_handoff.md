# V6 canonical training handoff

## Canonical dataset and physical scale

Only `heat3d_v6_p1g_geometry_deconfounded1024_v0` is accepted for V6-layer
training. P1a--P1f remain tracked and are marked `archived`; P1g-v1 remains
tracked but is non-canonical. No raw directory was deleted in this handoff.

| Quantity | Canonical P1g-v0 range |
|---|---:|
| Package footprint / total stack thickness | 10 x 10 mm / 4.175 mm |
| Layers / layer thickness | 9 / 50--1600 um |
| Solver mesh / operator points | 65 x 65 x 57 = 240825 / 1024 |
| kx, ky | 0.2--400 W/(m K) |
| kz | 0.3--400 W/(m K) |
| top h / bottom h | 1000 or 1400 / 20 or 120 W/(m2 K) |
| top / bottom ambient | 300 / 300 K |
| Package power | 4 or 6 W |
| Source count | 3--10 |
| Total source area | 15.519--39.579 mm2 |
| Per-source area | 2.191--6.004 mm2 |
| Per-source power | 0.226--3.000 W |
| Source surface power density | 7.933--59.301 W/cm2 |
| Volumetric q | 5.570e8--5.354e9 W/m3 |
| Source control-volume count | 264--1008 |
| peak DeltaT / mean DeltaT | 29.378--76.166 K / 24.444--58.120 K |
| Rth_peak | 7.344--12.694 K/W |
| top / bottom heat fraction | 93.770--98.723% / 1.277--6.230% |

The acceptance amendment leaves the original qualification JSON unchanged.
The new training gate passes: `<29 K=0`, `<30 K=18/1024=1.758%`,
`30--80 K=1006/1024=98.242%`, `>80 K=0`, and `>100 K=0`.

## Loader and leakage contract

The dedicated loader reads the canonical manifest and locks whole `group_id`
values to `train=768`, `valid_iid=128`, and `test_iid=128`. Its node condition
has 11 channels: `kxyz`, `q`, four mutually exclusive BC flags, `top_h`,
`bottom_h`, and `top_T_inf-T_ref`. `T_ref` is the prescribed bottom Robin
ambient, not a fixed bottom surface temperature. There is no Dirichlet
projection; the native branch uses an all-zero projection mask.

P1g's 1024 points are irregular and frozen per geometry group. Runtime batching
uses one real B24 graph batch per optimizer update. Within each B24, only the
existing dummy graph edge is repeated to pad edge tensors to the batch maximum;
all real nodes and edges remain unchanged. One B24 forward/backward is followed
by one clipping operation and one AdamW update. With train=768 and
`drop_last=false`, every epoch is exactly 32 B24 batches and 32 updates, with no
tail batch or geometry-based split.

V6's Global FiLM context retains 24 dimensions. V5 positions 18--20 are
replaced by `log_bottom_h_W_m2K`, `top_T_inf_K`, and `bottom_T_inf_K`; all
standardization is fit on the 768 training samples only. The last physics-gate
regional BC slot consumes normalized `bottom_h`, never a synthetic bottom
fixed-temperature offset.

## Baseline migration

`V6_01_V4best` resolves from `V4P5_02_clean_baseline_raw_B28_e600` with no
model, loss, optimizer, graph, epoch, seed, or selection-metric change.
`V6_02_V5best` resolves directly from the frozen
`configs/heat3d_v5/V4P5_42_canonical.yaml`; its only model metadata difference
is the dimension-preserving V6 Global Context schema.

Both use random initialization, train/validation/prediction B24/B32/B32,
`drop_last=false`, and
600 epochs. V6_01 preserves `valid_base_mse`; V6_02 preserves canonical
`valid_rel_rmse_v4_pct` (point-global true-RMS relative RMSE). Old final-probe,
post-training legacy diagnostics, and baseline comparison are disabled.

Resolved configs:

- `configs/heat3d_v6/resolved/V6_01_V4best.resolved.yaml`
- `configs/heat3d_v6/resolved/V6_02_V5best.resolved.yaml`

## Prior B28 two-host GPU preflight reference

The preflight contract is one epoch only: V6_01 runs on `devbox` and V6_02
runs on `wsl2`. It materializes only `train` and `valid_iid`; the manifest's
test IDs are integrity-checked but are not graphed, predicted, or used for
selection. These engineering results are not formal model-performance results
and do not authorize an e600 launch.

Checkpoint replay requires an exact deserialized parameter tree, an exact
saved-prediction NPZ, a replay maximum absolute difference no greater than
0.1 K, and a whole-field replay RMSE no greater than 0.01 K. The two floating
point tolerances cover non-bitwise-deterministic GPU scatter/reduction order;
the exact parameter and NPZ checks prevent a different checkpoint or saved
prediction payload from being accepted. The recovery command is inference
only and writes a new directory, leaving the original e1 outputs and failed
attempt logs untouched.

Frozen evidence and its checker:

- `configs/heat3d_v6/v6_training_handoff_gpu_preflight.json`
- `scripts/check_heat3d_v6_gpu_preflight.py`

| Preflight | Host | Updates | Peak GPU memory | First / steady update | Reload max / RMSE |
|---|---|---:|---:|---:|---:|
| V6_01 V4 baseline | devbox | 28 | 2569.29 / 9169.92 MiB | 64.48 / 29.99 s | 0.01785 / 0.00168 K |
| V6_02 V5 baseline | wsl2 | 28 | 2894.99 / 9169.92 MiB | 104.99 / 40.36 s | 0.04620 / 0.00562 K |

Both historical runs used 1024 nodes, B28 effective updates from graph-compatible B8
microbatches, and one retained B12 update; the B12 position may move under
sample shuffling. Losses, gradients, updates, checkpoints, and prediction
arrays were finite. V6_02's 24-dimensional context standardizer was fit on
the 768 train samples only. No test graph or prediction was materialized.

For engineering visibility only, the e1 valid point-global true-RMS relative
RMSE was 24.4995% for V6_01 and 40.6509% for V6_02. These random-initialized
one-epoch values must not be used to rank the migrated baselines. Earlier
failed or superseded output and log directories remain on their source hosts;
the JSON records their paths and reasons.

## Effective-B24 two-host GPU preflight

The formal migration now uses one exact `3 x B8 -> B24` update. The bounded e1
preflights used the same scientific configurations as the formal YAMLs, with
only the preflight runtime identity and `epochs=1`. Both runs consumed 96
fixed-size B8 microbatches, produced 32 B24 updates, and used four B32
validation batches. There was no B4/B12 tail and no geometry-based batch
split. Test/all roles remained unmaterialized.

Frozen evidence and checker:

- `configs/heat3d_v6/v6_training_handoff_b24_gpu_preflight.json`
- `scripts/check_heat3d_v6_b24_gpu_preflight.py`

| Preflight | Host | Epoch time B24 / B28 | Peak GPU memory B24 / B28 | First / steady B24 update | Reload max / RMSE |
|---|---|---:|---:|---:|---:|
| V6_01 V4 baseline | devbox | 478.96 / 1270.82 s (-62.31%) | 2609.47 / 2569.29 MiB (+1.56%) | 42.83 / 8.82 s | 0.01611 / 0.00163 K |
| V6_02 V5 baseline | wsl2 | 883.70 / 1989.06 s (-55.57%) | 2869.92 / 2894.99 MiB (-0.87%) | 70.15 / 11.46 s | 0.04019 / 0.00421 K |

All recorded loss, gradient, parameter, and update norms were finite. The
checkpoint parameter tree and saved NPZ reloads were exact; independent GPU
prediction replay stayed within 0.1 K maximum and 0.01 K whole-field RMSE.
V6_02's 24-dimensional context was fit on the 768 training samples only. The
formal YAMLs remain at e600 and were not launched.

## B32 selective-launch gate

The B24 YAMLs remain the unified fallback. `V6_01_V4best_B32` and
`V6_02_V5best_B32` change only `run.batch_size` in the scientific payload;
model, loss, optimizer, LR schedule, epochs, graph, seeds, and selection metric
remain frozen. Each B32 update accumulates four pure B8 microbatches, so 768
training samples produce 96 B8 microbatches and 24 B32 updates without a tail
or geometry split.

Before the e5 gate, padded/unpadded inference was checked on two different
geometries. Maximum forward, loss, and gradient differences were respectively
`1.01e-6`, `7.45e-9`, and `1.37e-6`; padding uses no target and repeats only
dummy-to-dummy edges. The frozen gate and checkers are:

- `configs/heat3d_v6/v6_b32_selective_launch_gate.json`
- `scripts/check_heat3d_v6_b32_configs.py`
- `scripts/check_heat3d_v6_dummy_padding_equivalence.py`
- `scripts/run_heat3d_config_with_nvml_monitor.py`

Formal manual commands (prepared only, not executed):

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_01_V4best.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_02_V5best.yaml
```

### B32 e5 gate closeout

Both assigned e5 runs completed with exact `4 x B8 -> B32` accumulation, 96
pure B8 microbatches, 24 optimizer updates per epoch, no tail batch, and no
geometry split. All recorded losses, gradients, parameters, and updates were
finite; checkpoint/export/reload passed. Only `train` and `valid_iid` were
materialized.

| Run | Host | Epoch 2--5 mean | B24 steady reference | Speedup | Allocator peak | Change from B24 | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| V6_01 V4 baseline B32 | devbox | 336.24 s | 444.95 s | +24.43% | 2592.83 MiB | +0.92% | fail: memory growth |
| V6_02 V5 baseline B32 | wsl2 | 862.16 s | 825.01 s | -4.50% | 3020.62 MiB | +4.34% | fail: memory growth and no speedup |

B32 is numerically executable, but it does not pass the pre-registered joint
selection gate. The unified formal choice therefore remains the two B24 YAMLs.
The user deferred formal launch after the e5 closeout, so no e600 process was
started. Frozen evidence and its checker are:

- `configs/heat3d_v6/v6_b32_e5_gate_closeout.json`
- `scripts/check_heat3d_v6_b32_e5_gate_closeout.py`

## Native-B24 launch amendment

The active formal configs now set `batch_size=24` and `micro_batch_size=24`.
This replaces the historical `3 x B8` implementation with 32 real B24
forward/backward passes and 32 optimizer updates per epoch. Model, loss,
optimizer, LR schedule, epochs, graph settings, seeds, selection metric,
validation/prediction B32, and `drop_last=false` remain unchanged. Historical
B8 and B32 preflight artifacts above remain immutable evidence; the native-B24
preflight uses new output directories and is the only launch gate for the
current formal configs.
