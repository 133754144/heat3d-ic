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

P1g's 1024 points are irregular and frozen per geometry group. The unpadded
graph edge tensor shape therefore differs across groups. Runtime batching
shuffles samples and then groups graph-shape-compatible examples with B28 as a
maximum and `drop_last=false`; this increases the number of epoch batches as
explicitly allowed. The real smoke realized B8 because each frozen geometry
group contains eight BC/power cases.

V6's Global FiLM context retains 24 dimensions. V5 positions 18--20 are
replaced by `log_bottom_h_W_m2K`, `top_T_inf_K`, and `bottom_T_inf_K`; all
standardization is fit on the 768 training samples only. The last physics-gate
regional BC slot consumes normalized `bottom_h`, never a synthetic bottom
fixed-temperature offset.

## Baseline migration

`V6_01_V4best` resolves from `V4P5_02_clean_baseline_raw_B28_e600` with no
model, loss, optimizer, graph, epoch, seed, or selection-metric change.
`V6_02_V5best` resolves from the requested `V4P5_42_canonical`, whose repository
config ID is `V4P5_42_gate6q_objective_only_e600`; its only model metadata
difference is the dimension-preserving V6 Global Context schema.

Both use random initialization, B28/B32/B32, `drop_last=false`, 600 epochs,
and `valid_base_mse` selection. Old final-probe, post-training legacy
diagnostics, and baseline comparison are disabled. The one-batch real-data
smokes performed forward, backward, and one AdamW update without saving state;
both were finite. They are engineering checks, not training results.

Resolved configs:

- `configs/heat3d_v6/resolved/V6_01_V4best.resolved.yaml`
- `configs/heat3d_v6/resolved/V6_02_V5best.resolved.yaml`

Formal manual commands (prepared only, not executed):

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_01_V4best.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v6/V6_02_V5best.yaml
```
