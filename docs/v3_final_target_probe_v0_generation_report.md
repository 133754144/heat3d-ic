# Heat3D v3 Final-Target Probe v0 Generation Report

## Scope

This report records the first 10-sample v3 final-target probe preparation. It is
not a formal validation set, benchmark, model-performance claim, or
publication-ready dataset.

Generated subset:

`data/heat3d-thermal-simulation/subsets/v3_final_target_probe_v0`

Committed control files:

- `configs/heat3d_v3_final_target_probe_manifest_v0.json`
- `tools/generate_heat3d_v3_final_target_probe.py`
- `scripts/check_heat3d_v3_final_target_probe.py`
- `scripts/visualize_heat3d_v3_final_target_probe.py`

Generated data stay under ignored `data/` and should not be committed.

## Generation Summary

Command:

```bash
python3 tools/generate_heat3d_v3_final_target_probe.py \
  --manifest configs/heat3d_v3_final_target_probe_manifest_v0.json \
  --output-subset data/heat3d-thermal-simulation/subsets/v3_final_target_probe_v0 \
  --write \
  --overwrite
```

Checker:

```bash
python3 scripts/check_heat3d_v3_final_target_probe.py \
  --manifest configs/heat3d_v3_final_target_probe_manifest_v0.json \
  --subset data/heat3d-thermal-simulation/subsets/v3_final_target_probe_v0 \
  --expected-count 10 \
  --output-json output/heat3d_v3_final_target_probe/check_v0.json
```

Checker result: pass.

- sample count: 10 / 10
- resolution: 1024 points, grid shape `16 x 16 x 4`
- labels generated: 10 / 10
- max solver residual norm: `1.2162285718826292e-16`
- max bottom Dirichlet error: `0.0`
- duplicate `q/k/T` hashes: none detected
- paired 4096 samples: deferred for all 10 scenes

Visualization:

```bash
python3 scripts/visualize_heat3d_v3_final_target_probe.py \
  --subset data/heat3d-thermal-simulation/subsets/v3_final_target_probe_v0 \
  --output-dir output/heat3d_v3_final_target_probe/figures
```

Output:

- figure directory: `output/heat3d_v3_final_target_probe/figures/`
- generated figures: 31 PNG files
- figure manifest: `output/heat3d_v3_final_target_probe/figures/figure_manifest.json`
- per probe: 3D scatter, z-mid slice, and source-near slice for `k/q/T`
- P09 extra: `kx/ky/kz` source-slice figure

## Probe Table

| probe | family | stressor | k mode | source | BC | label status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | random material-block composite | non-layered conductivity routing | iso1 | multi-block power | nominal top-h | generated | high/low-k random blocks |
| P02 | random material-block composite | disconnected conduction paths | iso1 | compact hotspot + weak background | nominal top-h | generated | sparse high-k bridges |
| P03 | random material-block composite | local hotspot confinement | iso1 | contained hotspot | low top-h | generated | low-k barrier boxes |
| P04 | random material-block composite | multi-scale material discontinuity | iso1 | multi-block power | high top-h | generated | nested multi-scale interfaces |
| P05 | random volumetric source | non-IC source topology | iso1 | multi-blob power | nominal top-h | generated | ellipsoid-like blobs |
| P06 | random volumetric source | anisotropic power distribution | iso1 | elongated power | nominal top-h | generated | strip plus weak background |
| P07 | IC motif random background | vertical heat escape | iso1 | via-adjacent hotspot | high top-h | generated | TSV-like scalar high-k path |
| P08 | IC motif random background | IC source off manifold material context | iso1 | active hotspot motif | nominal top-h | generated | hotspot motif in random blocks |
| P09 | anisotropic/tensor-k patch | tensor-like spreading mismatch | diag3 | patch-adjacent hotspot | nominal top-h | generated | diag3 only; full tensor-k deferred |
| P10 | extreme BC/contact | V1 top-h extrapolation | iso1 | compact hotspot | very high top-h | generated | localized contact/side asymmetry deferred |

## Checker Statistics And Figures

| probe | q frac | k clusters | T min/max/mean/std K | BC | figures | notes |
| --- | ---: | ---: | --- | --- | --- | --- |
| P01 | 0.031250 | 8 | 300.000/302.528/300.154/0.290 | nominal_top_h | `v3_probe_P01_r1024_3d_scatter.png`<br>`v3_probe_P01_r1024_source_slice.png`<br>`v3_probe_P01_r1024_zmid_slice.png` | high/low-k block composite |
| P02 | 1.000000 | 11 | 300.000/306.496/300.318/0.645 | nominal_top_h | `v3_probe_P02_r1024_3d_scatter.png`<br>`v3_probe_P02_r1024_source_slice.png`<br>`v3_probe_P02_r1024_zmid_slice.png` | weak global source background plus hotspot |
| P03 | 0.023438 | 12 | 300.000/307.709/300.337/1.006 | low_top_h | `v3_probe_P03_r1024_3d_scatter.png`<br>`v3_probe_P03_r1024_source_slice.png`<br>`v3_probe_P03_r1024_zmid_slice.png` | low-k barrier confinement |
| P04 | 0.031250 | 9 | 300.000/303.681/300.220/0.423 | high_top_h | `v3_probe_P04_r1024_3d_scatter.png`<br>`v3_probe_P04_r1024_source_slice.png`<br>`v3_probe_P04_r1024_zmid_slice.png` | nested multi-scale k interfaces |
| P05 | 0.041992 | 7 | 300.000/301.638/300.186/0.247 | nominal_top_h | `v3_probe_P05_r1024_3d_scatter.png`<br>`v3_probe_P05_r1024_source_slice.png`<br>`v3_probe_P05_r1024_zmid_slice.png` | random volumetric heat blobs |
| P06 | 1.000000 | 8 | 300.000/301.663/300.211/0.217 | nominal_top_h | `v3_probe_P06_r1024_3d_scatter.png`<br>`v3_probe_P06_r1024_source_slice.png`<br>`v3_probe_P06_r1024_zmid_slice.png` | elongated source plus weak full-domain background |
| P07 | 0.011719 | 10 | 300.000/302.222/300.052/0.164 | high_top_h | `v3_probe_P07_r1024_3d_scatter.png`<br>`v3_probe_P07_r1024_source_slice.png`<br>`v3_probe_P07_r1024_zmid_slice.png` | TSV-like high-k escape path |
| P08 | 0.117188 | 8 | 300.000/303.277/300.120/0.240 | nominal_top_h | `v3_probe_P08_r1024_3d_scatter.png`<br>`v3_probe_P08_r1024_source_slice.png`<br>`v3_probe_P08_r1024_zmid_slice.png` | active hotspot motif in random background |
| P09 | 0.017578 | 14 | 300.000/306.543/300.186/0.602 | nominal_top_h | `v3_probe_P09_r1024_3d_scatter.png`<br>`v3_probe_P09_r1024_source_slice.png`<br>`v3_probe_P09_r1024_zmid_slice.png` | anisotropy max 14.545; k mean = 39.05 / 33.66 / 32.75; extra `v3_probe_P09_r1024_k_channels_source_slice.png` |
| P10 | 0.018555 | 7 | 300.000/302.009/300.076/0.158 | very_high_top_h_candidate | `v3_probe_P10_r1024_3d_scatter.png`<br>`v3_probe_P10_r1024_source_slice.png`<br>`v3_probe_P10_r1024_zmid_slice.png` | localized_top_contact=false; side_asymmetry=false; V1 global top Robin scope confirmed |

q bounding boxes:

| probe | q bbox min -> max m |
| --- | --- |
| P01 | `[0.002667, 0.002667, 0.0006667] -> [0.007333, 0.007333, 0.001333]` |
| P02 | `[0, 0, 0] -> [0.01, 0.01, 0.002]` |
| P03 | `[0.004, 0.004, 0.0006667] -> [0.006, 0.006, 0.001333]` |
| P04 | `[0.002667, 0.002667, 0.0006667] -> [0.007333, 0.007333, 0.001333]` |
| P05 | `[0.001333, 0.002, 0.0006667] -> [0.008, 0.008, 0.001333]` |
| P06 | `[0, 0, 0] -> [0.01, 0.01, 0.002]` |
| P07 | `[0.004, 0.003333, 0.0006667] -> [0.006, 0.005333, 0.0006667]` |
| P08 | `[0.002, 0.003333, 0.0006667] -> [0.008, 0.006667, 0.001333]` |
| P09 | `[0.003333, 0.004, 0.0006667] -> [0.005333, 0.005333, 0.001333]` |
| P10 | `[0.004, 0.004, 0.0006667] -> [0.006, 0.006, 0.001333]` |

## Capability Boundaries

P01-P08 generated physics labels with the existing reference solver v2. P09 also
generated a physics label, but only as a diagonal anisotropic `k_field` with
shape `(N,3)`. Full tensor conductivity `(N,6)` is not generated in v0.

For P09, checker output includes channel-wise `kx/ky/kz` statistics and
anisotropy ratio. The current v0 maximum anisotropy ratio is `14.545`.

P10 generated a physics label using the current V1 boundary model: global top
Robin, bottom Dirichlet, and adiabatic sides. Localized top contact and side
asymmetry are recorded as generator/solver/schema gaps and are not represented
by fabricated labels.

The paired 4096 versions are intentionally not generated in v0. The manifest
contains the paired scene policy and checker support, but 4096 generation should
wait for a separate dense-solver and memory stability gate.

## Input Policy

The generated model-facing physics arrays remain:

- `coords.npy`
- `k_field.npy`
- `q_field.npy`
- BC information in metadata for loader feature construction

`layer_id.npy`, `region_id.npy`, `material_id.npy`, probe family tags, and
semantic scene ids are metadata/bookkeeping only and must not be treated as
default model inputs.
