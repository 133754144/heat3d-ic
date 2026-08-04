# V6 P1i controlled cross-resolution protocol audit

## Scope and frozen inputs

This audit is valid-only and inference-only. It uses the frozen P1i seed0
point-global checkpoint at epoch 559 (`51567afe...b90e`), the SHA-ranked fixed
32-sample `valid_iid` subset, and train-only normalization metadata. Test and
sealed IID roles remain closed. The model, checkpoint, graph scientific
parameters, and objective are not changed or selected during this audit.

The machine-readable preregistration is
`configs/heat3d_v6_p1i/v6_p1i_controlled_cross_resolution_protocol.json`.

## Historical protocol reconstruction

The trace table is
`configs/heat3d_v6_p1i/v6_p1i_cross_resolution_historical_protocol_audit.csv`.
The historical experiments establish the following boundaries:

- The volume-only ladder changed support distribution, Global Context, scale
  inputs, and regional-node count together. Its roughly 140% errors do not
  isolate resolution.
- The anchored-volume diagnostic attributed most of that gap to context drift,
  but still allowed the regional mesh to grow with physical resolution.
- The source-aware P1h ladder was much more stable. Its remaining increase from
  0.90% to 4.49% (or 0.90% to 1.84% with anchor-derived context/scale) still
  confounded query resolution and `Nr`.
- Production Anchor-derived inference is a valid deployment path, but it is not
  an upstream-style direct-`N` invariance experiment.
- The existing P1i structured direct-`N` results are retained only as a
  measure-conservative full-graph re-discretization diagnostic. It is neither
  checkpoint-IID nor a formal same-distribution invariance test.

## Upstream protocol and Heat3D correction

The official source is frozen at RIGNO commit
`3e4b307c90f34237d0c1e5e497d4301116e9c3db`. In `rigno/test.py`, a spatial
resolution test synchronously permutes/truncates `u`, `c`, and `x`, uses
`x_in=x_out`, and rebuilds graph metadata with
`rmesh_correction_dsf=train_space_dsf/inference_space_dsf`. In
`rigno/models/rigno.py`, higher-resolution inputs receive additional regional
subsampling, while lower-resolution inputs receive Delaunay-simplex-centroid
refinement. The paper's resolution figure and the current source's `space_dsfs`
list are version-distinct evidence and are not merged into one claimed grid.

Heat3D is 3-D. Its inherited `_upsample_pointset` raises the refinement factor
to the coordinate dimension. Passing `0.5` literally at `N=512` would turn the
initial `Nr=128` into about 1024, not the training-scale target of about 256.
This audit therefore solves the correction from the explicit target:

- `N>=1024`: correction is `(N/4)/256 = N/1024`;
- `N=512`: correction is `(128/256)^(1/3)=0.793700526`, so simplex-centroid
  refinement returns about 256 regional nodes.

Every worker checks the realized `Nr`; this protocol is named
`upstream_regional_scale_corrected_heat3d_3d` rather than `upstream_like`.

The audited upstream files are content-bound as follows:

- `rigno/test.py`: `b94fc47392efad6ebac2efbcd5207f9338d5f55d444dca2062cfd7cbb5937944`;
- `rigno/models/rigno.py`: `d54175c4c5803d5552031c3e5d8cf1782fcd666b3f7b78f1a17f317f682d9800`.

## Controlled design

For each discretization seed 0/1/2/3 and each valid sample, one label-independent
master ordering is built from solver coordinates, control volume, layer IDs,
and registered q/k block metadata. Prefix quotas preserve the frozen P1i
support family: 25% block, 12.5% interface, 6.25% top, 6.25% bottom, and 50%
volume. The 512/1024/2048/4096/8192/16384 sets are strictly nested within the
same sample and discretization seed. Different seeds are independent
discretizations and are not claimed to be nested with one another.

All literal q/k block nodes are ordered before any extension. If a high-N 25%
block quota exceeds the finite literal block capacity, the remainder is drawn
only from a target-independent halo in the same active layers. Core and halo
counts are reported separately; this avoids silently changing the quota or
dropping low-capacity samples.

Full solver nodes are assigned to the nearest selected node within the same
layer. Aggregated control volume, source power, and CV-weighted conductivity
moments are conserved to floating-point tolerance. Temperature is excluded
from support and graph construction. The selected support is used for both
input and output coordinates.

The four factor cells at 1024/4096/16384/65536 are:

| Cell | Support | Regional mesh |
|---|---|---|
| A | source-aware | fixed near training `Nr=256` |
| B | source-aware | grows as approximately `N/4` |
| C | legal structured | fixed near training `Nr=256` |
| D | existing legal structured | grows as approximately `N/4` |

Direct-`N` results remain a measure-conservative full-graph re-discretization
diagnostic. A versus B isolates
regional-scale drift under the same support; A versus C diagnoses support
distribution; C versus D isolates regional scale on structured support.

## Metrics and attribution

Each primary cell reports support point-global true-RMS relative RMSE,
sample-first CV-relative RMSE, raw CV RMSE K, peak/source/interface errors, and
the same model-plus-reconstruction metrics on the common 240825-node field.
Oracle reconstruction is reported separately as the sampling floor.

Graph diagnostics exclude dummy nodes/edges and record actual `Nr`, p2r/r2r/r2p
edge counts, in/out-degree distributions, edge-length distributions, isolated
regional nodes, and weakly connected r2r components. Feature diagnostics retain
the 24-D standardized Global Context, regional QK summaries, physical log-scale,
and predicted log-scale, with drift measured against `N=1024` of the same seed.

The final attribution must distinguish resolution, support distribution,
regional-mesh scale, and feature drift. It must not promote or tune a model from
these direct-`N` cells.
