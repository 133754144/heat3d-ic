# V6-P1g geometry-factor deconfounding audit: heat3d_v6_p1g_geometry_deconfounded1024_v1

## Decision

This is the immutable P1g-v1 whole-version revision of P1g-v0. It reuses every v0 factor-assignment row, source size, area, power fraction, seed, and all P1f scientific contracts. Before solving, all groups receive one frozen alignment definition: paired centroids coincide for partly-aligned groups and are displaced by exactly two solver-mesh intervals for offset groups. No case, seed, or factor level was selected from labels. No sample was filtered, replaced, Rth-back-calculated, or locally patched.

The qualification gate **passed**. Peak ΔT spans 30.039–76.385 K; 1024/1024 cases are in 30–80 K.

## Factor deconfounding

All three factor margins are identical in train/valid/test. Globally, each of the 8×4×2 source-count/layout/alignment combinations appears exactly twice; all global pairwise Cramér's V and mutual information are zero (up to floating-point roundoff). Train is also pairwise independent. With only 16 groups, a valid/test 8×4 count-layout table cannot be independent (expected 0.5 case/cell); the frozen complementary schedules attain the preregistered minimum-support construction while keeping count-alignment and layout-alignment independent. Every count sees multiple layouts and both alignments in every split.

P1g-v1 paired-centroid diagnostics confirm 0 mm displacement for partly-aligned sources and exactly 0.3125 mm for offset sources. The P1f-v0 files and hashes remain untouched.

## Representation and leakage QC

Every sample retains 1024 points, covers every layer and interface, and reuses coordinates only within its geometry group. Point selection is frozen before solving and uses no label. IDW-8 reconstruction is a post-solve representation diagnostic—not model inference and not a selection criterion. Full-field CV-RMSE median is 0.834094 K (P95 1.708763 K). Maximum absolute per-layer mean error is summarized in the companion JSON/CSV, as are layer-drop errors and per-layer point counts.

Input-only geometry signatures have 0 exact duplicates and 0 pairs below the frozen standardized-distance threshold 1e-06.

## Artifacts

- Frozen config: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1.yaml`
- Joint tables: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1_joint_contingency.csv`
- Projection summary: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1_projection_diagnostics.csv`
- Layer diagnostics: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1_layer_projection_errors.csv`
- Machine-readable audit: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1_geometry_audit.json`
- Qualification decision: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_v1_qualification.json`
