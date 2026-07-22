# V6-P1g geometry-factor deconfounding audit

## Decision

P1g is a new version derived from immutable P1f-v0. It preserves the P1f stack, materials, 2×2×2 BC/power block, whole-version temperature gate, frozen 1024-point sampling contract, and 96/16/16 group split. Only the geometry assignment and newly seeded geometry instances were rebuilt. No sample was filtered, replaced, Rth-back-calculated, or locally patched.

The qualification gate **failed**. Peak ΔT spans 29.378–76.166 K; 1006/1024 cases are in 30–80 K.

## Factor deconfounding

All three factor margins are identical in train/valid/test. Globally, each of the 8×4×2 source-count/layout/alignment combinations appears exactly twice; all global pairwise Cramér's V and mutual information are zero (up to floating-point roundoff). Train is also pairwise independent. With only 16 groups, a valid/test 8×4 count-layout table cannot be independent (expected 0.5 case/cell); the frozen complementary schedules attain the preregistered minimum-support construction while keeping count-alignment and layout-alignment independent. Every count sees multiple layouts and both alignments in every split.

P1g also makes the alignment label geometrically truthful: `offset` groups have no shared upper/lower source slots, while `partly_aligned` groups have positive overlap. This corrects the new P1g geometry instances only; the P1f-v0 files and hashes remain untouched.

## Representation and leakage QC

Every sample retains 1024 points, covers every layer and interface, and reuses coordinates only within its geometry group. Point selection is frozen before solving and uses no label. IDW-8 reconstruction is a post-solve representation diagnostic—not model inference and not a selection criterion. Full-field CV-RMSE median is 0.826625 K (P95 1.702908 K). Maximum absolute per-layer mean error is summarized in the companion JSON/CSV, as are layer-drop errors and per-layer point counts.

Input-only geometry signatures have 0 exact duplicates and 0 pairs below the frozen standardized-distance threshold 1e-06.

## Artifacts

- Frozen config: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml`
- Joint tables: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_joint_contingency.csv`
- Projection summary: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_projection_diagnostics.csv`
- Layer diagnostics: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_layer_projection_errors.csv`
- Machine-readable audit: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_geometry_audit.json`
- Qualification decision: `configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_qualification.json`
