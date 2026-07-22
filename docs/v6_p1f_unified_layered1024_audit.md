# V6-P1f final qualification

- dataset: `heat3d_v6_p1f_unified_layered1024_v0`;
- geometry groups / cases: `128` / `1024`;
- peak DeltaT min/median/max: `30.055` / `46.353` / `77.816 K`;
- below 30 K: `0`;
- 30--80 K: `1024/1024` (`100.00%`);
- above 100 K: `0/1024` (`0.00%`);
- above 120 K: `0`;
- max BC--power |Pearson| / |Spearman|: `0` / `0`;
- maximum energy-balance relative error: `1.203e-10`;
- qualification: `PASS`.


## Split audit

| split | groups | cases | source area min/median/max mm2 | aggregate density min/median/max W/cm2 | peak DeltaT min/median/max K |
|---|---:|---:|---:|---:|---:|
| train | 96 | 768 | 15.56/24.59/39.86 | 10.03/19.78/38.56 | 30.05/46.44/77.43 |
| valid | 16 | 128 | 15.85/23.01/37.45 | 10.68/20.08/37.85 | 30.56/46.42/77.82 |
| test | 16 | 128 | 15.79/24.83/34.79 | 11.50/19.13/38.00 | 30.08/46.10/72.52 |

Each split has source-count proportions `0.125` for every count from 3 through 10, layout proportions `0.25` for each of four layout families, the same fixed material profile, and the exact same 2x2x2 BC/power factorial.  Pairwise KS p-values for total source area and upper-layer power fraction are all above `0.67`.

## Physical integrity

- minimum source resolution: `270` control volumes and `7` in-plane intervals;
- maximum q: `5.688889e+09 W/m3`;
- maximum single-source power: `3.000 W`;
- maximum surface power density: `60.749 W/cm2`;
- every sample covers all layers and interfaces with one group-frozen 1024-point set.

The gate is applied to the complete version.  No case was filtered, replaced,
or retained conditionally.  Pilot geometry and sample IDs are forbidden from
the final dataset; only a globally frozen contract may advance.
