# V6-P1f pilot qualification

- dataset: `heat3d_v6_p1f_temperature_shaping_pilot128_v0`;
- geometry groups / cases: `16` / `128`;
- peak DeltaT min/median/max: `30.098` / `46.044` / `75.920 K`;
- below 30 K: `0`;
- 30--80 K: `128/128` (`100.00%`);
- above 100 K: `0/128` (`0.00%`);
- above 120 K: `0`;
- max BC--power |Pearson| / |Spearman|: `0` / `0`;
- maximum energy-balance relative error: `1.203e-10`;
- qualification: `PASS`.
- total source area min/median/max: `16.64/24.52/40.69 mm2`;
- aggregate power density min/median/max: `9.83/20.72/36.06 W/cm2`;


## Physical integrity

- minimum source resolution: `297` control volumes and `8` in-plane intervals;
- maximum q: `4.667806e+09 W/m3`;
- maximum single-source power: `3.000 W`;
- maximum surface power density: `53.010 W/cm2`;
- every sample covers all layers and interfaces with one group-frozen 1024-point set.

The gate is applied to the complete version.  No case was filtered, replaced,
or retained conditionally.  Pilot geometry and sample IDs are forbidden from
the final dataset; only a globally frozen contract may advance.
