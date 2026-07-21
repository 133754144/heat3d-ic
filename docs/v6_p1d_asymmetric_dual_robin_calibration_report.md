# V6-P1d asymmetric dual-Robin calibration

## Scope and frozen physical contract

This phase performs only deterministic finite-volume heat solves and dataset
quality control.  It performs no model training or model inference.  The
package is the explicit P1c B path, from PCB, substrate, interposer and
bump/underfill through two active silicon dies, TIMs and the heat spreader.
Both exterior ambients are 300 K, sides are adiabatic, and all material
interfaces retain the frozen perfect-contact assumption.

The boundary family is asymmetric by construction:

- top Robin: 500--2500 W/(m2 K);
- bottom Robin literature-focus: 20--200 W/(m2 K);
- bottom Robin 1 W/(m2 K): one near-adiabatic control family only.

The sources are eight equal-area/equal-power rectangles distributed across
the lower and upper active dies, four per layer.  Registered total emitting
areas are 16, 32, 48 and 64 mm2.  Package power is a frozen discrete input;
no sample uses its solved thermal resistance to infer power.

## Primary-evidence boundary rationale

The traceable evidence is frozen in `docs/v6_p1d_literature_matrix.csv`.
Q3D supports explicit die/interposer/substrate/PCB paths and independently
specified exterior heat exchange, and includes a stacked-package example at
20 W total power.  Alexandrov et al. report a package model
using a heat-sink-equivalent 2500 W/(m2 K) coefficient.  Yovanovich et al.
provide a 500 W/(m2 K) microelectronics heat-sink boundary example.  The
QFN32/64 and PLCC experiments show that package/board exterior transfer is
surface- and flow-dependent, supporting asymmetric rather than identical
top/bottom coefficients.  A PoP study with 12 W/(m2 K) on free surfaces
anchors weak natural-convection scale; consequently 1 W/(m2 K) is explicitly
labelled a deliberately weaker near-adiabatic control, not a main
literature-focus condition.  The inherited B-layer material values retain the
P1c scholarly-book provenance and are not used as evidence for the h range.

## Search and preregistration

The first search is a complete 64-case grid over eight BC families, two power
levels and four emitting areas.  All 64 attempts are retained in the search
JSON/CSV, all naturally fell inside 30--80 K, and none generated a final
sample.  A globally balanced, deterministic assignment then froze the 16
pilot cases before generation: two power slots per BC family and four cases
per emitting-area level.

The first 64-sample expansion retained those two slots across every source
area.  It passed physical QC and all 64 cases fell in the requested window,
but the realized temperature-bin counts were 20/12/5/27.  This result is
retained as calibration trial 1, rather than discarded.

The balanced 64-sample expansion froze four *family-level* discrete powers
before solving any of its cases.  It did not calculate a different power from
each sample's Rth, inspect layouts before selecting them, delete results or
replace seeds.  Its design and realized bin counts were both 16/16/16/16, so
it passed the expansion gate.

| family | top h | bottom h | frozen package-power slots (W) |
| --- | ---: | ---: | --- |
| f00 near-adiabatic control | 500 | 1 | 1.8 / 2.4 / 3.0 / 3.6 |
| f01 | 500 | 20 | 1.8 / 2.5 / 3.1 / 3.7 |
| f02 | 750 | 50 | 2.8 / 3.7 / 4.6 / 5.5 |
| f03 | 1000 | 100 | 3.7 / 5.0 / 6.2 / 7.3 |
| f04 | 1500 | 200 | 5.5 / 7.2 / 8.9 / 10.7 |
| f05 | 2000 | 20 | 6.8 / 9.0 / 11.2 / 13.4 |
| f06 | 2500 | 100 | 8.5 / 11.0 / 13.8 / 16.5 |
| f07 | 2500 | 200 | 8.5 / 11.0 / 13.8 / 16.5 |

## Pilot results

| dataset | samples | peak DeltaT range (K) | peak-bin counts | window hits | max energy error | max branch closure error |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| final pilot16 | 16 | 39.740--71.842 | 5/3/1/7 | 16 | 2.40e-10 | 2.40e-10 |
| retained 64 trial 1 | 64 | 39.373--75.015 | 20/12/5/27 | 64 | 2.40e-10 | 2.40e-10 |
| balanced 64 trial 2 | 64 | 35.655--77.359 | 16/16/16/16 | 64 | 2.63e-10 | 2.63e-10 |
| frozen 1024 | 1024 | 35.522--77.760 | 256/256/256/256 | 1024 | 2.63e-10 | 2.63e-10 |

The balanced 64 design also has eight cases per BC family, 16 cases per area,
and 16 cases per preregistered temperature slot.  Its top heat fraction is
0.9435--0.9980; the explicit package retains a physically nonzero bottom path,
while the high-resistance PCB/substrate path makes top cooling dominant in
this frozen material/geometry family.

## Corrected branch resistance

The old junction-to-surface temperature drop divided by package total power
is an internal path quantity, not an ambient branch resistance.  P1d reports:

```
R_top_branch    = (T_junction - T_inf_top) / Q_top
R_bottom_branch = (T_junction - T_inf_bottom) / Q_bottom
R_effective     = (T_junction - 300 K) / P_total
```

With equal ambients, the independent check is
`R_effective = 1 / (1/R_top_branch + 1/R_bottom_branch)`.  Junction
temperature is source-power-weighted on the solver grid, and the top/bottom
heat rates come from the two exterior Robin faces.  Internal
junction-to-surface and film contributions remain separate diagnostic fields.

## Mesh, source and projection QC

The native base mesh is 64 x 64 x 56 layer-aligned intervals (240,825 nodes).
Two extreme representative cases were also solved on 48 x 48 x 48 and
80 x 80 x 70 meshes.  Base-to-fine peak differences were 0.071% and 0.100%;
the largest checked mean difference was 0.278%.  Every source has at least
seven in-plane mesh intervals and 240 control volumes.

Every sample freezes its 1024 irregular coordinates before the temperature
solve, using fixed volume/source/interface/top/bottom strata.  Every layer and
every interface is represented, all input/label arrays and metadata are
hashed, and coordinates are selected without temperature labels.  The peak
projection gap was at most 0.169 K for pilot16 and 0.310 K for balanced64.

## 1024-sample frozen design and result

The final design is a Cartesian preregistration of eight BC families, four
family-level power slots, four emitting areas and eight layout seeds.  Thus it
contains 128 cases per BC family, 256 per emitting area and 256 per intended
temperature slot.  The 1 W/(m2 K) bottom control accounts for 12.5%; the other
87.5% use the literature-focus 20--200 W/(m2 K) bottom range.

All 1024 cases were generated and all naturally entered 30--80 K.  The actual
four-bin counts are exactly 256/256/256/256, the median peak rise is 55.523 K,
and the power range is 1.8--16.5 W.  The maximum energy-balance and parallel
branch-closure errors are 2.63e-10.  The minimum source resolution is seven
in-plane intervals and 240 control volumes; every sample covers all nine
layers and all eight interfaces in its frozen 1024 points.  The maximum
solver-to-projected peak gap is 0.346 K.

The dataset manifest SHA256 is
`70e00496b929f4bd1d5af7846cf75953678e2dc4f95b2f0c582cea15a3a8ac17`.
No sample was filtered, resampled or replaced after solve.

## Interpretation and limits

The calibration establishes a controlled asymmetric dual-Robin *effective
boundary family* for the explicit B package, not universal convective
coefficients.  The package geometry/material values, perfect contact and
uniform exterior Robin approximation remain modeling assumptions.  The
30--80 K coverage is evidence for the frozen discrete family only; it is not a
license to tune individual future samples by their labels or thermal
resistance.  Dataset splitting and any downstream model work are outside
P1d.
