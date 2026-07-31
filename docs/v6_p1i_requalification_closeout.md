# V6-P1i data requalification closeout

Status: **complete; formal1024_v1 qualified**.

This work performed dataset construction and physics qualification only. It
ran no model training or inference, did not modify V6/P1h, and did not patch or
replace any formal1024_v0 sample.

## Formal1024_v0 zero-solve postmortem

Formal1024_v0 remains a permanent qualification failure. Its 5.743380 K gap is
between `v6p1if_0923` (165.906 K) and `v6p1if_0650` (171.650 K), near the
0.998 empirical quantile. The midpoint KDE density is about 12.1% of the
global maximum, so the gap is an extreme upper-tail coverage defect, not a
core-region hole or modal valley.

The old generator coupled source size and severity strongly
(`rho=0.9454`). Peak temperature was also dominated by severity
(`rho=0.9865`, single-feature `R²=0.7329`) and mean source area
(`rho=0.9306`, `R²=0.8552`). Power remained physically relevant
(`rho=0.8025`). Effective resistance retained a strong inverse top-h response
(`rho=-0.9786`), and top-h versus peak remained strongly negative after
controlling log-power (about `-0.918`). The complete evidence is in the
postmortem JSON/CSV/figure.

## Target-independent split reconstruction

Four schemes were compared at both 128 and 1024 input definitions before any
solve:

1. hash within Sobol octets;
2. one global hash;
3. an independent Sobol split dimension;
4. balanced pre-solve assignment over the complete input population.

Continuous variables used pairwise KS, discrete variables used frequency/TV,
and a joint standardized mean/covariance discrepancy was added. The balanced
pre-solve scheme was selected and preregistered without temperatures, solver
outputs, or model errors. On formal1024_v1 it achieved maximum KS `0.134115`,
TV `0.117188`, and joint discrepancy `0.036497`, all below the frozen
`0.16/0.15/0.05` gates.

## Global-rule reconstruction

All failed pilots and engineering aborts remain in the generation-attempt
registry. No sample was filtered, replaced, or power-backsolved from thermal
resistance.

- v3--v12 exposed outer-safety, histogram-occupancy, or gap failures.
- v13 was stopped after 24 solves because its sample prefix collided with v12;
  the partial targets were excluded from scientific-rule revision.
- v14 passed all 128 gates, but the first 1024-input preflight found an
  unsolved 1.6089 W design below the 1.7 W literature floor.
- Per contract, the formal build was stopped before freeze and the workflow
  returned to a new pilot.
- v15 added an analytical whole-domain power constraint. Its frozen rule spans
  1.7282--19.4888 W for every allowed severity/top-h/jitter combination.
  Pilot128_v15 passed all gates under the new seed 612818.

The literature contract records the primary-source basis and separately marks
bulk conductivity, effective conductivity, geometry/stack thickness,
convective coefficients, component/package power, and derived volumetric q.
Derived q is not represented as a directly measured literature quantity.

## Formal1024_v1 qualification

Formal1024_v1 used the accepted v15 rule unchanged and a new seed 612819. Its
1024 input definitions, split, ranges, geometry/support preflight, generator,
config, acceptance contract, and code were hashed before the first formal
solve.

| Gate | Result | Frozen limit |
|---|---:|---:|
| peak DeltaT range | 30.374--173.098 K | 20--180 K safety |
| inside 30--150 K | 993/1024 = 96.973% | >=95% |
| 12-bin max/min | 2.509 | <=3 |
| core q05--q95 gap | 1.139 K | <=5 K |
| full sorted gap | 4.077 K | <=5 K |
| KDE modes | 2 | <=3, not 4 |
| power--peak Spearman | 0.7769 | >=0.45 |
| top-h--Reff Spearman | -0.9790 | <=-0.30 |
| top-h partial Spearman | -0.8962 | <=-0.30 |
| severity--peak Spearman | 0.7008 | abs <=0.90 |
| source-area--peak Spearman | -0.0232 | abs <=0.80 |
| max single-latent R² | 0.6296 | <=0.80 |
| minimum local-region support | 4 | >=4 |
| maximum energy-balance error | 1.314e-10 | <=1e-6 |
| maximum linear residual | 1.225e-10 | <=1e-7 |

The 12 temperature-bin counts are
`[55, 85, 86, 62, 78, 138, 103, 96, 85, 81, 69, 55]`. All 1024 samples,
including the 31 above 150 K, are retained. Formal1024_v1 is qualified as a
dataset artifact; this closeout does **not** authorize training.

## Artifacts and boundaries

- Machine-readable lifecycle:
  `configs/heat3d_v6_p1i/v6_p1i_requalification_manifest.json`
- Formal artifact/hash manifest:
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_artifact_manifest.json`
- Formal audit:
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_distribution_audit.json`
- Formal dataset:
  `data/heat3d_v6_p1i_continuous_physics1024_v1`
- Deterministic checker:
  `scripts/check_heat3d_v6_p1i_requalification.py`

Formal1024_v0 remains unchanged and forbidden for training. No claim is made
here about downstream model accuracy, test performance, or suitability beyond
the frozen P1i physics/distribution qualification contract.
