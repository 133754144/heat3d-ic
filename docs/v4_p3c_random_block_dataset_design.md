# V4 P3c Random-Block Dataset Design

Read this file only for V4 P3c dataset generation, parameter-space, split, or
random-block design decisions.

## Scope

This is a design and registry-planning document only. It does not generate data,
run a solver, start training, or write artifacts.

## Target Task

```text
coords + k(x) + q(x) + BC -> T(x)
```

Boundary conditions are model inputs. Layer-stack, interface, material, and
region labels remain generator metadata and evaluation-group metadata, not
default model-input features.

## Dataset Goal

V4 P3c should build a publication-oriented random-block dataset for steady 3D IC
thermal surrogate modeling. The dataset should stress physical field variation
through conductivity fields, volumetric power fields, and boundary conditions,
while keeping the learning task aligned with the V4 standard input/output
contract.

The random-block design should not be a stack-template interpolation exercise.
IC-like motifs such as TSV-adjacent paths, spreaders, TIM-like regions, and
active-die hotspots may appear, but they are sampled within a unified random
generation distribution.

## Existing Final-Probe Assessment

The V3 final-probe set is useful as a reference diagnostic because it contains
random material-block composites, sparse high-k bridges, low-k barriers,
multi-scale interfaces, multi-blob and elongated heat sources, TSV-like paths,
localized anisotropy, and deferred contact/asymmetric-BC motifs.

It must not define hard P3c ranges:

- it is a 10-sample OOD probe, not a train distribution;
- earlier audits found final-probe k, q, top_h, and DeltaT amplitude outside
  the medium1024 train envelope;
- some final-probe samples were generator or metadata compatibility tests;
- final_probe remains a reference diagnostic only, not a pass/fail split.

P3c should absorb final-probe motifs into the main randomized family instead of
creating a separate stress split or copying final-probe extremes.

## Literature And Source Implications

| source | design implication |
| --- | --- |
| 3D-ICE 4.0, Zhu et al., 2025, arXiv:2512.05823 | preserve material heterogeneity and anisotropy; keep vertical conduction and grid metadata auditable |
| DeepOHeat, Liu et al., 2023, arXiv:2302.12949 | learn an operator from power/material/boundary functions to temperature fields |
| DeepOHeat-v1, Yu et al., 2025, IEEE TCPMT/arXiv:2504.03955 | keep solver quality and confidence/audit metadata separate from model training changes |
| Non-uniform BSPDN power maps, Chen et al., 2025, arXiv:2508.02284 | include fine-grained localized and synthetic power maps; uniform power alone hides hotspot risk |
| SAU-FNO 3D-IC thermal simulation, Huang et al., 2025, arXiv:2510.15968 | include high-frequency/local hotspots and train/test random split from the same distribution |
| Multiscale thermal modeling review, Barua et al., 2026, arXiv:2604.03290 | track TIM/TBR/anisotropy risks, but defer finite contact until validation is publication-grade |
| Heat3D P1 medium1024 audit, local doc | use current k/q/top_h scales as solver-scale references, not as a final dataset proof |

## Generation Distribution

P3c uses one unified random generation distribution:

- no stress split;
- splits are only `train` and `test`;
- split assignment uses a fixed random seed after generation;
- post-generation audit must compare train/test distributions before the data is
  accepted;
- final_probe is run only as a reference diagnostic after the main distribution
  is stable.

Recommended post-generation audit fields:

- k min/max/percentiles by conductivity class and diag3 component;
- q min/max/percentiles, total power, active volume fraction, hotspot count;
- BC flag distribution and top Robin h range;
- geometry extent, aspect ratio, and block-size distribution;
- raw T range, raw DeltaT range, and normalized DeltaT range;
- solver residual, energy balance residual, and bottom Dirichlet error.

## Random k(x) Rules

- Sample k in log space by material class, not by a single global uniform range.
- Every sample has a background effective-stack class plus 1 or more random
  rectangular material blocks.
- Low-k barriers and high-k bridges must both be represented.
- Silicon-like and spreader/TSV-like classes should appear as semantic classes,
  not arbitrary unlabeled scalar outliers.
- `diag3` anisotropy is allowed as metadata-tagged conductivity, but the first
  production dataset may keep scalar/isotropic k if checker support is not ready.
- Store generator metadata for k class, block count, block extents, and diag3
  status. Do not pass class IDs as default model inputs.

## Random q(x) Rules

- Sample q density in log space within the registry envelope.
- Include compact hotspots, multi-block hotspots, elongated strips, weak
  background plus hotspot, dual-z sources, and TSV-adjacent sources.
- Preserve explicit metadata for q family, active volume fraction, source count,
  source z locations, q density range, and total power.
- Avoid pure uniform-power datasets; uniform or weak-background fields are
  allowed only as part of the mixture.
- P3c smoke must check DeltaT amplitude before scaling to larger datasets.

## Boundary And Contact Rules

V4 production/default contact model is fixed:

```text
R_contact=0_perfect_contact
```

Finite interface thermal resistance is implemented/deferred for solver smoke
only. It is not part of the V4 P3c dataset, default solver path, or production
label path.

Boundary conditions are sampled only within the registry. The default P3c
production path varies top Robin h and keeps side walls adiabatic and bottom
Dirichlet semantics explicit.

## P3c Route

0. Source survey and design constraints: no data generation.
1. Parameter registry and machine-readable JSON: no data generation.
2. Generator schema and dry checker: validate metadata and no-write behavior.
3. 16-sample smoke dataset: check solver convergence, energy balance, Tmax
   sanity, metadata completeness, and runtime.
4. Smoke closeout: repair only generator/metadata issues that block P3d.
5. 64-sample pilot handoff: P3d audits failure rate, coverage, split sanity, and
   runtime scaling.
6. 1024 full-dataset handoff: P3e generates the full set only after P3d passes.
7. Model retraining comparison: P3e retrains the V4P1_12 route and compares old
   and new data on IID/test/final-like/Tmax-underprediction diagnostics.
