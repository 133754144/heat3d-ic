# V4 P3c Random-Block Dataset Design

Read this file only for V4 P3c dataset generation, parameter-space, split, or
random-block design decisions.

## Scope

This is a design and registry-planning document only. It does not generate data,
write a generator, run a solver, start training, or write artifacts.

Docs define design intent only. P3c risks must be enforced by the machine
registry, the P3c dry-run generator, and the checker. P3c-2 validates only the
dry-run scene/array contract in memory; it must not write a dataset or call the
solver.

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
IC-like motifs such as TSV-adjacent paths, spreaders, TIM-like regions, HBM-like
anisotropy, and active-die hotspots may appear, but they are sampled within one
unified random generation distribution.

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
| HBM thermal measurement, Chalise and Cahill, 2023, arXiv:2303.06785 | include anisotropic HBM-like material anchors, especially in-plane versus through-plane k contrast |
| DeepOHeat, Liu et al., 2023, arXiv:2302.12949 | learn an operator from power/material/boundary functions to temperature fields |
| DeepOHeat-v1, Yu et al., 2025, IEEE TCPMT/arXiv:2504.03955 | keep solver quality and confidence/audit metadata separate from model training changes |
| Non-uniform BSPDN power maps, Chen et al., 2025, arXiv:2508.02284 | include fine-grained localized and synthetic power maps; use 200 and 2500 W/m2/K HTC anchors |
| SAU-FNO 3D-IC thermal simulation, Huang et al., 2025, arXiv:2510.15968 | include high-frequency/local hotspots and train/test random split from the same distribution |
| Heat3D P1 medium1024 audit, local doc | use current k/q/top_h scales as solver-scale references, not as final_probe hard ranges |

## Generation Distribution

P3c uses one unified random generation distribution:

- no stress split;
- splits are only `train` and `test`;
- split assignment uses a fixed random seed after generation;
- post-generation audit must compare train/test distributions before the data is
  accepted;
- final_probe is run only as a reference diagnostic after the main distribution
  is stable, and it is not part of pass/fail gating.

Recommended post-generation audit fields:

- k min/max/percentiles by conductivity class and diag3 component;
- q min/max/percentiles, total power, active volume fraction, hotspot count;
- BC flag distribution and cooling-regime coverage;
- geometry extent, aspect ratio, and block-size distribution;
- raw T range, raw DeltaT range, and normalized DeltaT range;
- solver residual, energy balance residual, and bottom Dirichlet error.

## Expected DeltaT Distribution

DeltaT binning is generator quality control, not a model-performance metric.
P3c-3 and P3c-4 may recalibrate the bins after smoke/pilot solver evidence.

Every generated sample audit should record:

- `deltaT_peak_K`;
- `deltaT_p95_K`;
- `deltaT_bin`;
- `q_rescale_factor`;
- `reject_reason`.

Initial bins:

| bin | rule | generator action |
| --- | --- | --- |
| `reject_low` | `deltaT_peak_K < 0.02` | reject or rescale q upward |
| `low` | `0.02 <= deltaT_peak_K < 0.2` | keep only if distribution coverage needs low-amplitude cases |
| `nominal` | `0.2 <= deltaT_peak_K < 2.0` | preferred production mass |
| `hard` | `2.0 <= deltaT_peak_K <= 8.0` | keep as hard in-distribution cases if solver audit passes |
| `review_high` | `8.0 < deltaT_peak_K <= 15.0` | manual review until P3c-3/4 calibration |
| `reject_high` | `deltaT_peak_K > 15.0` | reject or rescale q downward |

## Random k(x) Rules

- Sample k in log space by material class, not by a single global uniform range.
- k entries must carry `literature_anchor`, `sampling_envelope`, and
  `rationale` in the registry.
- Every sample has a background effective-stack class plus 1 or more random
  rectangular material blocks.
- Low-k barriers and high-k bridges must both be represented.
- Silicon-like, HBM-like anisotropic, and spreader/TSV-like classes should
  appear as semantic classes, not arbitrary unlabeled scalar outliers.
- Production v0 targets `diag3_target_fraction=0.20`; the generator and checker
  must support diag3 fields before production generation.
- Store generator metadata for k class, block count, block extents, and diag3
  status. Do not pass class IDs as default model inputs.

## Random q(x) Rules

- Sample q density in log space within the registry envelope.
- Include compact hotspots, multi-block hotspots, elongated strips, weak
  background plus hotspot, dual-z sources, and TSV-adjacent sources.
- q registry entries must record source volume fraction, integrated power
  target, and DeltaT target bin.
- Preserve explicit metadata for q family, active volume fraction, source count,
  source z locations, q density range, integrated power target, and total power.
- Avoid pure uniform-power datasets; uniform or weak-background fields are
  allowed only as part of the mixture.
- P3c smoke must check DeltaT amplitude before scaling to larger datasets.

## P3c-2b Array Preflight Contract

P3c-2b materializes real arrays in memory only: `coords`, `k_field`, `q_field`,
`bc_features`, and `sample_meta`. It must not write `data/` or `output/`, must
not call the solver, and must keep DeltaT QC as `pending_until_solve`.

Executable policies:

- `background_k_policy`: initialize all nodes with `effective_stack_medium_k`
  by default. Allowed backgrounds are `effective_stack_medium_k`,
  `silicon_like`, and `hbm_like_anisotropic_k`; `low_k_dielectric_underfill`
  may appear only as minority background or block material and is not the
  default background.
- `k_overlap_policy`: `deterministic_priority_override`; initialize full-domain
  background k first, then apply blocks in deterministic order. Each node keeps
  only the final k value, while metadata records `covered_by_blocks` and
  `winning_block_id`. Arithmetic-mean merge is not the generator default.
- `q_overlap_policy`: `sum_volumetric_sources`; overlapping q blocks sum per
  cell. Max pooling is not used for generator q merge.
- `power_calibration_policy`: after block projection, use realized volume and
  integrated-power target to compute calibrated q density and record target
  power, realized volume, calibrated q density, realized power, and power error.

Background k reference values are common semiconductor substrate/material
anchors, not final_probe-derived hard ranges. P3c uses:

- `effective_stack_medium_k`: default background, suggested reference values
  around 10/30/60 W/m/K for equivalent substrate/interposer composites;
- `silicon_like`: allowed background, reference values around 100 to 150 W/m/K
  for silicon-like die/substrate material;
- `hbm_like_anisotropic_k`: allowed diag3 background, HBM-like anchors near
  in-plane 100/140 W/m/K and through-plane 7/2 W/m/K;
- `low_k_dielectric_underfill`: non-default background/block-only low-k anchor
  around 0.5 to 8 W/m/K.

Shape rules:

- scalar samples: `k_field` shape is `[N, 1]`;
- diag3 samples: `k_field` shape is `[N, 3]`;
- `q_field` shape is `[N, 1]`;
- `bc_features` shape is `[N, 4]` for top, bottom, side, interior flags.

## Boundary And Contact Rules

V4 production/default contact model is fixed:

```text
R_contact=0_perfect_contact
```

Finite interface thermal resistance is implemented/deferred for solver smoke
only. It is not part of the V4 P3c dataset, default solver path, or production
label path.

Boundary conditions are sampled only within the registry. The default P3c
production path varies top Robin h through cooling regimes, while keeping side
walls adiabatic and bottom Dirichlet semantics explicit.

## P3c Route

- P3c-0 design spec.
- P3c-1 parameter registry.
- P3c-2 generator skeleton.
- P3c-3 16-sample smoke + audit.
- P3c-4 64-sample pilot + split audit.
- P3c-5 P3b-lite validation subset selection.
- P3c-7 1024 production candidate.
- P3c-8 closeout + handoff to training.
