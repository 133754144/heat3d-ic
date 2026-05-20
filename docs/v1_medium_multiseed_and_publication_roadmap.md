# Heat3D v1 Medium Multi-Seed and Publication Roadmap

## Scope

This document summarizes the current Heat3D v1 medium256 research-stage
diagnostics and proposes the next steps toward a more publishable experimental
protocol. It is diagnostic / research-stage guidance only. It is not a formal
benchmark report, not an OOD generalization claim, and not a publication-ready
model-performance conclusion.

## 1. Current Stage Summary

The current V1 medium256 stage has reached a controlled training and analysis
loop:

- physics-label medium256 dataset tooling exists;
- generation, checking, label diagnostics, training export, comparison,
  run-analysis, and error-binning tools are available;
- the JAX training runner can run on the SSH server environment when the
  server JAX/JAXLIB installation exposes the desired accelerator backend;
- `background_l1_relative` is the current useful supervised loss mode;
- the important current training setting is:

```text
loss_mode = background_l1_relative
lr = 1e-2
epochs = 300
background_relative_weight = 0.05
hotspot_weight = 0.02
loss_weight_schedule = constant
seeds = 0, 1, 2
```

Current SSH observations:

- seed 0 gives a very strong diagnostic result;
- seed 1 is stable but visibly weaker;
- seed 2 is better than seed 1 but still below seed 0;
- the `lr=0.01` setting is effective, but seed sensitivity is now a central
  concern;
- low-DeltaT `bin_0` background bias still appears in diagnostics;
- OOD BC and OOD stack candidate splits remain unstable in mean-field metrics;
- the current results should be described as research diagnostics, not formal
  benchmark or publication-ready performance.

The next immediate need is systematic multi-seed aggregation, not another
blind loss or learning-rate adjustment.

## 2. Meaning For The 3D IC Thermal Simulation Goal

The long-term goal is to build a fast steady-state thermal-field surrogate for
3D IC settings with:

- arbitrary or highly variable thermal-conductivity distributions;
- arbitrary or highly variable heat-source distributions;
- explicit boundary-condition encoding;
- multilayer heterogeneous stack structures;
- recovered 3D temperature fields suitable for downstream diagnostics.

The current V1 stage is meaningful because it shows that a RIGNO-like /
neural-operator-style runner can learn nontrivial multilayer 3D temperature
fields under explicit BC and heterogeneous conductivity inputs. It also shows
that loss design and learning rate strongly affect low-temperature background
calibration and high-temperature hotspot behavior.

The three central issues identified so far are:

1. background-hotspot tradeoff: improving low-DeltaT background calibration can
   weaken high-DeltaT peak / hotspot diagnostics;
2. seed sensitivity: seed 0 is much stronger than seeds 1 and 2 under the same
   nominal setting;
3. OOD stack mean-field generalization: held-out stack-style candidate behavior
   remains weaker and less stable.

This means V1 is a useful controlled diagnostic stage, but it cannot yet be
described as handling arbitrary 3D IC structures. The correct wording is
medium-scale controlled diagnostic / benchmark-candidate preparation.

## 3. Is There Overfitting In The Small Dataset?

The current evidence should be interpreted cautiously. It is not enough to say
simply "yes" or "no".

Training and validation behavior:

- if train and valid metrics improve together, the result is not classic
  immediate train-only overfitting;
- however, train/valid synchronization does not prove broad generalization
  because valid samples are still drawn from a limited medium256 design space.

Train-vs-valid gap:

- a large train-valid gap would point toward classic overfitting;
- a smaller gap with large seed variance can instead indicate optimization
  sensitivity, unstable basins, or sensitivity to initialization under the
  current loss and learning rate.

Seed sensitivity:

- seed 0 being much stronger than seeds 1 and 2 is a warning sign for reporting;
- it may reflect optimization sensitivity rather than only dataset overfitting;
- publication-style reporting must include mean/std and best/median/worst, not
  only the best seed.

Split behavior:

- `test_id`, `test_ood_bc_candidate`, and `test_ood_stack_candidate` should be
  read as candidate diagnostics only;
- unstable OOD stack / OOD BC mean-field behavior suggests the model has not
  learned robust arbitrary-structure behavior.

Dataset scale and diversity:

- medium256 is still small for neural-operator-style claims;
- stack templates, source patterns, k-field distributions, and BC variations
  are deliberately limited;
- there is real risk of over-adaptation to current stack templates or source
  distributions;
- without an independent larger held-out distribution, overfitting cannot be
  ruled out.

To decide this more rigorously, the project needs medium512/medium1024-style
datasets, more structure templates, stricter held-out splits, and multi-seed
statistics.

## 4. Should The Dataset Be Expanded?

The recommendation is staged expansion, not "larger is always better".

### Stage 1: Finish medium256 stability analysis

Before expanding, complete the multi-seed summary for seeds 0/1/2:

- quantify mean/std for overall metrics;
- report best, median, and worst seed;
- inspect split-wise and condition-wise stability;
- inspect error bins, especially `bin_0`, `bin_1`, `bin_3`, and `bin_4`.

This keeps medium256 as a fast ablation and debug set.

### Stage 2: Design medium512 or medium1024

After medium256 stability is understood, expand with controlled design:

- more stack templates;
- more heat-source modes;
- richer k-field distributions;
- more held-out stack and held-out BC categories;
- clear train / valid / test_id / test_ood_bc / test_ood_stack splits;
- retained source-power, convergence, residual, and boundary diagnostics.

The key is structured coverage, not just sample count.

### Stage 3: Publication-style benchmark-candidate dataset

A stronger paper-facing dataset should include:

- multiscale stack structures;
- equivalent interconnect / blockwise equivalent regions;
- diagonal anisotropic conductivity;
- random power maps with physically audited integrated power;
- explicit boundary-condition encoding;
- fixed train/test protocol;
- multi-seed reporting;
- comparisons against zero_delta, MSE, `background_l1_relative`, and possibly
  simple CNN/FNO/GNO baselines if feasible.

Medium256 should remain as an ablation/debug set even after expansion.

## 5. Medium1024 Dataset Planning

Medium1024 should be designed before it is generated. The purpose is not to
hide or average away medium256 weaknesses, but to create a better controlled
setting for testing the weaknesses already identified:

- seed stability under fixed loss and learning-rate settings;
- low-DeltaT background bias, especially `bin_0`;
- high-DeltaT underprediction in `bin_3` and `bin_4`;
- OOD BC candidate mean-field robustness;
- OOD stack candidate mean-field robustness;
- interaction between held-out BC and held-out stack in a separate combined
  candidate split.

Medium256 should remain the fast ablation/debug set. Medium1024 should become a
larger benchmark-candidate preparation set only after its manifest, generator
support, checker, and partial-smoke protocol are reviewed.

The planned medium1024 protocol should include:

- a fixed 1024-sample manifest with explicit split counts;
- clear separation of `test_id`, `test_ood_bc_candidate`,
  `test_ood_stack_candidate`, and `test_ood_combined_candidate`;
- broader source/q patterns, including sparse, strip-like, low-background, and
  high-dynamic-range cases;
- broader k-field/material modes, including more diagonal anisotropic and
  blockwise-equivalent regions where supported;
- more stack templates and at least two held-out stack families;
- explicit marking of planned-only modes that are not yet consumed by the
  generator.

Publication-oriented use of medium1024 requires a fixed manifest, fixed split,
multi-seed reporting, condition-wise evaluation, and conservative scope
language. It should still be called a planned research benchmark candidate
until the full dataset exists, label diagnostics pass, and the evaluation
protocol is frozen.

The immediate implementation path is `medium1024_gapA`: a generation-ready
candidate that keeps the roadmap manifest intact while adding only low-risk
generator modes for low-power background cases, high-dynamic-range source
cases, high-contrast / low-k interface variants, and very-low / very-high
top-Robin candidates. This lets the project run 16/32-sample smokes and
64/128-sample pilots before deciding whether full 1024 generation is justified.
Medium256 remains the debug and ablation set throughout this process.

## 6. How To Optimize Toward A Publishable Version

### A. Experimental Protocol

- report multi-seed mean and standard deviation;
- report best, median, and worst runs;
- include split-wise and condition-wise evaluation;
- avoid only reporting the best seed;
- use a fixed training budget and fixed data generation manifest;
- avoid claims based only on smoke runs.

### B. Optimization Stability

- run seed 3 only if the seeds 0/1/2 summary leaves uncertainty that affects
  the next decision;
- focus LR exploration around `lr=0.01`, not uncontrolled large jumps;
- consider e500 only after checking validation trends, not as a default;
- consider checkpoint selection by validation metric;
- consider warmup_cosine around `lr=0.01` only after the multi-seed summary.

### C. Loss / Calibration

- keep `background_l1_relative` as the current useful loss;
- continue analyzing `bin_0` residual positive bias;
- avoid overfitting the loss design to seed 0;
- treat post-hoc calibration only as a diagnostic baseline;
- physical residual loss can be introduced later, after supervised loss
  behavior is better characterized.

### D. Dataset And Generalization

- expand structure variation before claiming arbitrary 3D IC structure
  handling;
- add more held-out stack types;
- include more random k/q distributions;
- keep solver and label diagnostics attached to every generated sample.

### E. Publication Story

A potential publishable angle could be:

```text
Physics-aware supervised neural operator diagnostics for multilayer 3D IC
thermal prediction with explicit boundary encoding and heterogeneous
conductivity fields.
```

The current version is not yet publication-ready. It is a promising
research-stage framework with a clear diagnostic pipeline and identified
technical bottlenecks.

## 7. Concrete Next Steps

### Priority 1

- Run the multi-seed summary for seeds 0/1/2.
- Decide whether seed 3 is necessary.
- Summarize stable metrics and uncertainty.

### Priority 2

- Decide whether current seed variance is acceptable for the next research
  milestone.
- Try checkpoint selection or longer training only if the multi-seed validation
  trend justifies it.

### Priority 3

- Design a medium512 or medium1024 manifest, but do not generate it until the
  experiment protocol is agreed.
- Define a stricter formal OOD stack split.

### Priority 4

Prepare paper-style figures and tables:

- dataset composition;
- model input/output contract;
- split-wise performance;
- condition-wise performance;
- error-bin diagnostics;
- seed sensitivity table.

## 8. Reporting Boundary

Use conservative wording:

- diagnostic;
- research-stage;
- controlled training;
- benchmark-candidate preparation;
- medium-scale experimental protocol.

Do not claim:

- formal benchmark completion;
- OOD generalization solved;
- high-fidelity solver validation;
- production-ready thermal simulator;
- publication-ready model performance.

## 9. V1 Freeze And V2 Transition

V1 is frozen as a diagnostic baseline. V2 will address model capacity, optimizer
design, hotspot / field-shape diagnostics, staged loss behavior, and a more
formal training protocol while keeping the frozen V1 results as the comparison
reference.
