# Heat3D-IC: Graph neural operators for steady 3D IC thermal fields

Heat3D-IC is a research codebase for surrogate modeling of steady thermal
fields in heterogeneous, multilayer 3D IC-like structures. The repository
uses a graph neural operator derived from [RIGNO](https://arxiv.org/abs/2501.19205)
and evaluates it against a finite-volume reference workflow.

The current scientific baseline is the frozen V6/P1i closeout. The active
research branch is `research/v7`, whose purpose is publication-readiness:
standardized evaluation, a clean reproducible ML pipeline, stronger baselines,
physics and OOD evidence, and a defensible deployment boundary. V7 capabilities
that have not been measured are research objectives, not current results.

The V7 contract is [docs/v7_research_contract.md](docs/v7_research_contract.md).
The static runtime/dependency findings are in
[docs/v7_active_dependency_audit.md](docs/v7_active_dependency_audit.md).
Historical stage documents remain unchanged; the V6 evidence index is
[docs/v6_phase_index.md](docs/v6_phase_index.md).

## Problem formulation

The supervised operator-learning task is written as

```text
(coordinates, k(x), q(x), Robin boundary conditions, geometry/interface metadata)
    -> steady DeltaT(x) / T(x)
```

Here `k(x)` is the heterogeneous conductivity field and `q(x)` is the
distributed heat-source field. The model produces a temperature-rise field;
absolute temperature is recovered with the reference boundary temperature
when that reporting path is used. The V6/P1i formal setting has ideal interface
contact (`R_contact=0`) and a finite-volume reference field with 240,825 nodes.

The V6/P1i high-resolution route is:

```text
1,024 source-aware condition points
    -> graph neural operator inference
    -> 16,384-point query field
    -> layer/interface-aware reconstruction
    -> 240,825-node full field aligned with the FVM reference
```

This describes the frozen V6 route. It does not establish that the same route
is optimal for V7, arbitrary geometries, or unseen physical regimes.

## Current validated V6/P1i results

The formal P1i dataset is
`heat3d_v6_p1i_continuous_physics1024_v1` with a `768/128/128`
train/validation/test split. The primary frozen checkpoint is
`V6_06_V5best_P1i_seed0_reliable_B24`, selected at epoch 559. The dataset,
checkpoint, full-field sidecar, protocol, and artifact identities are recorded
in [docs/v6_p1i_closeout.md](docs/v6_p1i_closeout.md) and
[docs/v6_p1i_p6a_publication_tables.md](docs/v6_p1i_p6a_publication_tables.md).

The table below is the tracked V6/P1i `E16384_reconstruction` evidence. The
`test_iid` result is a corrected 128-sample confirmatory held-out test set that
was opened once after route and checkpoint freeze; it was not used for model or
route selection. Metric names are kept separate because they aggregate errors
differently.

| metric | valid32 | test_iid (128) |
| --- | ---: | ---: |
| point-global relative RMSE | 2.7023% | 2.9920% |
| sample-first CV-relative RMSE | 2.7385% | 2.9485% |
| raw CV RMSE | 2.2848 K | 2.3891 K |
| source-region RMSE | 3.8721 K | 3.9405 K |
| peak RMSE | 4.0178 K | 5.7263 K |
| interface RMSE | 0.3857 K | 0.3555 K |

The primary WSL2 timing evidence uses the public boundary
`in-memory k/q/BC -> synchronized full-field result`, paired against the
persistent CPU FVM reference:

- fresh median/p95: `0.8832/0.9830 s` for the neural reconstruction route;
- Q2 throughput: `1.7263 sample/s`;
- paired fresh/Q2 speedup versus the FVM reference: `2.001x/2.005x`;
- resident timing: `0.00644 s`, reported separately from the public fresh
  end-to-end boundary.

These are V6/P1i results, not V7 results. The three lifecycle timing seeds are
paired workload repetitions and hardware-state evidence; they are not three
additional model seeds. The V6 evidence summary explains the machine roles,
timing lifecycle, peak-error tail, and frozen sealed-data status in
[docs/v6_p1i_publication_evidence_summary.md](docs/v6_p1i_publication_evidence_summary.md).

## Evaluation terminology frozen for V7

V7 separates two evaluation levels. A result must state its level, split,
query resolution, reconstruction status, timing boundary, and reference
solver status.

### Training-resolution evaluation (Level-A)

Level-A is native-resolution model evaluation. The model is trained and tested
at the registered conditioning/query resolution, with no claim that a later
reconstruction step provides deployment-scale evidence. V7 will use Level-A to
answer whether source-aware conditioning, context, and scale features improve
over a vanilla RIGNO baseline under matched data, parameter, training, and
hardware budgets.

The current V6/P1i table above is an `E16384_reconstruction` full-field result;
it must not be relabeled as a V7 Level-A native-1024 accuracy result.

### High-resolution deployment evaluation (Level-B)

Level-B evaluates the deployment workflow: a lower-resolution conditioning
set, a high-resolution query set such as 16k or 32k, optional
layer/interface-aware reconstruction, and a full-field comparison with the
240,825-node FVM reference. It reports accuracy, reconstruction error, memory,
failure cases, and the complete timing boundary. Fresh, cache-hot, resident,
Q1/Q2, and known-topology reuse are separate measurements.

The frozen V6/P1i evidence supports a 16k reconstruction route on the stated
valid/test roles. The V7 contract treats 16k as principal, 32k as exploratory
until qualified, and 64k as optional stress testing rather than a production
claim. No cache-hot or known-topology number replaces a new-case Level-B
end-to-end measurement.

## V7 research objectives

V7 is not a new result set in this README. Its objectives are governed by
G0–G9 in the contract:

1. establish a reproducible ML pipeline and remove production-path dependence
   on smoke, check, development, and private cross-script APIs;
2. compare against vanilla RIGNO and representative external/strong baselines
   under a matched protocol, with explicit ablations;
3. test conditioning/query resolution decoupling at 16k, 32k, and optionally
   64k with an accuracy–latency–memory Pareto report;
4. measure conservation, flux, Robin-boundary, interface, and hotspot residuals;
5. evaluate geometry, source, and physical OOD regimes, including a controlled
   finite-contact-resistance case when registered;
6. complete at least one external benchmark or industrial-style case and one
   thermal design-optimization loop;
7. open a newly named V7 sealed test exactly once after all decisions are
   frozen, then publish a clean-checkout artifact.

The candidate method claims are intentionally conditional:

- physics-conditioned operator;
- source-aware sparse conditioning;
- resolution-decoupled inference; and
- deployment-scale acceleration under matched production boundaries.

Therm-FM and DeepOHeat-v2 are literature targets that inform the V7 baseline,
OOD, physics, and trustworthiness plan. Their capabilities are not claims about
this repository. The same applies to any future comparison with FNO, DeepONet,
GINO, MeshGraphNets, or other thermal surrogates until the comparison is
actually registered and measured.

## Frozen artifacts and data policy

V6/P1i data, sidecars, predictions, checkpoints, logs, raw matrices,
publication tables, manifests, receipts, selection records, and sealed-data
status are frozen. V7 documentation and future experiments must not overwrite,
rename, regenerate, or reinterpret them. In particular, V6 `test_iid` is
historical confirmatory evidence and V6 sealed IID remains ungenerated and
unopened.

This D0/G0a documentation stage does not train, infer, solve, generate data,
evaluate a model, access held-out or sealed labels, or modify a model or frozen
artifact. A V7 Quick Start is intentionally not published yet: the active
dependency audit has found that the current formal paths still cross legacy
and smoke-named modules, so presenting a clean V7 command would be misleading.

## Limitations

- The validated P1i evidence is synthetic and comes from one registered formal
  distribution; it is not an industrial sign-off workload.
- The formal physical setting uses `R_contact=0`; finite contact resistance,
  experimental packaging, and temperature-dependent material behavior are not
  validated V6 claims.
- The current evidence does not establish arbitrary-geometry, cross-chip,
  few-shot, or universal PDE transfer.
- The current evidence does not establish uncertainty quantification,
  calibration, hotspot trustworthiness, or physics-consistent generalization.
- The test peak error is larger than the aggregate field errors and has a
  disclosed high-energy tail; average field metrics do not replace hotspot
  analysis.
- The approximately `2x` fresh speedup is the supported public V6/P1i timing
  claim. Resident and known-topology reuse timings have different boundaries
  and must not be combined with it.
- V7 has not yet validated its external baselines, ablations, OOD matrix,
  physics residuals, design loop, or final sealed test.

## References and attribution

The graph-operator backbone is derived from RIGNO:

- upstream repository: <https://github.com/camlab-ethz/rigno>;
- paper: <https://arxiv.org/abs/2501.19205>.

The project-specific literature map and citation responsibilities are in
[ATTRIBUTION.md](ATTRIBUTION.md). The V7 literature and claim boundary are
maintained in [docs/v7_research_contract.md](docs/v7_research_contract.md).

```bibtex
@inproceedings{mousavi2025rigno,
  title         = {RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains},
  author        = {Sepehr Mousavi and Shizheng Wen and Levi Lingsch and Maximilian Herde and Bogdan Raonic and Siddhartha Mishra},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025}
}
```
