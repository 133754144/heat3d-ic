# Heat3D-IC: Steady 3D IC Thermal Field Prediction

Heat3D-IC is a research codebase for steady thermal surrogate modeling of
multilayer heterogeneous 3D IC structures with graph neural operators. The
current repository state closes V4 around a clean-IID baseline and an explicit
hard-challenge protocol; it does not claim industrial thermal-signoff accuracy
or solved out-of-distribution behavior.

For the stage history and implementation overview, see
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). For the frozen V4 result table and
metric definitions, see [docs/v4_closeout.md](docs/v4_closeout.md). Upstream
inheritance and citation requirements are in [ATTRIBUTION.md](ATTRIBUTION.md).

## Project Overview

The current V4 task is:

```text
[k(x), q(x), boundary conditions, geometry/extent features]
    -> steady DeltaT(x) / T(x)
```

The standard model path is equivalently expressed as:

```text
coords + k(x) + q(x) + BC -> T(x)
```

Boundary conditions are model inputs. Layer-stack and interface metadata are
kept for dataset generation and evaluation grouping rather than added as
default model features.

## Current V4 Physical Scope

V4 covers supervised steady-state temperature-rise prediction for synthetic,
multilayer 3D IC-like structures with:

- heterogeneous conductivity fields `k(x)`, localized power fields `q(x)`,
  and boundary-condition features;
- sample-local coordinates plus geometry/extent features used by the active
  semantic input path;
- recovered absolute temperature `T(x)` from predicted DeltaT and reference
  boundary temperature;
- a non-periodic RIGNO graph path with discrete physical coverage and repair;
- explicit clean-IID, hard-challenge, all-IID reporting, and fixed final probes.

The scope does not yet include a production package model, transient dynamics,
hard PDE or Dirichlet constraints, discrete residual training, or multi-seed
performance evidence.

## Current Dataset And Split

The current V4 benchmark dataset is ignored locally and is not the historical
UnitCube demo:

- dataset: `data/heat3d_v4_p5_clean_nohard_v0`;
- formal split:
  `configs/heat3d_v4/candidate1024_p5_clean_nohard_train672_valid128_test128_hardchallenge_seed0.json`;
- clean train / valid_iid / test_iid: `672 / 128 / 128`;
- hard_train_holdout / hard_challenge_valid / hard_challenge_test:
  `121 / 12 / 12`.

The clean splits contain zero `physical_hard_keep` samples. Hard samples remain
unchanged in their challenge roles. `all_iid` is a reporting union of the
matching clean and hard split; it is never a training split.

The legacy `v0_unitcube_demo` remains useful only as an execution smoke
dataset. It is not the current V4 training dataset, benchmark, or baseline.

## Frozen V4 Baseline

`V4P5_02_clean_baseline_raw_B28_e600` is the frozen V4 clean baseline:

- raw coordinate encoding, plain MSE, B28, seed 0, and clean-nohard training;
- 600-epoch schedule, with the formal selected checkpoint at best epoch 405;
- selection metric: normalized `valid_base_mse`, not a raw or final-probe
  diagnostic;
- checkpoint kind below is `best`; RMSE and MAE values are K, while corr and
  amplitude ratio are dimensionless. Split metrics are sample-first means
  unless explicitly marked point-global.

| split | checkpoint | sample-first RMSE K | sample-first MAE K | corr | amp ratio | point-global raw DeltaT RMSE K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clean valid_iid | best e405 | 0.119 | 0.058 | 0.962 | 1.095 | 0.170 |
| clean test_iid | best e405 | 0.153 | 0.074 | 0.964 | 1.076 | 0.236 |

The best checkpoint has point-global clean-valid `rel_rmse_v4_pct=67.17%`.
Hard challenge remains substantially harder: sample-first RMSE is 4.709 K on
hard valid and 3.683 K on hard test. See the complete best/final tables,
point-global cross-checks, low-DeltaT diagnostics, and final-probe results in
[docs/v4_closeout.md](docs/v4_closeout.md).

`V4P5_03_clean_fourier_freq4_B_safe` completed at B28 without OOM, but its
best clean valid/test sample-first RMSE is 0.148/0.175 K. It also degrades P02
and P06 final probes despite a local P09 gain. Fourier frequency 4 is therefore
a negative V4 ablation, not a V5 default.

## Reproducible Entry Points

Run from the repository root after placing the P5 dataset at the ignored path
above and activating the project environment:

```bash
python3 scripts/run_heat3d_v4_config.py \
  --config configs/heat3d_v4/generated/V4P5_02_clean_baseline_raw_B28_e600.yaml
```

The registry is authoritative:

```bash
python3 -B scripts/check_heat3d_v4_registry.py
python3 -B scripts/prepare_heat3d_v4_run.py \
  --config-id V4P5_02_clean_baseline_raw_B28_e600 \
  --config-id V4P5_03_clean_fourier_freq4_B_safe \
  --dry-run
```

The commands above prepare or launch a registered V4 path only when the
operator has explicitly approved training. They do not make a clean-IID result
equivalent to hard-challenge or final-probe performance.

### Legacy UnitCube Smoke

The older UnitCube commands are retained only to check the historical minimal
execution path. They are not the V4 benchmark:

```bash
python3 scripts/train_heat3d_operator.py \
  --data-dir data/heat3d-thermal-simulation/subsets/v0_unitcube_demo/samples \
  --epochs 1 --batch-size 1 --n-train 1 --n-valid 1 --n-test 1 \
  --output-dir /tmp/heat3d_unitcube_legacy_smoke \
  --checkpoint-name smoke.pkl
```

## Evaluation Protocol

V4 uses the metrics contract in
[configs/heat3d_v4/metrics_v0.json](configs/heat3d_v4/metrics_v0.json):

1. Select checkpoints only with normalized `valid_base_mse`.
2. Report MSE/RMSE/MAE as model-performance metrics and separately identify
   whether they are sample-first or point-global.
3. Report clean valid_iid/test_iid, hard_challenge valid/test, and their
   all-IID reporting unions separately.
4. Treat raw DeltaT, final-probe, region/hotspot, and low-DeltaT diagnostics as
   report or diagnosis metrics, not replacement selection metrics.
5. Aggregate per sample before reporting split/group mean, median, or standard
   deviation. A flattened global error alone is insufficient.

The P5 closeout used read-only checkpoint inference for missing test/hard
predictions. Those artifacts remain ignored under `output/` and are not part
of this repository history.

## RIGNO Adaptation

The repository retains the RIGNO graph-operator core and adds Heat3D-specific
data semantics, graph construction, training controls, checkpoint provenance,
and split-aware diagnostics. The active V4 path uses semantic normalization,
sample-local isotropic coordinates, log-extent broadcast features, a
post-decoder condition residual, and `valid_iid` prediction by default.

## Limitations

- The P5 data remains synthetic and controlled rather than a complete package,
  chiplet, TSV, BEOL, or thermal-signoff workload.
- The hard challenge has large amplitude/top5/strong-q errors and remains a
  separate evaluation line.
- Results are seed-0 evidence, not a multi-seed claim.
- The current model has no implemented hard bottom-Dirichlet constraint or
  discrete physical residual objective.
- Raw coordinate encoding is the V4 baseline; Fourier frequency 4 did not
  improve this setting.

## V5 Roadmap

The following are planned research directions only:

- q/target decomposition;
- a global physics-scale branch;
- shape-scale decomposition for amplitude failures;
- bottom-Dirichlet hard constraint;
- discrete physical residual metrics; and
- multi-seed evaluation.

## Citation And Attribution

This project derives its retained graph-operator core from RIGNO:

- Upstream repository: <https://github.com/camlab-ethz/rigno>
- Paper: <https://arxiv.org/abs/2501.19205>

Please cite RIGNO when using the retained upstream model core:

```bibtex
@inproceedings{mousavi2025rigno,
  title         = {RIGNO: A Graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains},
  author        = {Sepehr Mousavi and Shizheng Wen and Levi Lingsch and Maximilian Herde and Bogdan Raonic and Siddhartha Mishra},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  year          = {2025}
}
```
