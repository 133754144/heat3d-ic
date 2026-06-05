# Heat3D v3 P3-b RIGNO Trained-Path Audit

## Purpose

P3-a showed the same `sample_000` target can be fit by a pointwise MLP to
`0.583%` relative RMSE after 1000 epochs, while P2 RIGNO graph-policy smoke
stayed at high small-sample error. P3-b therefore audits the trained RIGNO path
without changing model, decoder, loss, objective, or graph semantics.

The audit checks whether q/k/BC still affect trained outputs, whether processor
gradients and rnode latent updates are active, and whether the decoder depends
more on processed rnodes or latent pnodes.

## Commands

Local short check:

```bash
python3 scripts/audit_heat3d_v3_p3b_rigno_trained_path.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --policy legacy \
  --epochs 5 \
  --lr 1e-5 \
  --output-json output/heat3d_v3_p3b_rigno_path/local_legacy_e5.json
```

Devbox longer audit:

```bash
python3 scripts/audit_heat3d_v3_p3b_rigno_trained_path.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --policy legacy \
  --epochs 300 \
  --lr 1e-5 \
  --output-json output/heat3d_v3_p3b_rigno_path/devbox_legacy_e300.json

python3 scripts/audit_heat3d_v3_p3b_rigno_trained_path.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --policy nearest_repair \
  --epochs 300 \
  --lr 1e-5 \
  --output-json output/heat3d_v3_p3b_rigno_path/devbox_nearest_e300.json
```

## Local Check

`legacy`, 5 epochs, lr `1e-5`:

| phase | relative RMSE | raw RMSE | raw MAE | processor grad | rnode latent relative change | zero rnodes relative delta | zero pnodes relative delta | judgment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| initialized | 72.01% | 2.166314e-01 | 1.671729e-01 | 2.517e-01 | 8.107e-01 | 7.889532e-03 | -9.355914e-02 | active path |
| trained e5 | 71.52% | 2.151550e-01 | 1.661037e-01 | 2.395e-01 | 8.116e-01 | 1.066485e-02 | -9.017197e-02 | pnode-dominant |

Local e5 is only a compatibility check. It confirms finite output, nonzero
encoder/processor/decoder/output gradients, active processor latent changes, and
decoder sensitivity to both processed rnodes and latent pnodes.

## Devbox Result

300 epochs, lr `1e-5`, `sample_000`:

| policy | initial loss | final/best loss | trained relative RMSE | raw RMSE | raw MAE | judgment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| legacy | 1.214658e+00 | 1.093013e+00 | 68.30% | 2.054607e-01 | 1.565238e-01 | pnode-dominant |
| nearest_repair | 1.270305e+00 | 1.137618e+00 | 69.68% | 2.096111e-01 | 1.622061e-01 | pnode-dominant |

Trained-path sensitivity:

| policy | zero q rel delta | shuffle q rel delta | shuffle k rel delta | zero BC rel delta | shuffle BC rel delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 4.870105e-03 | 6.961931e-03 | -2.492655e-02 | 6.865218e-02 | 3.457322e-02 |
| nearest_repair | 4.934216e-03 | 6.551693e-03 | -2.900326e-02 | 5.883529e-02 | 3.293862e-02 |

Trained processor/decoder path:

| policy | encoder grad | processor grad | decoder grad | output grad | rnode latent rel change | zero rnodes rel delta | shuffle rnodes rel delta | zero pnodes rel delta | shuffle pnodes rel delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 3.258e+00 | 2.137e-01 | 2.554e+00 | 1.250e+00 | 7.830e-01 | 1.907341e-02 | 3.890747e-02 | -6.293266e-02 | 1.665166e-02 |
| nearest_repair | 3.489e+00 | 2.999e-01 | 2.860e+00 | 1.454e+00 | 8.068e-01 | 1.182549e-02 | 4.592917e-02 | -5.000570e-02 | 1.528543e-02 |

Both policies keep active q/k/BC sensitivity after training. q sensitivity is
weaker than BC/k but nonzero by output-change and loss-change checks. Processor
gradients are nonzero, and processor output changes rnode latents by roughly
`0.78x` to `0.81x` relative norm. Decoder ablation confirms dependence on both
processed rnodes and latent pnodes, but the larger pnode ablations make the path
pnode-dominant.

## Next Judgment

P3-b does not support processor-underuse as the primary failure mode: processor
gradients and rnode latent changes are active. It also does not show that
nearest-repair alone improves one-sample fitting: trained relative RMSE remains
near `70%`, slightly worse than legacy in this 300-epoch audit.

The current bottleneck looks more like decoder/regional path capacity or routing:
the model uses regional information, but the decoder remains pnode-dominant and
does not fit the same target that the pointwise MLP fits to `0.583%` relative
RMSE. Next work should audit decoder feature use and regional-to-point output
routing before changing loss/objective or adding P5 pointwise/local decoder
paths.
