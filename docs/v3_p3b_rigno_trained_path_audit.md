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

Pending.

## Next Judgment

If the 300-epoch RIGNO audit remains high-error while q/k/BC sensitivity and
gradients are active, the bottleneck is more likely decoder/regional path
capacity or routing than target normalization. If q sensitivity remains weak
after training, input scaling or heat-source encoding should be audited before
adding pointwise skip.
