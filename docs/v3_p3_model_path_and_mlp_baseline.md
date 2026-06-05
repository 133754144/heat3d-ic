# Heat3D v3 P3 Model Path and MLP Baseline

## Purpose

P2-b showed graph coverage repair is safe but not sufficient: legacy,
nearest_repair, and discrete_radius all stayed around 73% to 75% relative RMSE
on the 16-sample fitting smoke. P3 therefore audits the RIGNO model path and
checks whether the same sample/target can be fit by a pointwise MLP.

## Commands

Local checks:

```bash
python3 scripts/audit_heat3d_v3_p3_model_path.py \
  --subset "/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small" \
  --output-json output/heat3d_v3_p3_model_path/model_path_audit_sample000.json

python3 scripts/run_heat3d_v3_pointwise_mlp_1sample_baseline.py \
  --subset "/Users/xuyihua/.codex/worktrees/8d2b/3D IC Heat/data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small" \
  --epochs 50 \
  --lr 1e-3 \
  --output-json output/heat3d_v3_p3_model_path/mlp_1sample_local_short.json
```

Devbox long run command:

```bash
python3 scripts/run_heat3d_v3_pointwise_mlp_1sample_baseline.py \
  --subset data/heat3d-thermal-simulation/subsets/v1_multilayer_bc_eq_supervised_small \
  --epochs 1000 \
  --lr 1e-3 \
  --hidden-size 128 \
  --hidden-layers 3 \
  --output-json output/heat3d_v3_p3_model_path/mlp_1sample_e1000.json
```

## Model Path Audit

Sample: `sample_000`. Feature names:

`k_x`, `k_y`, `k_z`, `q`, `is_top`, `is_bottom`, `is_side`, `is_interior`,
`top_h`, `top_T_inf_minus_T_ref`, `bottom_T_fixed_minus_T_ref`.

Identified columns:

- k: `[0, 1, 2]`
- q: `[3]`
- BC: `[4, 5, 6, 7, 8, 9, 10]`

Initialized RIGNO baseline: relative RMSE `72.01%`, finite output.

Sensitivity summary:

| ablation | output change RMSE | loss change | relative RMSE change |
| --- | ---: | ---: | ---: |
| zero_q | 2.180181e-02 | 9.147614e-03 | 2.705450e-03 |
| shuffle_q | 1.950792e-02 | 1.251918e-02 | 3.700063e-03 |
| zero_bc | 3.380874e-01 | 2.309599e-01 | 6.546071e-02 |
| shuffle_bc | 4.657461e-01 | 6.806598e-02 | 1.989392e-02 |
| shuffle_k | 4.406853e-01 | 7.265516e-02 | 2.121601e-02 |

Gradient norm summary:

| component | grad norm |
| --- | ---: |
| encoder | 8.967675 |
| processor | 0.251650 |
| decoder | 15.974733 |
| output | 1.789704 |
| other | 0.000000 |

The initialized path is sensitive to k/BC and less sensitive to q on this
sample. Gradients are nonzero through encoder, processor, decoder, and output,
but processor gradients are much smaller than encoder/decoder.

## MLP Baseline

Local 50-epoch smoke, hidden `128 x 3`, lr `1e-3`:

| best_epoch | best_loss | raw DeltaT RMSE | raw DeltaT MAE | relative RMSE | <=20% | <=2% |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 50 | 1.192728e-01 | 6.787140e-02 | 5.368451e-02 | 22.56% | false | false |

Devbox 1000-epoch result, hidden `128 x 3`, lr `1e-3`:

| best_epoch | final_loss | best_loss | raw DeltaT RMSE | raw DeltaT MAE | relative RMSE | <=20% | <=2% |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 976 | 9.228293e-05 | 7.969997e-05 | 1.754468e-03 | 1.312043e-03 | 0.583% | true | true |

The optional 3000-epoch / hidden-256 run was not needed because the 1000-epoch
baseline already passed both the 20% and 2% relative-RMSE gates.

## Next Judgment

The pointwise MLP fits the same sample and target to `<=2%` relative RMSE while
RIGNO P2 remains around high relative error. This points away from target
normalization as the primary blocker and toward RIGNO decoder/model-path
bottlenecks. Move next to a deeper RIGNO decoder/model-path audit before
changing loss or adding pointwise skip.
