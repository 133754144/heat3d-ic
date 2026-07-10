# V4P3_07/08 Training Assessment

Scope: read-only assessment of existing 600-epoch results. No retraining,
diagnostics rerun, prediction export, or artifact sync was performed.

Sources:

- V4P3_07: devbox
  `output/heat3d_v4_runs/V4P3_07/loss_summary.json`
- V4P3_08: WSL2
  `output/heat3d_v4_runs/V4P3_08/loss_summary.json`

Both runs used the formal candidate1024 split map with
`prediction_split=valid_iid`, `all_groups_status=skipped`,
`best_params_storage=cpu`, and `status_ok=true`.

## Metrics

`train_loss` at best epoch uses `epoch_mean_train_batch_loss`, because full
train metrics were only recorded at scheduled epochs.

| Run | Point | Epoch | train_loss | valid_base_mse | raw_rmse_K | rel_rmse_v4_pct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V4P3_07 | best | 103 | 0.271294 | 0.328003 | 1.617338 | 392.663 |
| V4P3_07 | final | 600 | 0.001506 | 0.413679 | 1.816288 | 440.965 |
| V4P3_08 | best | 129 | 0.147483 | 0.375970 | 1.731555 | 420.393 |
| V4P3_08 | final | 600 | 0.009691 | 0.435193 | 1.862950 | 452.294 |

## Overfit Judgment

Both runs overfit under the current 600-epoch regime.

V4P3_07 reaches best validation at epoch 103. After that, train loss continues
down from about 0.271 at the best point to 0.00151 at epoch 600, while
valid_base_mse worsens from 0.328 to 0.414.

V4P3_08 reaches best validation at epoch 129. After that, train loss continues
down from about 0.147 at the best point to 0.00969 at epoch 600, while
valid_base_mse worsens from 0.376 to 0.435.

## Suitability For V4

The current mode is useful as a diagnostic training path, but it is not a good
formal V4 training default yet. The model can fit the training split, but
validation improvement peaks early and then degrades. Best validation errors
are still high, with best rel_rmse_v4_pct around 393% for V4P3_07 and 420% for
V4P3_08.

V4P3_07 is better than V4P3_08 on this readout, so the boundary-distance path
remains the stronger candidate among these two, but the gap does not resolve the
training-regime problem.

## Next Directions

- semantic_full / q-log-k-log ablation
- sample weighting / curriculum
- short 50-100 epoch runs
- offline diagnostics

Optional best-checkpoint valid_iid/test_iid prediction export was skipped. Both
runs have best checkpoints, but generating split-level predictions would be a
new inference/export step and is outside this minimal read-only assessment.
