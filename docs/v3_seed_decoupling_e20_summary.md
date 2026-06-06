# Heat3D v3 Seed-Decoupling e20 Summary

## B96 e20 Results

| run | model_seed | batch_order_seed | graph_seed | initial valid_iid | final valid_iid | best valid_iid | initial stress | final stress | best stress | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model0_batch0_graph0 | 0 | 0 | 0 | 1.143563 | 0.397296 | 0.397296 | 1.067964 | 0.488732 | 0.488732 | 20 |
| model1_batch0_graph0 | 1 | 0 | 0 | 1.212437 | 1.001534 | 1.001226 | 1.250301 | 1.030514 | 1.030116 | 18 |
| model2_batch0_graph0 | 2 | 0 | 0 | 1.659969 | 1.000030 | 1.000030 | 1.672051 | 1.029391 | 1.029391 | 20 |
| model0_batch1_graph0 | 0 | 1 | 0 | 1.143562 | 0.402916 | 0.402916 | 1.067965 | 0.491428 | 0.491428 | 20 |
| model0_batch2_graph0 | 0 | 2 | 0 | 1.143563 | 0.418902 | 0.418902 | 1.067966 | 0.504830 | 0.504830 | 20 |

Conclusions:

- Model initialization dominates this e20 matrix: `model_seed=1/2` stay near valid_iid loss 1.0, while `model_seed=0` reaches 0.40-0.42.
- Batch order has a secondary effect under `model_seed=0`: batch seeds 0/1/2 end at 0.397/0.403/0.419 valid_iid loss.
- The old seed1 e400 weakness is not explained mainly by coupled batch order; decoupled e20 with `model_seed=1,batch_order_seed=0` is already weak.
- Do not prioritize a decoupled seed1 e400 rerun yet. It is better to first test whether batch composition changes improve the early trajectory.

## B88 Batch Composition Audit

Audits were run on devbox and wrote ignored outputs under
`output/heat3d_v3_batch_audit/`.

| plan | batch sizes | max stack fraction | max source fraction | max k-region fraction | max bc fraction | max power-scale fraction | summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B96 current_graph_shape | 7x96 + 1x32 | 0.781-1.000 | 0.583-0.958 | 0.510-1.000 | 0.510-1.000 | 0.594-1.000 | Tail batch remains; many batches are near single stack/BC/power class. |
| B88 current_graph_shape | 8x88 | 0.568-1.000 | 0.727-0.989 | 0.511-1.000 | 0.920-1.000 | 0.784-1.000 | Removes B32 tail but still has highly concentrated graph-shape batches. |
| B88 sample_shuffle seed0 | 8x88 | 0.273-0.352 | 0.159-0.193 | 0.216-0.250 | 0.352-0.511 | 0.807-0.909 | Much more balanced stack/source/k-region/BC distribution; power-scale remains skewed because the train split itself is skewed. |

Conclusion:

- B88 alone is not enough if the runner keeps graph-shape grouping before
  chunking.
- B88 `sample_shuffle` is worth a next e20 batching-policy evaluation, pending a
  runner design decision. No B88 training YAML is prepared in this task.
