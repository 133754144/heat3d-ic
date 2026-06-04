# Heat3D v2 e100 and M1.5 stratified results

Scope: existing `medium1024_gapA_full1024_v2` labels only. All completed runs use the stratified split map with `valid_iid` as primary validation and `valid_stress` as diagnostic validation. These are research-stage diagnostics, not formal benchmarks.

## E50 Baseline Vs E100

| run | epochs | best_epoch | best_valid_iid | final_valid_iid | final/best | final_valid_stress | wall-clock s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_3e-4_e50` | 50 | 45 | 0.4504 | 0.4706 | 1.045 | 0.6768 | 731.7 |
| `base_3e-4_e100` | 100 | 100 | 0.4154 | 0.4154 | 1.000 | 0.5930 | 1214.9 |
| `base_1e-4_e50` | 50 | 50 | 0.4995 | 0.4995 | 1.000 | 0.6654 | 729.7 |
| `base_1e-4_e100` | 100 | 100 | 0.4452 | 0.4452 | 1.000 | 0.6188 | 1206.1 |
| `warmup_cosine_e50` | 50 | 50 | 0.4887 | 0.4887 | 1.000 | 0.6464 | 734.0 |
| `warmup_cosine_e100` | 100 | 86 | 0.4339 | 0.4340 | 1.000 | 0.6036 | 1213.4 |

## M1 Vs M1.5 Capacity

| model/run | params summary | status | best_valid_iid | final_valid_iid | final_valid_stress | field_var valid_iid/stress | spatial_corr valid_iid/stress | p95/p99/peak error | top_k_overlap | hotspot_mae |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `M1_e50_base_3e-4` | latent64 / edge64 / steps4 | completed | 0.4504 | 0.4706 | 0.6768 | 1.388 / 4.115 | 0.782 / 0.771 | 0.0494 / 0.0959 / 0.0844 | 0.669 | 0.0652 |
| `M1_e100_base_3e-4` | latent64 / edge64 / steps4 | completed | 0.4154 | 0.4154 | 0.5930 | 1.325 / 3.952 | 0.797 / 0.789 | 0.0495 / 0.0935 / 0.0632 | 0.681 | 0.0625 |
| `M1_e100_lr1e-4` | latent64 / edge64 / steps4 | completed | 0.4452 | 0.4452 | 0.6188 | 1.193 / 3.807 | 0.784 / 0.771 | 0.0488 / 0.0955 / 0.0837 | 0.663 | 0.0644 |
| `M1_e100_warmup_cosine` | latent64 / edge64 / steps4 | completed | 0.4339 | 0.4340 | 0.6036 | 1.298 / 4.070 | 0.791 / 0.783 | 0.0498 / 0.0947 / 0.0763 | 0.665 | 0.0638 |
| `M1.5_e50_base_3e-4` | latent96 / edge96 / steps6 | OOM before epoch 1 completed | NA | NA | NA | NA | NA | NA | NA | NA |

M1.5 failed on SSH WSL during the first training epoch with:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 101.99MiB.
```

No `loss_summary.json`, predictions, or diagnostics were produced for M1.5. The result should be treated as a B192 memory feasibility failure, not a model-quality result.

## Interpretation

1. e100 clearly improves over e50. The strongest case is `base_3e-4`: best_valid_iid improves from 0.4504 to 0.4154, final_valid_iid improves from 0.4706 to 0.4154, and final/best improves from 1.045 to 1.000.
2. `base_3e-4_e100` is the best e100 configuration to keep. It has the best scalar valid_iid loss, best final_valid_stress among the completed e100 runs, and the best split-aware field-shape diagnostics among the e100 set.
3. e100 helps valid_stress but does not solve it. `base_3e-4` improves final_valid_stress from 0.6768 to 0.5930, and valid_stress spatial correlation improves to 0.789. However, stress field_variance_ratio remains high at 3.952, so stress over-amplitude remains unresolved.
4. `base_1e-4_e100` and `warmup_cosine_e100` confirm that these e50 runs were still under-trained, but neither surpasses constant `3e-4` at e100. `warmup_cosine_e100` is the closest alternative with best_valid_iid 0.4339.
5. M1.5 cannot be evaluated at B192 without a memory change. Possible next tests are M1.5 with lower batch size, gradient accumulation, or checkpointing/JIT memory work. Do not interpret the OOM as evidence against capacity scaling.
6. The current evidence favors extending/regularizing the existing M1 B192 training before adding more loss complexity. The old hotspot/background composite terms should not be reintroduced as-is.

## Resolution Assessment

Current low-resolution data is enough to validate the v2 pipeline, stratified split protocol, B192 baseline behavior, and split-aware diagnostics. It is not enough to fully describe real 3D IC local gradients around TSVs, bumps, BEOL, and interfaces. Do not generate new full high-resolution data yet; the next data-side step should be a small 64-128 sample high-resolution pilot.
