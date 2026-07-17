# V32 attention diagnostics

Scope is valid_iid-only; test/hard/sealed were not accessed.

| checkpoint | epoch | entropy | max weight | residual/mean-pool | classification |
|---|---:|---:|---:|---:|---|
| point_global_best | 474 | 0.861658 | 0.047089 | 0.755705 | effective_regional_selection |
| legacy_best | 474 | 0.861658 | 0.047087 | 0.755706 | effective_regional_selection |
| sample_first_best | 366 | 0.860971 | 0.046440 | 0.737419 | effective_regional_selection |
| final | 600 | 0.860726 | 0.047484 | 0.759337 | effective_regional_selection |

For the point-global checkpoint, mean per-sample Pearson/Spearman correlations are:

| feature | Pearson | Spearman |
|---|---:|---:|
| log1p_q_inverse_kz_relative | -0.155270 | -0.205135 |
| log1p_q_relative | -0.129580 | -0.181598 |
| log_inverse_kz_relative | -0.440721 | -0.634077 |
| source_present_fraction | -0.136643 | -0.182248 |

The attention is neither uniform nor collapsed. Its residual is large relative to mean pooling, while the learned weights are negatively correlated with source occupancy/q and most strongly with inverse-kz. This supports an attention-bias/residual-strength audit before additional seeds.
