# V5 Gate 6A no-training diagnosis

仅访问 `train` 与 `valid_iid`；未加载或评估 test/hard。checkpoint 为 N3 best e402。

## Point-global oracle views

| split | joint % | oracle-scale % | oracle-shape % | physics-scale % |
|---|---:|---:|---:|---:|
| train | 4.3067 | 3.9509 | 1.7820 | 67.4996 |
| valid_iid | 24.0756 | 17.5261 | 17.6443 | 65.1565 |

## Loss and gradient scale

| split | loss | mean | global grad norm | backbone | shape decoder | scale head | FiLM | bypass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | shape_cv_loss | 0.00201594 | 0.0392508 | 0.0337793 | 0.0194381 | 0 | 0.00185026 | 0.00428052 |
| train | log_scale_loss | 0.000234557 | 0.288085 | 0.0371722 | 0 | 0.284353 | 0.0274745 | 0 |
| train | relative_field_loss | 0.00222567 | 0.255233 | 0.0499234 | 0.0191913 | 0.248294 | 0.0248158 | 0.00422453 |
| train | raw_absolute_field_loss | 0.00151851 | 0.282774 | 0.0416286 | 0.0182323 | 0.274082 | 0.0525828 | 0.0031016 |
| valid_iid | shape_cv_loss | 0.0278337 | 0.175057 | 0.14725 | 0.0917176 | 0 | 0.0166774 | 0.0164973 |
| valid_iid | log_scale_loss | 0.0302585 | 1.13465 | 0.165882 | 0 | 1.10828 | 0.177836 | 0 |
| valid_iid | relative_field_loss | 0.0524829 | 0.533597 | 0.197753 | 0.0869596 | 0.471661 | 0.123942 | 0.0152396 |
| valid_iid | raw_absolute_field_loss | 0.0295448 | 0.328836 | 0.118274 | 0.0397842 | 0.299038 | 0.0559492 | 0.00276937 |

完整四分位、gradient cosine、Q4、top-5/top-10 与逐 top-10 样本贡献见 JSON。
