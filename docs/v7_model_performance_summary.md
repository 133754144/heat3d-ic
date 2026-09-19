# V7 cross-stage model performance summary

本表只汇总冻结的 valid-only 证据，不构成跨评价域 overall leaderboard。所有 `±`
均为 **SD across training seeds**。sealed IID 未打开；本阶段没有训练、调参或
checkpoint reselection。

## Metric contract

- 全局 RMSE：`point_global_relative_rmse_pct`。
- 样本 RMSE：`sample_first_relative_rmse_pct`。
- 主 Corr：`Corr_CV (%)`；主 Amp：`Amp_CVRMS (%)`。
- `Corr_point` / `Amp_range` 仅作 supplementary diagnostics。
- RMSE [K] 为当前 evaluator 的 unweighted physical-node RMSE；V6 historical
  `2.967838 ± 0.019180 K` 明确是 CV-weighted historical RMSE，不能与当前
  `rmse_K` 静默混用。

## A. P1i native1024

| Model | Training cases | Training budget / checkpoint | Domain | Global RMSE % | Sample RMSE % | RMSE K | Corr_CV % | Amp_CVRMS % |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Heat3D V6 e600-trained / PG-best | 768 | 600e; PG-best e559/e455/e587; not epoch600 checkpoint | P1i native1024 | 2.0273±0.0947 | 1.6294±0.0131 | N/A | N/A | N/A |
| Heat3D V7 P1i-e200 | 768 | independent 200e cohort | P1i native1024 | 2.0577±0.0806 | 1.6919±0.0478 | 1.6045±0.0628 | 99.2716±0.0548 | 99.8791±0.2151 |
| GINO | 768 | 301e; valid-selected best | P1i native1024 | 17.7315±1.6551 | 15.0139±0.9130 | 13.8267±1.2906 | 96.2965±1.6137 | 102.8707±0.1361 |
| Transolver | 768 | 500e; valid-selected best | P1i native1024 | 18.5565±0.9266 | 16.0028±1.0310 | 14.4701±0.7225 | 97.2029±0.2197 | 101.9345±1.7185 |

V6 的 Corr/Amp 为 `N/A — exact three-seed prediction artifacts unavailable`，未用
其他 checkpoint 或 aggregate 推算。

## B. P1i full-field240825

| Model | Training cases | Training budget / checkpoint | Domain | Global RMSE % | Sample RMSE % | RMSE K | Corr_CV % | Amp_CVRMS % |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Heat3D V6 historical | 768 | 600e-trained / PG-best family | P1i full-field240825 | 3.4426±0.0584 | 3.9583±0.0097 | 2.9678±0.0192 CV-weighted historical | N/A | N/A |
| Heat3D V7 P1i-e200 + U-v2 | 768 | independent 200e; frozen sidecars | P1i full-field240825 | 3.4786±0.1728 | 3.1262±0.1772 | 2.7625±0.1372 | 98.9563±0.0819 | 97.3367±0.2102 |
| Therm-FM | 768 | 200e; loss-selected normalized-loss checkpoint | P1i full-field240825 | 3.1490±0.4603 | 2.5972±0.2399 | 2.5007±0.3656 | 98.2852±0.1678 | 100.0987±0.0140 |

## C. DeepOHeat-v1 full-field571256

| Model | Training cases | Training budget / checkpoint | Domain | Global RMSE % | Sample RMSE % | RMSE K | Corr_CV % | Amp_CVRMS % |
|---|---:|---|---|---:|---:|---:|---:|---:|
| DeepOHeat-768 | 768 | 100k iterations × 50 sampled function instances; valid-selected best | DeepOHeat-v1 full-field571256 | N/A | 1.3052±0.2683 | N/A | N/A | N/A |
| Heat3D-768-e600 | 768 | 600e fixed endpoint; U-v2 direct-query | DeepOHeat-v1 full-field571256 | N/A | 0.7099±0.0078 | N/A | N/A | N/A |
| DeepOHeat-full-minus-valid128 | 99,872 | 100k iterations × 50 sampled function instances; valid-selected best | DeepOHeat-v1 full-field571256 | N/A | 1.1467±0.0383 | N/A | N/A | N/A |

主配对是 `DeepOHeat-768 vs Heat3D-768-e600`（same physical-case count）。
`DeepOHeat-full-minus-valid128` 仅作为 99,872-case large-data regime reference。该
表不支持 same-information-budget 或 same-compute claim。

## Interpretation boundary

三组分别回答 native sparse、P1i full-field capability evolution 和
DeepOHeat-domain data-regime comparison；禁止跨组统一排名。V6 行严格称为
`V6 e600-trained / point-global-best checkpoint family`，不称为 epoch600 checkpoint。
完整机器可读内容见 `docs/results/v7_model_performance_summary.json`。
