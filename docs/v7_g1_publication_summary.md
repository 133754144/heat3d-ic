# V7 G1 Publication Summary

本页是后续论文写作的唯一 G1 summary 入口。所有数字均直接复制自 frozen formal evidence；未重新推理、未重新定义 metric。完整 machine-readable tables 见 [`docs/v7_g1_publication_tables/`](v7_g1_publication_tables/)。

## Frozen scope

- P1i `valid_iid` only；训练矩阵为 7 variants × 3 seeds = 21 runs，200 epochs。
- H1/H1b/H3/H4 使用 native 1024-point registered metrics。
- H2 是一个 hypothesis group，包含两个预注册 contrasts：H2a 与 H2b。
- H2 primary 为 `U16384→240825`；`U-direct-240825` 仅作 reconstruction-route robustness。
- H2 primary metric 为 `source_region_RMSE_K`；effect 定义为 `ablation_error − Full_error`。
- H2 bootstrap：10,000 次 two-level resampling，seed `20260829`，percentile 95% CI。

## Main ablations

| Hypothesis | Frozen comparison | Primary metric | Effect | 95% CI | Claim |
| --- | --- | --- | ---: | --- | --- |
| H1 | Full vs Vanilla | `point_global_relative_rmse_pct` | 20.43909 pp | [17.12366, 23.60488] pp | `SUPERIORITY_SUPPORTED` |
| H1b | Full vs capacity-matched Vanilla | `point_global_relative_rmse_pct` | 22.37245 pp | [17.44747, 27.71174] pp | `SUPERIORITY_SUPPORTED` |
| H3 | Full vs no FiLM | `sample_first_relative_rmse_pct` | 0.29417 pp | [0.21456, 0.37041] pp | `SUPERIORITY_SUPPORTED` |
| H4 | Full vs scale-correction-off | `raw_K_CV_RMSE_K` | 171.44018 K | [164.25621, 178.46001] K | `SUPERIORITY_SUPPORTED` |

H1/H1b 的结论仅说明完整 Heat3D conditioning architecture 相对 Vanilla 的组合收益；H1b 同时保留 capacity-matched Vanilla 的逐 seed 高方差。

## H2a / H2b common-domain attribution

H2a 与 H2b 是同一个 H2 hypothesis group 内的两个 preregistered contrasts，不是三个独立 hypothesis。

| Contrast | Comparison | Primary effect (K) | 95% CI (K) | Per-seed effects (0/1/2) | Claim |
| --- | --- | ---: | --- | --- | --- |
| H2a | Full vs generic support | 1.74615 | [1.48199, 2.04268] | 1.96483 / 1.56857 / 1.70998 | `SUPERIORITY_SUPPORTED` |
| H2b | Full vs CV-only support | 1.84057 | [1.53070, 2.18014] | 1.66052 / 1.71218 / 2.13916 | `SUPERIORITY_SUPPORTED` |

两条 U route 对 H2a/H2b 均保持相同的正向 attribution direction 与 claim status；direct route 只作为 sensitivity，不替代 primary。support 的正式名称是 `physics-layout-aware sparse support`，不是 source-amplitude-aware support。

## Interpretation boundary

G1 支持 Heat3D internal mechanisms matter：完整 conditioning architecture、physics-layout-aware sparse support、FiLM secondary contribution，以及当前 frozen P1i formulation 下 learned scale correction 均有对应的内部 ablation evidence。

G1 不证明 RIGNO 优于 GINO、Transolver 或其它外部模型；不作 `test_iid`、OOD、external-superiority、SOTA 或 deployment claim。native-1024 H2 结果是 supplementary diagnostic；旧 mixed-domain summary 不用于 direct cross-variant comparison。

## Evidence pointers

- Main ablation table: [`g1_main_ablation_table.json`](v7_g1_publication_tables/g1_main_ablation_table.json)
- H2 common-domain table: [`h2_common_domain_attribution_table.json`](v7_g1_publication_tables/h2_common_domain_attribution_table.json)
- H2 route robustness table: [`h2_u_route_robustness_table.json`](v7_g1_publication_tables/h2_u_route_robustness_table.json)
- Parameter/control table: [`parameter_count_control_table.json`](v7_g1_publication_tables/parameter_count_control_table.json)
- Figure plan/index: [`v7_g1_figure_manifest.json`](v7_g1_figure_manifest.json)
- Final science seal: [`v7_g1_final_science_seal_receipt.json`](v7_g1_final_science_seal_receipt.json)
