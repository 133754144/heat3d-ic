# GINO E3 bounded qualification

## Verdict

`E3_PASS` / `GINO_AUTHOR_SEMANTICS_QUALIFIED`。

本次 qualification 使用 pinned Open3D `FixedRadiusSearch` + `torch-scatter`，
纯 PyTorch 仅作 diagnostic oracle。没有加入新 tolerance，也没有把 E2 的跨进程
strict-repeatability 作为准入条件。

## Execution receipt

- devbox WSL2，RTX 5070，CUDA 13.0，PyTorch 2.9.0+cu130，Open3D 0.19.0，
  torch-scatter 2.1.2+pt29cu130。
- upstream `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`。
- 6 个 fresh child processes：3 个同 seed（20260907）重复、3 个 inter-seed
  reference（0/1/2）；每个包含 train 与 valid_iid fixture、5 次 optimizer update。
- raw aggregate：`/tmp/g2_p13_gino_e3_qualification_20260915.json`，SHA256
  `e5393305822383b6e8cdaca230174ec36eff1a45c3d91bd91e7a6e3b5f947725`。
- runner SHA256：`9b7a96ad0690b804f14288aa768991a41a3939a149d990c4cd2764dff0e49ca1`。
- frozen `r_in=0.15`、`r_out=0.033`、latent `32^3`、11 physical features、
  train-only statistics、Open3D/scatter semantics 均保持不变。

## Observations

所有 forward/loss/gradient/update finite；authoritative backend 的 graph 在固定
fixture 上可重复；checkpoint restore 在 child 内 state 相等。896 geometry audit
沿用已关闭的非边界 disagreement=0；2057 条差异均已标记为 radius-boundary
classification，不再要求 fallback bitwise/allclose。

same-seed 与 inter-seed 的中位相对量如下：

| quantity | same-seed | inter-seed | same/inter |
| --- | ---: | ---: | ---: |
| loss | 5.30e-6 | 3.93e-2 | 1.35e-4 |
| prediction | 2.57e-4 | 4.68e-1 | 5.48e-4 |
| parameters | 2.14e-4 | 1.42 | 1.51e-4 |
| updates | 1.14e-2 | 1.42 | 8.05e-3 |

这些量只用于数值稳定性诊断；没有读取 accuracy 来改变任何合同。冻结的 bounded
runner 没有采集 peak VRAM，因此该字段在 JSON 中明确为
`NOT_INSTRUMENTED_BY_FROZEN_BOUNDED_RUNNER`，不伪造数值。

## Release

GINO 可按已冻结 manifest 进入 formal cohort；formal backend 仅为
Open3D + torch-scatter，fallback 不能升级为 formal。E2 strict-repeatability
历史 FAIL-CLOSED receipt 保留但已被 E3 reproducibility policy supersede；不删除
或改写历史证据。正式训练仍须串行，并继续禁止 P1i `test_iid`/sealed。

