# V7-G2 final evidence lock

状态：`G2_DEVELOPMENT_COMPLETE_VALID_ONLY_EVIDENCE_CLOSED`
P24：`READY_FOR_REVIEW_NOT_UNLOCKED`

本锁只覆盖 P1i `valid_iid=128`。本轮没有训练、调参、checkpoint
reselection，也没有读取新的 `test_iid`、sealed IID 或 DeepOHeat
official100。

## Frozen identities and domains

- Table A 是 V7 P1i-e200 三 seed family（train768/valid128）与 GINO、
  Transolver 的 native1024 对比。
- Table B 是同一 valid128、240,825-node full field 上的 Heat3D V7
  P1i-e200 + `U_v2_direct240825` 与 Therm-FM。
- `Heat3D-on-DeepOHeat-v1 e200`、V6 canonical、V7 e600 不属于本锁的
  Heat3D reference。
- 每个 Heat3D seed 的 config、checkpoint、native prediction 和 direct
  sidecar SHA，以及 P1i dataset/split/truth SHA，都在
  `docs/v7_g2_final_evidence_freeze.json` 中逐 seed 绑定。
- U-v2 sidecar 的绑定来自 immutable direct-route receipt（包含 checkpoint
  SHA、route、valid manifest 与 truth archive），不是文件名或历史描述推断。
  它是 `POSTHOC_IMMUTABLE_RECEIPT_BOUND_NOT_REINFERRED`；原有
  checkpoint→P1i replay `FAIL_CLOSED` 证据仍保留，不能把 sidecar 身份
  证据改写为本阶段重新推理。

## Metric definition audit

V4 `rigno/heat3d_v2_field_shape_diagnostics.py`（SHA
`cdcf1bf2…93798734`）定义：

- `Corr_point (%) = 100 ×` unweighted centered spatial correlation；
- `Amp_range (%) = 100 × (max(pred)-min(pred))/(max(true)-min(true))`；
- legacy `top_k_overlap` 保留为 top-5 diagnostic。

V5 `rigno/heat3d_v5_metrics.py` 与 `heat3d_v5_shape_scale.py`（SHA 见
shape-definition receipt）定义：

- `Corr_CV (%) = 100 ×` control-volume-weighted centered correlation；
- `Amp_CVRMS (%) = 100 × CV_RMS(pred)/CV_RMS(true)`；
- `scale_log_error = log(CV_RMS(pred)/CV_RMS(true))`。

最终 evaluator 的名称全部带 V4/V5 语义限定，不再使用裸 `Corr` 或
`Amp`。幅度校准误差按样本先算 `|Amp_range-1|`、`|Amp_CVRMS-1|`，再在
seed 内聚合；不能用 `|mean(Amp)-1|` 替代。true-hotspot 固定为每样本
真实温度最高的 1%（2,409 nodes），prediction 不参与定义。

## Statistical amendment

旧 P7/P23 receipts 保留。当前 bootstrap 是事后统计修正，不是
preregistered contract：每次以 physical case 有放回重采样，分别对三个
training seed 重算与主表相同的完整 estimator，再对 seed-level estimates
取均值。它与 training-seed SD、旧 hierarchical seed+case bootstrap、以及
prediction-ensemble estimand 明确分开；point estimate 与主表一致。

Heat3D − Therm-FM 的 10,000 次、seed=230023 paired percentile CI 见
`docs/results/v7_g2_final_bootstrap_amendment.json`。区间条件化于冻结的
三 seed cohort，不自动转写为普遍或 statistical superiority claim。

## Claim boundary and remaining evidence

支持的表述是：在冻结 valid-only、明确评价域内，可以报告 native1024 与
full-field240825 的描述性结果，并报告字段形状、幅度校准和 hotspot
diagnostics。Therm-FM 只能称 external-pretrained dense-input capability
reference；不得称 same-information-budget baseline。禁止 universal
superiority、universal efficiency、same-information-budget superiority 或
sealed/test generalization claim。

已知的 fail-closed 证据（d9d961a checkpoint replay、旧 P23 incident、旧
sidecar manifest 未枚举）均保留在 manifest/历史 receipts 中，不是静默
覆盖。负向 domain-gate test 对 sample-ID、coordinate-order、truth-row
mapping 的故意破坏均按预期 fail closed。

## Freeze recommendation

没有新的 implementation/scientific blocker；建议停止 G2 新训练与调参，
冻结 checkpoint、sidecar、dataset/split、evaluator、metric contract、
bootstrap contract 和 claim set。下一步只能由人工审阅后，另行建立
evaluation-only sealed preregistration；本提交不解锁 sealed。
