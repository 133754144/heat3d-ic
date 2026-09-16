# P22 Poseidon-T provenance / leakage audit

结论：`PASS_METADATA_LEVEL_NO_SAMPLE_OVERLAP_EVIDENCE`，允许进入真实 target qualification。

本审计只使用 Therm-FM 官方仓库 `haiyangxin/Therm-FM@1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1`、
Poseidon-T 官方 model card 与已冻结的 P21 资产 receipt。正式初始化明确是
`camlab-ethz/Poseidon-T@93adcbf10f75b45ac3bca3939cc3d3f239e1e663`，并使用
`replace_embedding_recovery`；没有使用 released thermal `model_T`，也没有下载 24 GB
Therm-FM checkpoint archive 或全量 thermal dataset。

官方公开 metadata/documentation 中没有发现与 V7 P1i 的 sample-level overlap 证据。该结论
是 metadata-level evidence，不是对未公开预训练样本的密码学非重叠证明，因此报告中必须保留
这一限定。Poseidon prior 属于 external PDE pretraining/foundation-model prior，不能写成
与 Heat3D 相同的 from-scratch training budget 或 same-information baseline。

P1i 仅采用 frozen train 768 / valid_iid 128；本审计未读取 `test_iid`、sealed 或
DeepOHeat official100。
