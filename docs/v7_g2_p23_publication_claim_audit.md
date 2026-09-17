# P23 publication claim audit

## 当前可支持的 valid-only 表述

- 在冻结的 P1i native-1024 valid-only cohort 中，Heat3D V6 的记录误差低于现有
  GINO/Transolver 结果；该表述仅描述共同 native validation 域，不延伸为 sealed
  或普遍泛化优势。
- Therm-FM 已完成 valid-only 三 seed 的 pretrained transfer 训练；它使用
  Poseidon-T external PDE pretraining，不能称为 same-budget scratch baseline。
- 所有三 seed dispersion 均应称 **SD across training seeds**，不称 sample SD。

## 尚未支持/等待 P23 证据

- Heat3D 相对 Therm-FM 的 full-field accuracy、hotspot 或 peak advantage：统一
  evaluator、逐 case pairing 和 bootstrap 尚未完成。
- extreme BC、high source count 或强材料异质性的 robustness advantage：condition
  analysis 尚未执行。
- `statistically superior`、universal accuracy/efficiency superiority：没有预注册
  且完整的 sealed/generalization 证据，不得使用。

## 明确禁止

- 把不同 representation、预训练状态或 supervision modality 写成
  same-information-budget comparison。
- 在本轮把 3×128 行当作 384 个独立观测，或用 n=3 seed t-test 替代 paired case
  bootstrap。
- 将 P23 valid-only 结果写成 test/generalization 结论；sealed IID 仍不能解锁。

由于旧混合 CSV 的 accidental `test_iid` read，P23 readiness 必须保持
`P23_NEEDS_AMENDMENT`，而不能写成最终 ready。
