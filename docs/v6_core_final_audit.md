# V6 clean integration final audit

本审计分支从当前 `main@332ef3f` 新建。该 main 在本目标开始前已经包含
PR #2 的 V6 squash commit；本轮没有回滚、重写或再次 merge main。

现有 `integration/v6-core@82beed9` 与 `main@332ef3f` 的 Git tree 均为
`b08defef2c39910c7e52152039597ef482d77c19`，说明 squash merge 没有改变
allowlisted V6 内容。稳定迁移范围继续以
`v6_core_integration_manifest.json` 为唯一真源，冻结 V6 结果未修改。

治理术语保持：

- hard：已打开 corrected confirmatory holdout 内的预注册 IID stress subgroup；
- 16384：IID 平均完整场精度最高模式，不外推为逐样本最优或 OOD 保证；
- FVM：legal structured-FVM mesh sensitivity。

阶段索引覆盖 P1a–P1h、V6_01–V6_04、P1g→P1h、失败
volume-representative probe、Anchor-derived、稀疏 KD-tree/cache、
4096/8192/16384/32768 决策以及 holdout/hard 治理。

六个 main V5 checker、V6 core checker 和 canonical dry-run 均通过；
没有训练、模型推理或 test/hard 访问。本分支只新增审计 manifest、
checker 和本说明，明确 `new_merge_to_main_performed=false`。
