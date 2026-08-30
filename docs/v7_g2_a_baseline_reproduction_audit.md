# V7 G2-A 外部神经算子基线审计

本分支基于 G1 冻结提交 `78e7651bab5ef41a8ca4e42c45f64b1b98f04ea7`，只做外部基线的上游复线、输入适配和非发表资格验证。正式 G1 的代码、配置和输出不属于本分支的写入范围。

## 1. 研究边界

- G2-A 的入口只接收冻结 V6/P1i 的点坐标和 11 维 raw condition features；adapter 不读取 `target`、`temperature.npy` 或任何 `test_iid`/sealed label。
- 允许的输入角色是 `train` 和 `valid_iid`。准确率若要计算，必须由现有 `rigno.heat3d_runtime.evaluation.EvaluationCore` 在外部显式提供 truth。
- G2-A 的 smoke/qualification 都是 `execution_role=compatibility_audit`、`scientific_evidence_eligible=false`、`formal_g2=false`，不产生 publication headline、G1 结果或正式 G2 multi-seed 结果。
- 不运行 solver，不把外部方法的默认归一化、网格化、padding 或 latent-query 规则暗中嫁接到 V7 P1i。

## 2. 上游方法与冻结 provenance

精确的 commit、license SHA、论文和代码地址见 [G2 candidate registry](../configs/heat3d_v7/g2_candidate_registry.json) 与 [machine-readable receipt](v7_g2_a_baseline_reproduction_receipt.json)。当前审计固定了：

| 方法 | 上游输入/机制 | 当前状态 |
| --- | --- | --- |
| GINO | 输入点云/函数，经 input GNO 映射到规则 latent query，再经 latent FNO 和 output GNO 查询输出点 | 官方 `neuralop.models.GINO` 32 点 CPU forward smoke PASS；冻结 P1i 1024 点 input-only forward PASS；valid-only Level-A 接口 smoke PASS |
| Transolver | 点坐标与点特征，经 Physics-Attention 的 learned physical-state slices 建模 | 官方 irregular-mesh `Model` 32 点 CPU forward smoke PASS；冻结 P1i 1024 点 input-only forward PASS；valid-only Level-A 接口 smoke PASS |
| Geo-FNO | 学习物理域到规则 latent 网格的 deformation，再执行 FNO | 上游已声明 deprecated；本轮只登记，不把它作为可运行 G2 baseline |
| Therm-FM | Poseidon/scOT 的网格热场 foundation-model adaptation | 已审计输入/依赖；其网格张量和 checkpoint 语义不能直接当作 P1i point-cloud adapter |
| DeepOHeat | 3D-IC 配置编码与 DeepONet/physics-aware path | 已冻结官方 MIT snapshot；数据/checkpoint convention 尚未与 P1i 对齐 |
| DeepOHeat-v2 | 离散 physics loss、预条件优化和 solver-gated self-improvement | 目前只冻结论文记录，未找到可验证的官方代码 snapshot；不推测实现 |

官方 GINO 的论文是 [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)，Transolver 的论文是 [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)。热领域候选包括 [DeepOHeat](https://arxiv.org/abs/2302.12949)、[Therm-FM](https://arxiv.org/abs/2605.22663) 和 [DeepOHeat-v2](https://arxiv.org/abs/2608.16080)。

## 3. P1i adapter contract

`rigno.heat3d_g2.inputs.P1IInputBatch` 是唯一输入边界：

```text
raw V6/P1i condition features [B, 1024, 11]
        + coordinates [B, 1024, 3]
        -> external baseline adapter
        -> direct prediction [B, 1024, 1]
        -> (optional, separate) EvaluationCore with valid_iid truth
```

GINO 的 upstream batching 要求共享 input/output geometry，因此当前资格入口固定 `batch_size=1`；这不是对 V7 G1 B24 contract 的修改。GINO 的 regular latent grid 是 adapter 显式参数，当前 smoke 使用 `3^3`；Transolver 使用官方 irregular-mesh model，不引入 V7 Sparse KD-tree 或 reconstruction semantics。

## 4. 复线结果与已知限制

已执行的原作者实现复线是小型 CPU forward smoke，不是论文数值复现：

- GINO：官方 snapshot 能在显式 `feature_dim=11` 下运行，输出 `[1, 32, 1]`。
- Transolver：官方 irregular-mesh snapshot 能在 `space_dim=3, fun_dim=11` 下运行，输出 `[1, 32, 1]`。
- 本地环境初次导入 GINO 缺少 `tensorly`；在 `/tmp/v7_g2_env` 安装 pinned `tensorly==0.9.0`/`tensorly-torch==0.5.0` 后通过。该依赖环境不修改 V7 production environment。
- Therm-FM、DeepOHeat 和 DeepOHeat-v2 不因未完成 adapter 而伪称 PASS；它们保持 `deferred` 或 `paper-only` 状态。

本轮在 devbox 使用两个冻结 P1i 样本完成了非发表资格 smoke：`v6p1if1_0000`（train，仅输入）和 `v6p1if1_0003`（valid_iid，输入后由独立 EvaluationCore 读取 `deltaT` truth）。GINO 与 Transolver 均输出 `[1,1024,1]`。valid-only 运行还证明了 `rigno.heat3d_runtime.evaluation.EvaluationCore` 的接线；其数值只是未训练随机模型的接口诊断，不能解释为 accuracy 或 baseline 结果。

P1i GINO 使用 `/tmp/v7_g2_pydeps` 中的临时 `opt-einsum==3.3.0` overlay；NeuralOperator 的 `tensorly==0.9.0`/`tensorly-torch==0.5.0` 也只存在于隔离的复线环境。它们没有修改 V7 production environment。

当前 G2-A 只完成 adapter/input/evaluator interface qualification，不包含 few-step training、external benchmark accuracy 或 multi-seed 训练。任何正式训练或跨方法 accuracy 比较都必须另行注册，不能从本 receipt 推出。

## 5. 文件与执行边界

- 代码 adapter 不 import V7 的 `scripts`、smoke/development runner 或 private cross-script API。
- G2 branch 不修改 `rigno.heat3d_runtime`、V7 G1 manifest、V6 frozen artifacts、publication evidence。
- 临时 upstream checkout 位于 `/tmp/v7_g2_upstream`；不作为 V7 仓库依赖，不将大型外部数据/checkpoint 纳入 Git。
