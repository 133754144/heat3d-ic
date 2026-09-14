# V7 G2-P12 CPU/document compatibility audit

本审计在 Heat3D seed0 devbox 训练运行期间于本地完成，只读取已有论文、上游 provenance、配置和 receipts；不启动 GPU，不读取任何 test/sealed/DeepOHeat official100，不下载 Therm-FM 大包。

| baseline | status | 最小下一步 |
|---|---|---|
| GINO | **FAIL-CLOSED** | 仅由 reviewer 解决 E2 fail-closed 与 E3 inherited qualification 的证据边界；不得改 radius、latent、reduction 或 tolerance |
| DeepOHeat-v1 | **NEEDS_AMENDMENT** | 保留 method-native cross-benchmark；冻结同 query-domain 的 evaluation-only 对齐，不声称相同输入/训练预算 |
| Therm-FM | **NEEDS_AMENDMENT** | 获批后才获取 model_T/单一 benchmark；保持 pretrained transfer track，先完成 grid/BC/material metadata 与泄漏审计 |

## 关键结论

- GINO 的 P1i `coords + 11 physical features` 可以原生表达，但当前正式 gate 仍为 fail-closed；pure-PyTorch fallback 不是 formal backend。
- DeepOHeat-v1 已完成独立 seed42 physics-informed training，但与 Heat3D 的 1024 sparse supervised contract 不是同一 information/training budget；只能作为 method-native semiconductor comparison。
- Therm-FM 官方接口是 Poseidon/scOT 网格张量，不能直接当作 P1i point-cloud baseline；约 24.10 GB checkpoint archive 和约 4.47 GB steady archive 仍未下载。
