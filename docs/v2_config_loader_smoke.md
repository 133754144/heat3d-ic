# Heat3D v2 config loader smoke

## 本轮目标

本轮只完成 P0：config loader smoke。目标是新增一个只读 YAML loader，让 v2 配置草案可以被机器读取、校验和简要汇总，但不接入任何训练流程。

## loader 边界

`rigno/heat3d_v2_config.py` 只使用标准库和 PyYAML，读取并校验 YAML 字段，不导入 JAX、Flax、Optax、runner、model 或 dataset loader。

它不会读取数据集内容，不创建 `output/`，不修改 YAML，也不会触发训练。

## 解决的问题

v1 runner 中很多状态分散在 CLI 参数、脚本默认值和人工命名约定里。P0 loader 先把这些状态变成可校验的结构化输入，重点解决：

- CLI 参数分散，难以复现实验配置；
- model、optimizer、loss、run、export、diagnostics 默认值隐含；
- frozen V1 baseline reference 之前主要存在于文档中，不够机器可读；
- controlled config 对 reference 文件路径缺少自动检查。

## 当前支持的配置

当前 smoke 脚本读取：

- `configs/heat3d_v2/smoke_minimal.yaml`
- `configs/heat3d_v2/medium1024_gapA_controlled.yaml`
- `configs/heat3d_v2/frozen_v1_reference.yaml`

其中 controlled config 会解析 `baseline_reference.path`，并确认 frozen reference 文件可读。

## 当前不做

本轮不替换 optimizer，不接 Optax AdamW，不修改模型，不新增 loss，不改 v1 runner 行为，不跑训练，不生成数据，也不生成训练 output。

## 下一步建议

下一步适合进入 P1：实现 config-to-command dry-run / v1 runner wrapper，只把 v2 config 转成可检查的命令草案，不实际运行训练。
