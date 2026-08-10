# V6/P1i P2 benchmark semantics

P1 accuracy 已冻结；本阶段只建立 workload 语义并补真实 known-support/new-physics timing。

| route | cold s | graph-rebuild s | known-support/new-physics s | replay s | known-physics speedup vs FVM |
|---|---:|---:|---:|---:|---:|
| B8192_recon | 10.3081 | 0.4418 | 0.0552 | 0.0027 | 30.53x |
| E32768_recon | 9.8509 | 0.5640 | 0.0809 | 0.0087 | 20.83x |
| B240825_direct | 13.6256 | 3.2197 | 0.2642 | 0.0862 | 6.38x |
| E240825_direct | 10.8995 | 1.5715 | 0.3929 | 0.0934 | 4.29x |

## Semantic gates

- `known_support_new_physics` 使用 32 个不同 group；每个样本读取真实 sidecar k/q 与原 BC、anchor context/scale。
- support、graph、reconstruction map 固定为第一个预注册 valid32 样本；32 个联合物理签名均唯一。
- `resident_runtime_graph_rebuild` 没有语义匹配的 FVM 状态，因此 speedup 为 N/A。
- temperature/metrics 不进入 P2 production timing；test/sealed 未访问。
