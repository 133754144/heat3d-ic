# P22 Therm-FM seed1 attempt-1 failure

2026-09-17 02:21（devbox）seed1 在 runner 的 required-path preflight 阶段失败：启动命令的
`--sample-root` 漏写了仓库 `data/` 目录。错误为 `FileNotFoundError`，发生在模型初始化、数据读取
和训练之前；seed1 output 目录没有创建，GPU 保持空闲。原始日志
`logs/g2_p22_thermfm_seed1.log` 保留，不覆盖。

这是工程路径错误，不是 scientific 或 numerical failure。按同一 seed、upstream/runner/config/
dataset/normalization 合同使用独立 `retry1` 日志重试；不得把该失败尝试与正式结果混合。
