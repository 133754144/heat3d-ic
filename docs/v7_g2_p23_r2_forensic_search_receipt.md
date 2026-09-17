# V7-G2 P23-R2 forensic checkpoint recovery

状态：**`FORENSIC_RECOVERY_EXHAUSTED_CHECKPOINT_NOT_RECOVERED`**

本 receipt 记录的是 devbox 上的只读取证搜索，不是新的训练或评估。搜索在
`research/v7-g2-baselines` 的远端工作树 commit
`6388db39136c614b39f2c5693de7b581bb59cbc4` 上完成。

## 目标与判定

| seed | frozen run | 目标 SHA256 | canonical 文件 | 结果 |
|---|---|---|---|---|
| 1 | `V6_07_V5best_P1i_seed1_reliable_B24` | `7197157969278d99648ef9b40d74005d759f32e52e9282e72a5586003d1e71f7` | `.../V6_07.../params_best_valid_point_global.pkl` | `NOT_FOUND` |
| 2 | `V6_08_V5best_P1i_seed2_reliable_B24` | `d67e0dac2e8ed8009ce7dcdf0b2de4543b10bc005c0bfaa51027ea721bb2ab49` | `.../V6_08.../params_best_valid_point_global.pkl` | `NOT_FOUND` |

身份判定只接受完整 SHA256。由于两枚文件均不存在，没有创建 recovery copy，
也没有对其他模型参数做替代性哈希匹配。

## 搜索范围

按“文件名/路径先缩小，候选再核 SHA”的顺序检查了：

- devbox 上的 `heat3d-ic`、`heat3d-ic-g2`、`heat3d-ic-gate6g`、工作副本及其
  Git worktree；
- V6 P1i runs、smoke、archives、supplemental publication、cross-resolution、
  Anchor/U-v2 目录；
- `/home/xyh/.cache`、`/tmp`、`/var/tmp`、`/mnt/wslg/distro/home/xyh`、
  `/mnt/c/Users` 以及几个用户级同步/模型目录；
- JSON/MD/log/txt/command receipt 中的 `V6_07`、`V6_08`、目标 SHA 前缀、
  `params_best_valid_point_global`、`U_v2`/`U-v2`；
- Git full-history 和 LFS 文件名/对象目录。

没有对整个 home 做无差别 SHA 扫描；只对明确的 V6 seed0 reference candidate
做了实际 SHA 校验。

## 关键证据

1. canonical V6 输出树只有 seed0 参数文件：
   `V6_06.../params_best_valid_point_global.pkl`，大小 `10,792,948` bytes，
   SHA256 为 `51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`。
   `V6_07` 和 `V6_08` canonical 目录/文件均不存在。
2. V6 P1i archives 只有 `V6_05` 的失败 legacy export；V6 smoke 只有 seed0。
3. U-v2 S1/S2/S3 的 direct-240825 与 16384 receipts 的唯一
   `checkpoint_sha256` 都是 seed0 的 `51567afe…b90e`。这些 receipts 只能用于
   provenance 追踪，不能把 U-v2 prediction 当作 seed1/seed2 的 canonical P23
   prediction。
4. 现存 V6 P1i log 是 seed0 log；其中对 seed1/2 的记录是 planned WSL2
   replication/configuration audit，不是已恢复的 checkpoint 或 output。
5. Git history 只找到 V6_07/V6_08 的 YAML 配置提交，没有 checkpoint binary。

## Fail-closed 结果

- 不训练、不重训、不替换 checkpoint、不重选 checkpoint；
- 不修改 evaluator/protocol；
- 不读取新的 `test_iid`、sealed IID 或 DeepOHeat official100；
- 不运行 seed1/2 replay，因此不生成六档 common-evaluator archive、paired
  bootstrap、condition analysis、Table B 或 winner/superiority claim；
- 既有 P23-R partial/blocked receipts 保持不变。

下一步只能从原始 WSL2/source storage 或独立不可变 archive 恢复两枚**完整目标
SHA256** 文件。恢复前不得重训或设计降级方案。
