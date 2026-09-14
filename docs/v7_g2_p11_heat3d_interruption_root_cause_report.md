# V7 G2-P11 Heat3D-on-DeepOHeat-v1 seed0 中断根因审计

审计范围：devbox 上 Heat3D-on-DeepOHeat-v1 seed0 的历史运行；只读检查
launch receipt、归档日志、progress/checkpoint、tmux/PID/reboot 记录及冻结的
runner/config/data provenance。没有读取 `test_iid`、sealed 或 DeepOHeat
official100 test，也没有启动新的 Heat3D 训练。

## 结论

epoch 11 这一份产物不是运行时崩溃，而是完成 epoch 11 后为
science-neutral execution-efficiency requalification 主动停止：
`STOPPED_FOR_ENGINEERING_EFFICIENCY_REQUALIFICATION`。没有 OOM、CUDA-XID、
NaN 或 traceback 证据，停止点位于完整 epoch 边界。

另有一个更早的 epoch 103 尝试在 devbox 主机重启时中断，分类为
`HOST_REBOOT_INTERRUPTION`；精确的进程退出码无法从保留 journal/tmp 日志恢复，
OOM 未被确认。两次事件不能混写成同一次 failure。

## 事件对账

| 事件 | evidence | 分类 | 是否可作 formal evidence |
|---|---|---|---|
| 2026-09-07 初始三路启动 | P8/P9 receipt；seed2 在无 progress 前 OOM，seed0/1 仅完成约 2 epoch 后按用户串行化指令停止 | 资源争用/用户主动停止 | 否；归档保留 |
| epoch 103 尝试 | devbox 于 2026-09-11 09:29 +0800 reboot；tmux 随 reboot 消失；无 kernel OOM/CUDA-XID 证据 | `HOST_REBOOT_INTERRUPTION` | 否；只保留至 epoch103 的历史归档 |
| epoch 11 串行尝试 | `stop_progress.json`；11 个完整 epoch 日志；PID/tmux 停止后不存在；停止原因明确写为效率资格化 | `EFFICIENCY_QUALIFICATION_STOP` | 否；不是 200-epoch formal run |

epoch 11 归档日志显示：首 epoch 含编译约 3019 s，随后每 epoch 约
1981--1999 s（约 33 min），与 E2 的 compiled-B24 约 85.95 h/200 epoch
runtime-only projection 一致。这支持效率停止的工程理由，但不构成 accuracy
结论。

## provenance 与 checkpoint

- epoch 11 attempt repo：`a59fde9c9a0edd2981b2a8be49f7e4036e533744`；runner
  `scripts/run_v7_g2_p6_heat3d_v1_formal.py` SHA
  `265337dc346d5e2e729c6361c005e64bb98cf2ff1f95b1ca2d859f5bbc1941d3`；config
  SHA `ee95f0e94a667703660eb5aa7229f3c7ed50efd0b696b9778cea979639fd21a5`；
  subset SHA `e719665176a22213487ee92c1aac993dd01b02a51555c7cd68bf81a13b861558`；
  label receipt SHA `a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd`；
  normalization SHA `3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db`。
- epoch 11 archive：`output/_engineering_archives/g2_e_seed0_epoch11_20260911`；
  total 10,737,762 bytes；archive receipt SHA
  `985962b082eee0e8e8225e48c8d57a5c3bb24f5e18aa334254aff2fd6ef16200`；best
  checkpoint SHA `a48926bf94e301b72ac3b5f092e370267a8cf7e65f7cb94944643cd30b1baec0`。
- 该 archive 只有 `params_best_sample_first.pkl`、progress/stop receipt 和日志，
  没有包含 optimizer state、scheduler state、RNG 和 atomic latest checkpoint。
- E2 的 bounded actual-model resume gate 曾在 2 个 optimizer updates 上通过，且
  provenance mismatch 会拒绝；该 gate 只证明 checkpoint 机制，不使 epoch 11
  的 qualification checkpoint 具备 exact-resume/formal 资格。

## resume 决策

历史 epoch 11 与 epoch 103 产物均**不允许直接 resume 为 formal evidence**：缺少
完整 optimizer/scheduler/RNG/batch-order state，且运行分别属于效率资格化停止和
主机重启中断。后续 Heat3D formal rerun 应从新鲜初始状态开始，先通过效率与
production exact-resume gate，再按冻结的 200 epoch/seed contract 运行；不得把
qualification-era best 参数当作正式起点或结果。

## Hard-boundary checks

- G1 未修改/重开；
- `test_iid`、sealed、DeepOHeat official100 test：均未访问；
- 没有因 valid accuracy 修改模型、loss、optimizer、batch 或 epoch；
- 本轮没有启动 Heat3D 新训练。

