# G2-E2 Heat3D runtime root-cause closeout

本文件只记录 devbox 上的 bounded runtime/profiling 证据；不构成 accuracy、收敛或 formal-training release。G1、P1i `test_iid`/sealed、DeepOHeat official test 均未访问。

## JIT/shape

当前 `V7FormalTrainer.step` 将 `value_and_grad`、gradient transform、optimizer update 和 `apply_updates` 放进一个 `jax.jit`。但 batch 被 closure 捕获，cache key 是 `batch_id`。768 个样本的 B24 训练有 32 个不同 batch signature，因此实际形成 32 个 train executables；p2r/r2p 有 31 个不同 padded edge shape，r2r 有 22 个。节点 shape 固定为 p-in/p-out `1025`、r `257`，变化主要来自图边 padded shape。三 epoch receipt 的 compiled-B24 compile phase 为 2207.40 s（36.79 min；control 为 2459.73 s/41.00 min）。精确 HLO module 数没有从当前 runner 单独导出，不能伪造为独立 HLO 计数。详见 `v7_g2_e2_heat3d_jit_shape_audit_receipt.json`。

## Bounded profiler

deterministic-XLA true 的一个静态 B24 fixture 完成了 1 个 post-warm step（56.161 s），trace 中 XLA GPU module/`GpuExecutable::ExecuteThunks` 约 50.357 s；`BUFFER_FLUSH` 约 2.059 s，`while.33+while.35` 约 2.576 s，`MemcpyD2D` 约 0.754 s。事件是嵌套的，不能相加；edge/node MLP、gather/scatter 在 XLA GPU module 内融合，XPlane protobuf 未被猜读。结合三 epoch compiled-B24 receipt 中 4756.69 s dispatch/device execution 对比 0.305 s explicit sync/host postprocess，主要类别判为 `SPARSE_GRAPH_KERNEL_BOUND_MIXED_WITH_SHAPE_JIT_BOUND`，不是 host-sync 主导。

## Deterministic A/B 与 projection

同一 fixture 的 deterministic=true/false bounded A/B 中，历史 deterministic=true 的 post-warm median 为 300.127 s，false 为 0.038605 s（约 7774×）；compile+first 为 349.191 s 对 23.740 s（约 14.7×）。该 300.127 s 数值现标记为 **superseded/non-representative outlier**，仅保留作 provenance；原因未知且不作推测。两边 finite，但 tree hash 不同，因此当时只保留为性能/语义 amendment 候选，不能改变 frozen deterministic 配置。正式 runtime baseline 仍采用已经完成的 compiled B24 三 epoch receipt：median epoch 1547.088 s，200 epoch 单 seed 约 **85.95 h**，三 seed 约 **257.85 h**；这只是 runtime projection。

## Host sync 与 cache

compiled B24 将 control 的 384 次 train scalar extraction 降到 192 次，explicit sync+host postprocess 从 74.735 s 降到 0.305 s，吞吐提高 8.42%；但 device dispatch/compute 仍占主导，历史 slim trajectory gate 已 fail-closed。JAX persistent compilation-cache 的 bounded probe 已完成：cold compile+first step 81.212 s，warm 47.068 s，阶段耗时降低 42.04%；cache inventory 两次均为 265 文件/4,985,287 bytes。该结果只证明 cold-start mitigation，不改变 steady-state projection 或科学协议。

## Gate

## 三模型 runtime projection（仅工程预算）

| 模型/路径 | bounded basis | 估计 3-seed walltime | 状态 |
|---|---|---:|---|
| Heat3D compiled B24 | 三 epoch B24 median epoch 1547.088 s，200 epoch × 3 | 257.85 h | `BLOCKED_RUNTIME_TARGET` |
| GINO optimized candidate | 既有单步 CUDA preflight（非 formal、未通过当前 equivalence） | 66.34 h | `FAIL_CLOSED` |
| Transolver frozen candidate | 既有单步 CUDA preflight（非 formal；模型级短程 gate 已通过） | 171.26 h | `BLOCKED_BY_GINO_COMMON_GATE` |

GINO/Transolver 的数字只是先前 bounded preflight 的粗略工程预算，不能与 Heat3D 的三 epoch projection 解读为同一测量协议，也不含 accuracy 或收敛结论。

- Heat3D：`BLOCKED_RUNTIME_TARGET`；compiled B24 是当前诊断 baseline，不释放长训练。
- GINO：896-sample geometry 当前复测为 boundary-only（`non_boundary_disagreement=0`），但 optimized output repeatability 未通过 frozen `1e-5`，故 `FAIL_CLOSED`；P8/E2 历史矛盾保留，不覆盖。
- Transolver：真实模型 hot-path trajectory 与 model-level exact-resume 均 PASS，但 common gate 仍被 GINO 阻塞，不能启动 formal。

下一步只有一条：在同一 devbox 保持所有 frozen science config 不变，完成 GINO optimized reduction/repeatability 的人工审计并收取 persistent-cache receipt；本轮不启动任何长周期 formal training。
