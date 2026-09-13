# V7 G2-E3 reproducibility and determinism qualification

本轮是 devbox 上的 bounded qualification，不是 formal/accuracy 运行。冻结的数据、split、模型、loss、optimizer、normalization、batch、radius 和 upstream 版本均未改变；所有历史 FAIL-CLOSED receipt 保留。

## 1. Reproducibility policy

`docs/v7_g2_e3_reproducibility_policy.json` 将原作者/upstream 实现语义设为 authoritative。GPU reduction 顺序造成的有限数值漂移不再以 bitwise parameter/optimizer hash 或 fallback `allclose(1e-5)` 作为 scientific gate。仍然 FAIL 的条件是 non-boundary graph disagreement、NaN/Inf、系统性 trajectory divergence、same-seed 噪声相对 inter-seed 不可忽略，或任何 scientific/provenance/backend mismatch。checkpoint load 必须在相同 backend contract 内恢复；resume 后允许正常 GPU numerical drift。

## 2. GINO upstream-semantic qualification

权威 backend 是 pinned upstream GINO 的 Open3D `FixedRadiusSearch` + `torch-scatter`；pure-PyTorch 仅为 diagnostic oracle。复用 896 个 train/valid_iid geometry audit：`non_boundary_disagreement=0`，历史边界-only disagreement=2057；P8/E2 等历史 receipt 未覆盖或删除。

六个 fresh child process（同 seed 20260907 三次、inter-seed 0/1/2 三次）均 `PASS_FINITE`，固定五步 trajectory，并在 child 内完成 checkpoint restore。same-seed/inter-seed 中位相对差及其比例如下：

| probe | same-seed median | inter-seed median | ratio |
|---|---:|---:|---:|
| loss | 3.1642e-6 | 3.92797e-2 | 8.0555e-5 |
| parameters | 2.0407e-4 | 1.41520 | 1.4420e-4 |
| prediction | 1.6770e-4 | 4.68387e-1 | 3.5804e-4 |
| updates | 1.1876e-2 | 1.41668 | 8.3828e-3 |

最大 ratio `0.0083828 < 0.10`，无 NaN/Inf 或 bounded probe 的系统性同 seed 漂移。因此 GINO 状态为 `GINO_AUTHOR_SEMANTICS_QUALIFIED`；这不是 formal training release。raw aggregate SHA256：`d6303129ca9308d62be255a066fa3e8ce357eb6fccdf5db523edbcacc691f4db`。

## 3. Heat3D deterministic flag audit

同一 frozen B24 fixture、fresh process、显式同步完整 `(params, optimizer_state, loss, gradients, updates, prediction)` 后计时：deterministic=true 的 seed0 重复 3 次，false 的 seed0 重复 3 次，false 的 seed1/2 各 1 次；每个 child 为 compile/first update + 5 个 post-warm steps。所有 8 个 child `PASS_FINITE`。

| flag | children | compile+first (s) | post-warm median step (s) | p95 of child medians (s) | median samples/s | max peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| true | 3 (seed0) | 73.822 / 43.264 / 43.022 | 39.7420 | 39.8352 | 0.6039 | 3,468,143,104 |
| false | 5 (seed0×3, seed1/2) | 22.436 / 3.308 / 3.302 / 16.368 / 17.316 | 0.03532 | 0.03654 | 689.25 (same-seed median) | 3,503,307,008 |

false 相对 true 的 synchronized post-warm speedup 为 `1125.29x`；compile/first 成本另列，不能当 steady-state。true 同 seed 的 loss/gradient/update/parameter/optimizer/prediction 相对差均为 0；false 同 seed/inter-seed 最大 ratio 为 `2.9771e-4`（prediction），远低于 0.10。true-vs-false 同初态五步的最大相对差为 loss `4.491e-6`、gradient `6.665e-5`、update `3.294e-4`、parameter `3.907e-6`、optimizer `3.016e-6`、prediction `7.590e-5`；全部有限，未观察到系统性 trajectory drift。结果只说明 deterministic reduction 有极大执行代价，不是 accuracy 证据。

因此生成但未应用 `HEAT3D_NONDETERMINISTIC_GPU_AMENDMENT_READY_FOR_APPROVAL`：建议将 `xla_gpu_deterministic_ops` 从 formal performance requirement 降为 debugging/reproducibility diagnostic option，仍须人工批准；formal config、200-epoch budget 和训练均未启动。raw aggregate SHA256：`e7a199b1f99ac1db6d2b95650329de6e9614a602b677577e6788ea69c87c7334`。

## 4. Transolver and common-gate scope

Transolver 的既有 actual-model execution/resume qualification 为 PASS；本轮不重复 preflight、不启动 formal。按照 scope amendment，GINO qualification 不再阻塞 Transolver 的独立资格，但两个模型的 formal launch 仍等待 versioned amendment 的人工审计/批准。common-task statistical hierarchy 未改变。

## 5. Boundary and release status

- GINO：`GINO_AUTHOR_SEMANTICS_QUALIFIED`；formal 未启动。
- Heat3D：`HEAT3D_NONDETERMINISTIC_GPU_AMENDMENT_READY_FOR_APPROVAL`；formal config 未改。
- Transolver：独立 execution/resume qualified；formal 未启动。
- G2 overall：`QUALIFICATION_COMPLETE_FORMAL_TRAINING_BLOCKED`。
- 本轮未修改/重开 G1，未访问 P1i `test_iid`/sealed 或 DeepOHeat official test，未启动 multi-HTC，未下载 Therm-FM，未启动任何长周期 formal training。

reviewer-facing machine-readable decisions：`docs/v7_g2_e3_gate_decision.json`；Heat3D amendment candidate：`docs/v7_g2_e3_heat3d_execution_amendment_candidate.json`。
