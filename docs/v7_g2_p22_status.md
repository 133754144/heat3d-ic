# V7 G2 P22 状态

P22 已完成 native 65×65 adapter、train-only normalization、Poseidon provenance audit 与真实
target bounded qualification。Therm-FM 是独立的 `PRETRAINED_TRANSFER_FOUNDATION_MODEL_BASELINE`；
HCP 保持 `REFERENCE_ONLY / OFFICIAL_ACCURACY_ASSETS_BLOCKED`，不阻塞 Therm-FM。

Therm-FM 使用 Therm-FM `1c338d0…` 与 Poseidon-T pinned revision `93adcbf…`。输入严格为
case-definition 直接栅格化的 layer-major 741 channels（57 layers × 13 channels），物理域
`65×65`，输出 57 channels；取消外部 128 padding，ScOT 内部 patch/window padding 与 recovery
crop 保持 upstream 语义。train-only 统计 SHA 为 `76acbd1a…44eec`，split 是 train 768 / valid 128；
没有读取 test_iid、sealed 或 DeepOHeat official100。

T4 使用真实 `deltaT_K` target，通过 native forward/backward、有限 loss、240,825-node
denormalized evaluator 和 checkpoint model/optimizer reload；3 steps loss 为
`1.1986 → 1.0873 → 0.9115`，状态 `PASS_T4_REAL_TARGET_NATIVE65`。该值仅是 pipeline
qualification，不是 accuracy claim。

在 T4 PASS 后，seed0 已按冻结 200-epoch / batch40 / AdamW / cosine / valid normalized p=2
选模合同在 devbox `PDEFormer` 中完成；完整性审计见
`docs/v7_g2_p22_thermfm_seed0_audit.json`。seed1 的首个尝试因命令漏写 `data/` 在 required-path
检查阶段失败（未访问数据），失败 receipt 与原日志保留；retry1 已使用绝对且已预检的正确路径
运行，目前约在 epoch 35/200。seed2 不并行启动，由独立 gate 等待 seed1 的
`COMPLETE_VALID_ONLY` receipt 后再执行绝对路径、空目录检查并串行启动。

当前尚无三 seed aggregation；在 seed1/seed2 完成并分别审计后才生成 valid-only 汇总。整个 cohort
继续禁止读取 `test_iid`、sealed 与 DeepOHeat official100。

HCP 审计确认官方 train/eval geometry 内联且仓库不含 checkpoint/raw accuracy bundle；native
smoke 可运行但不能宣称论文 accuracy，也不能原生接入 P1i variable geometry/material/source/BC。

机器可读状态：`docs/v7_g2_p22_status.json`。
