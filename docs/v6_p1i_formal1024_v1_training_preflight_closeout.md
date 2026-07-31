# V6-P1i formal1024_v1 训练前审计收口

## 判定

`heat3d_v6_p1i_continuous_physics1024_v1` 的训练前审计通过，判定为：

`authorized_for_training_with_frozen_applicability_boundaries`

本轮没有重新生成、筛选或替换样本，没有训练，也没有模型推理。授权只
适用于冻结的 P1i-v1 耦合输入分布；不是 strict-uniform、独立
power–BC、finite-contact 或 OOD 保证。

## 主要证据

| 指标 | 范围 | median | 12-bin entropy | bin CV | KS vs observed-range uniform | W1/range | max gap/range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| peak ΔT (K) | 30.3740–173.0984 | 90.7767 | 0.9515 | 0.4413 | 0.1548 | 0.0711 | 0.0286 |
| mean ΔT (K) | 22.0084–146.5273 | 72.6429 | 0.9311 | 0.5195 | 0.2059 | 0.0950 | 0.0432 |
| CV-RMS ΔT (K) | 22.4866–146.7174 | 73.0493 | 0.9337 | 0.5091 | 0.2002 | 0.0931 | 0.0435 |
| Reff peak (K/W) | 6.9922–21.9633 | 12.3489 | 0.9502 | 0.4581 | 0.1714 | 0.1077 | 0.0331 |
| top heat fraction | 0.8535–0.9883 | 0.9532 | 0.8691 | 0.7619 | 0.3447 | 0.2025 | 0.0556 |
| bottom heat fraction | 0.0117–0.1465 | 0.0468 | 0.8691 | 0.7619 | 0.3447 | 0.2025 | 0.0556 |

peak、mean、CV-RMS 和 Reff 的 12 个等宽 bin 全部非空、entropy 均大于
0.90、最大间隙均小于观测范围的 5%，支持 continuous broad coverage。
分箱计数、KS/Wasserstein 和 KDE 肩峰同时表明其不是严格 uniform。

## 已冻结限制

- power–top_h Spearman 为 `0.533374`；
- power–top_h 的物理 6×6 occupancy 有 `11` 个空 cell；
- high-power/low-top_h 角点样本数为 `0`；
- train 对 valid/test 的最大描述性单变量 KS 为 `0.127604`，最大
  range-normalized Wasserstein 为 `0.038394`；test 未用于规则调整；
- P1h、P1i、V6 random-block 均为 perfect contact，`R_contact=0`，无法
  学习有限接触热阻变化。

## 可复现与归档

- detached clean-checkout replay commit：
  `3f3e9e3974182dd340501b5e41e64824591b2aae`；
- 重放比较工件：`14`；
- SHA 不一致工件：`0`；
- 外部仓库：`133754144X/heat3d-thermal-simulation`；
- 冻结 tag：`p1i-formal1024-v1-27d2ea3b`；
- 外部 commit：`7b3af69e2164ad06d1c079fbde4d6cbd50183c9a`；
- 路径：
  `subsets/heat3d_v6_p1i_continuous_physics1024_v1/`；
- 已核验 sample files：`9216/9216`；
- 已核验 metadata files：`3/3`；
- missing/extra/SHA mismatch：`0/0/0`。

旧 `check_heat3d_v6_p1i_formal1024.py` 仍绑定 formal1024_v0 生成时的整份
attempts CSV；v1 的 append-only lifecycle 行会使其旧 artifact SHA
检查失败。该限制不通过改写 v0 checker 或 v0 工件来规避。本轮使用
`check_heat3d_v6_p1i_requalification.py` 验证不可变 v0 历史前缀、v1
冻结哈希和 1024 个样本，并由新的 training-preflight checker 复核本轮
工件。

## 工件索引

- 完整审计：
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_training_preflight_audit.json`
- 训练授权：
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_training_authorization.json`
- 外部归档：
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_archive_manifest.json`
- clean replay：
  `configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_clean_checkout_replay.json`
- checker：
  `scripts/check_heat3d_v6_p1i_training_preflight.py`

下一研究问题应限定为：在不查看 holdout 结果的前提下，训练模型对
power–top_h 缺失角点和 finite `R_contact` 的适用范围应如何通过独立
OOD 数据版本验证。本轮不创建该数据版本。
