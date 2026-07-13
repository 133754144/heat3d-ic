# V5 Gate 5 Native shape–scale

## 分支语义

Native 分支先屏蔽 Dirichlet 节点，再做 control-volume 归一化：

```text
psi_free = (1 - m_D) * psi
phi_hat = psi_free / CVRMS(psi_free)
DeltaT_hat_free = s_hat * phi_hat
```

target 使用完全相同的 `m_D -> CVRMS` 分解。由此自由场满足
`CVRMS(phi_hat)=1` 与 `CVRMS(DeltaT_hat_free)=s_hat`；最后才在 raw
temperature 空间执行 Dirichlet projection。

`scale_only` 仅更新 `global_scale_*`；`shape_only` 仅更新
encoder/processor、shape decoder 与 local bypass；`joint` 更新两支。正式
N0/N1 均使用 `joint`，best 仍按最低 valid reconstructed normalized-DeltaT
MSE（`valid_base_mse`）选择，四项 native loss 权重保持现有 `1/1/1/1`。
单支模式同时对另一支 reconstruction value 使用 stop-gradient，并在 optimizer
更新前后应用参数组 mask；`scale_only` 不计 shape-only loss，`shape_only` 不计
log-scale-only loss，避免冻结分支通过共享重建路径被间接更新。

## B0、N0、N1

| label | config | field head | scale input | Global FiLM | 固定项 |
|---|---|---|---|---|---|
| B0 | `V4P5_04_local_bypass_global_film` | legacy normalized-DeltaT | none | on | clean split、local bypass、capacity、batch、optimizer、LR、600 epochs |
| N0 | `V4P5_05_native_physics_only` | native shape–scale | global physics | off | 同上 |
| N1 | `V4P5_06_native_pooled_latent` | native shape–scale | global physics + mean pooled `rnodes_processed` | off | 同上 |

N0 与 N1 的唯一变量是 scale head 是否读取 mean-pooled processed rnodes。
B0 是用户指定的正式基线；B0 到 N0/N1 同时改变 native 输出语义并关闭
FiLM，因此该比较只回答候选整体是否更好，不能单独归因于其中一个变化。

## 诊断定义

- oracle scale：`s_true * phi_hat`，隔离 shape error。
- oracle shape：`s_hat * phi_true`，隔离 scale error。
- physics scale：`s_phys * phi_hat`，显示 residual scale head 相对 physics proxy 的贡献。
- scale error：`mean(abs(log(s_hat)-log(s_true)))`。
- shape error：`mean(CVRMS(phi_hat-phi_true))`。
- joint 与三个诊断场均报告 relative RMSE；joint 额外报告 amplitude ratio、
  CV-weighted spatial correlation、target hotspot RMSE 与 target top-5 RMSE。
- gradient smoke 对 shape CV、log-scale、relative field、raw field 四项 loss，
  分别报告 backbone、shape decoder、scale head gradient norm。

## N2/N3 决策

- 只有 N0/N1 至少一个在 valid reconstructed MSE、relative RMSE 和 raw/hotspot
  指标上稳定优于 B0，才准备下一轮；hard roles 仍只报告。
- 若 N1 明显优于 N0，优先 N3：在 N1 上恢复 identity Global FiLM，以检验
  pooled latent 与 FiLM 是否互补。
- 若 N0 不差于 N1，优先 N2：在 N0 上恢复 identity Global FiLM，保持较简单
  的 physics-only scale head。
- 若二者均无稳定收益，保留负结果，不通过追加 FiLM 掩盖 native 分支问题；先用
  oracle scale/oracle shape 与分 loss 梯度定位 scale 或 shape 瓶颈。

本 Gate 只实现、检查并登记配置，不启动 N0/N1 训练。
