# GINO E3 上游复现语义合同

审计日期：2026-09-15。本文只把原作者论文、官方
`neuraloperator/neuraloperator` 仓库及其 pinned checkout 作为权威来源；本地
V7 adapter 的观察不写成作者承诺。

## 权威来源与版本

- 论文：Zongyi Li et al., *Geometry-Informed Neural Operator for Large-Scale
  3D PDEs*, NeurIPS 2023，arXiv:2309.00583。
- 官方仓库：`https://github.com/neuraloperator/neuraloperator`。
- pinned commit：`00b7d86f8d74ff0af55da53eb585fe26df9c71f0`；该 checkout 的
  `git describe` 为 `2.0.0-65-g00b7d86`，没有指向该 commit 的 release tag。
- 许可证：MIT（官方 `LICENSE`；manifest 中记录的 SHA256 为
  `5f335b0603acfb1d180bf3117492e2076562075cc9b6353a3b94d5d5bc03ef33`）。
- 关键源码：`neuralop/models/gino.py`、`neuralop/layers/gno_block.py`、
  `neuralop/training/torch_setup.py`、`neuralop/training/trainer.py`、
  `neuralop/training/training_state.py`、`config/gino_carcfd_config.py`、
  `scripts/train_gino_carcfd.py`。

## 官方训练/模型语义

官方 CarCFD recipe 使用 `GINO_Small3d`：3D GNO 输入/输出图、规则 latent
grid、FNO modes `16^3`、4 个 FNO layers、hidden 64、instance norm、channel
MLP、Tucker rank 0.4；参数量约 13,673,988。输入 GNO 使用 mean reduction，
输出 GNO 使用 sum reduction。模型构造器的 `gno_use_open3d=True` 和
`gno_use_torch_scatter=True` 默认启用 Open3D `FixedRadiusSearch` 与
`torch_scatter` grouped reduction；关闭时才使用仓库内的 PyTorch/`segment_csr`
fallback。CarCFD config 暴露的默认 radius 为 0.033。

`scripts/train_gino_carcfd.py` 的官方路径是：加载
`CarCFDDataset`，训练 loader `batch_size=1, shuffle=True`，测试 loader
`batch_size=1, shuffle=False`，构造 `get_model(config)`，使用 AdamW
`lr=1e-3, weight_decay=1e-4`、StepLR(`step_size=50, gamma=0.5`)、301 epochs，
训练/测试损失均为 `LpLoss(d=2,p=2)` 的 relative L2。训练脚本调用
`Trainer.train(...)`，每个 eval interval 运行 test loader；该脚本没有传入
`save_best` 或 `save_every`，因此 V7 的 valid-best 规则是本地预注册合同，不是
官方脚本的选择承诺。官方模型的输出是 CarCFD pressure field，不是 V7 温度场。

V7 common-task adapter 只调整官方已有的 channel dimension（11 个物理特征、
1 个输出），并冻结 `r_in=0.15`、`r_out=0.033`、`32^3` latent grid；这属于
V7 数据接口合同，不应倒写成官方 CarCFD 的默认值，也没有加入 learned encoder、
solver、dense field 或额外 prior。

## seed、determinism 与 checkpoint

官方 `DistributedConfig.seed` 默认是 `None`。如果显式提供 seed，
`training/torch_setup.py` 会调用 `torch.manual_seed`、`torch.cuda.manual_seed`
（分布式时加 data-parallel rank），并设置 `torch.backends.cudnn.benchmark=True`。
在 pinned checkout 中没有找到 `torch.use_deterministic_algorithms(True)`、
`cudnn.deterministic=True`、`CUBLAS_WORKSPACE_CONFIG` 或其他强制 bitwise 的
设置；README、config、training script 也没有给出 bitwise 或跨进程完全相同的
承诺。

官方 `Trainer`/`training_state` 支持保存和恢复 model state、optimizer state、
scheduler state、regularizer state，以及 `manifest.pt` 中的 epoch。模型本身还
保存初始化 kwargs/version metadata。但官方 checkpoint 没有保存 RNG state、
数据 permutation/batch order、runner/config/data hash，也没有保证 process-level
exact resume。因此这里记录为
`ENGINEERING_LIMITATION_NOT_E3_FAILURE`，不能作为排除 GINO 的科学理由。

官方仓库没有多 seed variance 表或明确的跨进程 reproducibility receipt；这类
数值/资源观察属于本次资格化记录，不属于作者声明。

## E3 判定边界

本阶段 authoritative backend 是 pinned Open3D `FixedRadiusSearch` +
`torch-scatter`。纯 PyTorch 路径仅为 diagnostic oracle。E3 只要求：官方
operator semantics 与冻结 V7 adapter 保持一致，forward/loss/gradient/update
finite，无系统性 NaN/Inf 或 trajectory divergence，checkpoint load 恢复已保存
state，且没有违反作者明确声明。GPU reduction 顺序、Open3D kernel 的 process-level
浮点漂移，以及没有被作者承诺消除的 CUDA nondeterminism，只记录为 diagnostic，
不自创 tolerance。

本地执行环境为独立 `g2-gino-open3d-src`：Python 3.12.12、PyTorch
2.9.0+cu130、CUDA runtime 13.0、Open3D 0.19.0、torch-scatter 2.1.2+pt29cu130、
GPU NVIDIA GeForce RTX 5070。官方 `pyproject.toml` 只给出未锁定的基础依赖，
因此上述版本是 reconstructed qualification environment，而非作者官方 lock。

