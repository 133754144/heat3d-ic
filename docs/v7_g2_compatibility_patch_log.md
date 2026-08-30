# V7 G2 external baseline compatibility patch log

本日志只描述位于 `/tmp/heat3d-g2-*` 的临时 upstream checkout 与隔离依赖。第三方源码、数据、checkpoint、生成图和模型文件均未加入 Heat3D Git。

## DeepOHeat

- upstream：`Cadence-Celsius/DeepOHeat@46ccfe3fd43d99765e427480e7b6e0e16c3dbc70`。
- 官方 `prototype.py` 把 device 固定为 `cuda:3`，旧 checkpoint 在 PyTorch 2.9 需要显式 `weights_only=False`。
- Python 3.12 / NumPy 2.x 下，训练期 `gstools`/`pyDOE2` 依赖存在 `imp` 移除和二进制 ABI 问题。pretrained inference 实际只需要模型、power-map parser 和规则网格，因此临时 runner 直接复现 `grid_points_single_domain` 的官方 `51³` eval query mesh，绕过未用于推理结果的随机训练几何初始化。
- 模型结构、checkpoint、441 点 power-map 编码、温度反归一化 `293.15 + 25*u` 均未改变。10 个官方 showcase case 均成功。
- `single_htc_bc` 与 `multi_htc_bc` 的官方 eval import 会经过 `pyDOE2`（Python 3.12 已移除 `imp`）和旧 `gstools` binary（与 NumPy 2.x ABI 不兼容），即使 inference 本身并不使用随机训练采样。资格 runner 因此直接复现官方 `grid_points_single_domain` 的 `51³` deterministic mesh 与 low/middle/high beta，再加载原 checkpoint；两个实验各三例均成功，模型与数值语义未改。
- 隔离依赖尝试保持作者版本：`ordered-set==4.1.0`、`smt==1.3.0`、`gstools==1.4.1`。Python 3.12 resolver 安装了 NumPy 2.5.2/Scipy 1.18.1 transitive versions，导致上述旧 binary ABI 问题；未把这些包写入 Heat3D 环境。
- 未执行 full-semantics 1-epoch training：官方单 epoch 会拼接 20/50 个 parameter draws，并对完整 physics points 计算二阶导数；为控制本机运行且不改变 sampling，没有缩小该语义。

## Therm-FM

- upstream：`haiyangxin/Therm-FM@1c338d0fbe0dca25311eb896a9ea136a4f3d3cb1`。
- 使用 isolated `huggingface-hub==0.36.0` overlay 兼容主机已有 Transformers；未修改 Heat3D 或 Therm-FM `scOT/` 主路径。
- 官方 quick demo 的 3-epoch training 完成，但其同进程 post-training test 在 macOS 退出时触发 `Abort trap: 6` 与 leaked semaphore warning。把官方 `scOT/evaluate.py` 作为独立进程运行，并设 `dataloader_num_workers=0`、`OMP_NUM_THREADS=1`、`VECLIB_MAXIMUM_THREADS=1` 后完整通过并保存 metrics/predictions。
- 真实 steady checkpoint release 是一个 `24,100,784,430` byte 的单体 tar，混合所有 benchmark 与 T/B/L 三档；steady dataset 也是 `4,467,245,278` byte 单体 tar。为遵守“只取 model_T 和一个 benchmark”，没有下载整个归档，也没有用 B/L。单独发布的 HS_SC normalization constants 已下载并校验。

## GINO

- upstream：`neuraloperator/neuraloperator@00b7d86f8d74ff0af55da53eb585fe26df9c71f0`。
- isolated dependencies：`tensorly==0.9.0`、`tensorly-torch==0.5.0`、`opt-einsum==3.3.0`、`pytest==8.4.1`。
- 无源码 patch；官方 `neuralop/models/tests/test_gino.py` 为 `49 passed`。
- Heat3D P1i adapter 仍使用既有显式 latent-grid contract；没有把测试依赖加入 production environment。
- P1i protocol v2 使用 CarCFD 的 `32³` grid、radius 0.033、modes `16³`、4×64 FNO、Tucker rank 0.4。11-channel point function 在 upstream linear input integral 中要求 `fno_in_channels=11`；这是已有 channel dimension adjustment，不是新增 encoder。
- Mac 无 Open3D/torch-scatter，切换到上游提供的 pure-PyTorch neighbor search/reduction fallback；radius 与 input-mean/output-sum semantics 不变。
- PyTorch `weights_only=true` 拒绝 GINO state dict 携带的 `torch._C._nn.gelu` metadata。checkpoint 是 runner 在本地刚创建的，reload 改为显式 `weights_only=false`；不加载第三方不可信 checkpoint。
- 当前 frozen commit 的 `GINO_Small3d` config 仍包含 constructor 不接受的 `fno_domain_padding` 和旧 coordinate field；protocol v2 记录这一 upstream config/API drift，并只传 current official constructor 接受的等价字段。

## Transolver

- upstream：`thuml/Transolver@75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。
- 官方 Elasticity 数据只下载 `Random_UnitCell_XY_10.npy` 与 `Random_UnitCell_sigma_10.npy` 到 `/tmp`。
- 临时 patch 把脚本和 normalization/model tensors 的 `.cuda()` 改为 `.to(device)`，并增加 smoke-only `--ntest`；Physics-Attention、UnitTransformer、TestLoss 和 optimizer 路径未改。
- 运行缩为 16 train / 8 test、1 epoch、hidden 32、1 layer、4 heads、8 slices。缩小的是可行性资源预算，不是对正式架构的推荐值。
- 新 P1i gate 不复用上一条的缩小架构：严格使用 Elasticity shell 的 8 layers、hidden 128、8 heads、64 slices、batch 1、gradient clip 0.1、AdamW 与 `T_max=500` cosine schedule，仅把 `space_dim/fun_dim/out_dim` 调为 P1i 所需的 3/11/1。Mac CPU 一轮 train/valid/checkpoint/reload 通过。

## Geo-FNO

- upstream：`neuraloperator/Geo-FNO@d499bde15104f4ac34db6eaf57d3528a101e4ef6`，上游已明确 deprecated。
- 官方 Elasticity 数据只下载 `XY`、`sigma`、`rr` 三个数组到 `/tmp`。
- 临时 patch 把模型、IPHI 和 batch 的硬编码 CUDA 改为自动 device；同时把运行缩为 16 train / 8 test、1 epoch、4 Fourier modes、width 8、batch 2。
- learned deformation、nonuniform FFT/逆 FFT、FNO blocks、relative-Lp loss 都未改变。README 命令还需要显式 `PYTHONPATH=.` 才能解析根目录 `utilities3.py`。

## DeepOHeat-v2

- 无 compatibility patch。论文、arXiv metadata、GitHub repository 搜索和作者/机构关联仓库均未发现可验证 official implementation。
- 搜索到的第三方实验仓库不是作者仓库，未 clone、未引用为 original reproduction，也未启动 independent reimplementation。
