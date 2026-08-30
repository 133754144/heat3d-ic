# V7 G2-P2 local baseline freeze / protocol v3

日期：2026-08-31。机器可读合同为 `configs/heat3d_v7/g2_formal_preparation_protocol_v3.json`。本轮只在当前 Mac 和 G2 worktree 工作；没有 SSH/devbox、没有触碰 G1、没有 formal/长训练、没有 Therm-FM 24.10 GB archive、没有读取 `test_iid`/sealed。

## Frozen scope

正式 P1i common-task 只保留 **GINO、Transolver**；semiconductor-native 为 **DeepOHeat、DeepOHeat-v1**；Therm-FM 只在 transfer/data-efficiency track。Geo-FNO 与 DeepOHeat-v2 明确排除 formal engineering，只保留 related-work/capability comparison。

DeepOHeat 的 corrected receipt 固定 `2d_power_map=21×21×11=4,851`；single/multi HTC 仍各为 `51³=132,651`。三个 released checkpoint、参数量与本机 pretrained inference receipt 见 `docs/v7_g2_p2_deepoheat_corrected_receipt.json`。

DeepOHeat-v1 的 official paper/repo/code contract 见 `docs/v7_g2_p2_deepoheat_v1_audit.md`。它的 surface/volumetric physics tasks、separable outer-product operator、KAN trunk、full-mesh PDE/BC loss、Adam/schedule、normalization/eval、relative residual confidence、hybrid GMRES 与成本均已冻结。由于 release 没有 explicit license、data/checkpoint manifest、dependency lock 或 checkpoint，当前是 **D / formal semiconductor-native candidate**，不是 original reproduction PASS，也不能进入 P1i main table。

## GINO normalization 与 radius

正式 normalization 不再沿用 P1 smoke 的 per-node-index target statistics：

- coordinates：train-only per-axis bounds → unit cube；
- 11 physical features：跨全部 train samples 与 points 的 per-channel mean/std；constant channel scale=1；
- target：单输出 channel 的 train-global mean/std，跨 samples 与 points；
- valid/test 不拟合，point ordering 不参与，不增加信息。

CarCFD 的 geometry pipeline 对 SDF/area 做 train-range normalization，并用 `UnitGaussianNormalizer.from_dataset(dim=[1])` 处理其固定 release layout。P1i 有 11 个异质物理 channel，而且 formal contract 允许 variable point sets，因此不能把 CarCFD layout-specific statistics直接复制为 per-node-index target normalizer。该决定在 formal accuracy 前作出。

geometry-only radius receipt 见 `docs/v7_g2_p2_gino_radius_receipt.json`。在固定 4 train + 2 `valid_iid` 坐标样本上，upstream `r=0.033` 的 output GNO query coverage 为 100%，但 input GNO empty fraction 为 train 89.2845% / valid 89.0289%，median neighbors=0，判定 degenerate。根据 input-GNO nearest-source distance 的 train/valid p99=0.1310/0.1287，预先向上取 `r=0.15`：input coverage 达 99.7940%/99.8856%，median=9，output 仍 100%。没有查看 target、loss 或 accuracy。

`r=0.15` 是 **provisional formal radius**；HF 代理 503 阻断 768+128 全坐标同步，因此 formal seed0 前必须对完整 train/valid coordinate roles 重跑同一只读脚本，并做 GPU neighbor graph memory preflight。若完整 audit 改 radius，也只能按同一 geometry coverage rule，不能看 accuracy。

## Transolver frozen contract

Transolver 固定为 upstream Physics-Attention：8 layers、hidden 128、heads 8、slices 64、MLP ratio 1、dropout 0、official preprocess embedding、unified position off、ref 8；P1i 只调整 `space_dim/fun_dim/out_dim=3/11/1`。输入恰为 `coords + 11 physical features`，无 external prior、solver、dense-field side input 或 learned adapter。

参数量 716,737。loss 是 upstream decoded-output relative `TestLoss(size_average=false)`；AdamW lr 1e-3、weight decay 1e-5、clip 0.1；CosineAnnealingLR `T_max=500`；500 epochs、batch 1。formal seeds 固定 0/1/2，并绑定 Python/NumPy/Torch/model/dataloader RNG。checkpoint 保存 model、optimizer、scheduler、epoch、normalizers、config、upstream SHA 与 RNG metadata；P1 本机一轮 train→valid→checkpoint→reload 已 bitwise-equal PASS，本轮没有再训练。

正式 metric 输出必须先 decode 到 `deltaT_K`，再按 `v7_metric_contract.json` 产生全部 Level-A sufficient statistics 与精确命名指标；upstream relative L2 只列为诊断。远程先做单 batch/单 seed graph-memory 与 epoch-walltime preflight，经另行授权才排三 seeds。

## Common checkpoint policy

Heat3D G1、GINO、Transolver 的 primary 都按每个 model×seed run 独立选择：`valid_iid` 上最小 `sample_first_relative_rmse_pct`，tie 取 earliest epoch，每个 completed epoch 都是候选。禁止跨模型/seed/variant 选择，禁止 test/sealed，禁止看 formal result 后换 metric 或预算。

每个模型同时保存 upstream final-epoch checkpoint，并把其结果作为预注册 sensitivity。这样 primary 与 G1 已冻结合同一致，同时不丢失原作者 final-epoch semantics。

## Therm-FM 与 remaining gates

Therm-FM 保留 official tiny-demo PASS；model_T 是约 21M Poseidon/scOT，输入 `(N,P,L,H,W)` 按 layer-major flatten 到 `(N,L×P,H,W)`，输出 `(N,L_out,H,W)`，必须与 `config.json + pytorch_model.bin + normalization_constants.json` 同时使用。它仍是 `pretrained prior != same from-scratch training budget` 的 transfer track；本轮没有下载大 checkpoint/benchmark。

远程 G2 前仍需：完成 GINO 896-role coordinate-only audit 与 GPU resource gate；在完整 train role 物化并冻结 GINO/Transolver normalization statistics；生成 immutable formal launch manifests；等待 G1 结束并取得新的远程授权。DeepOHeat-v1 另需解决 license、official data/checkpoint 和 dependency provenance。
