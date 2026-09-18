# V7 validation visualization reproduction

本次重新加载 6 个 checkpoint，在 devbox RTX 5070 上进行推理；没有训练、solver 或 test/sealed 读取。

## 样本与展示口径

- P1i：`v6p1if1_0003`，`valid_iid` 首行，seed 0。Heat3D G1 Full 与 canonical vanilla RIGNO；GINO/Transolver 为 G2 正式 best。
- DeepOHeat-v1：`dhv1_volume_valid_00084`，source index 84，seed 0；Heat3D 为固定 e600，DeepOHeat 为 matched-768 validation-best。
- P1i：1024 原生点，同层 167 点 thin-plate RBF（smoothing=0）展示为 256×256；不是 65536 点推理，也不是密集 FVM 真值。37.3% 展示网格在支撑凸包外，修正版已统一屏蔽为灰色，不再显示外推温度。真实值和预测值使用相同插值。虚线框来自 sample_meta.json 中该层真实 q_blocks。
- DeepOHeat：571256 点完整直接输出，精确提取 101×101 的 z=0.12 切片，无场值平滑。
- 整组真实/预测统一色谱与范围，每个面板分别附色条；同组 reference 的数据和 RGBA 像素映射必须完全一致。误差共用零中心对称范围。无误差裁剪。
- 图中 relative L2 是整个原生/全场样本的非 CV 加权相对 L2，不是切片误差或多样本平均。单例不能替代 benchmark 排名。
- 新的 G1 forward 与归档 native 预测最大差异：Full 0.0098724 K、RIGNO 0.0303192 K；因此不声称 bitwise replay。checkpoint SHA256 完全一致。

## 文件

本地目录：`research_artifacts/v7_validation_figures/`，PNG/PDF、NPZ、每模型 JSON、两份组图 manifest、源元数据与脚本快照均在此。图与数组不入 Git。
每组图 manifest 包含 checkpoint 全路径及 SHA、样本与归一化身份、范围、切片和推理命令。

## 推理复现

执行代码仓库：`/home/xyh/myCodeGitOnly/heat3d-ic-g2`，commit `d9d961aa6970f60a45dea7995f1b82dab2e335d8`。
将 `scripts/export_v7_visualization_inference.py` 放到远端 `/tmp/`，设置 `HEAT3D_REPO` 指向该仓库。下面命令中的 output 必须改成未存在的目录（脚本拒绝覆盖）。
G1 的 `/tmp/v7_vis_*` 来自持久化归档：`/Users/xuyihua/.codex/worktrees/cbb8/3D IC Heat/research_artifacts/v7_g1_formal_archive/formal_21_runs/{Full,vanilla_RIGNO}_seed0/params_best_sample_first.pkl`；逐文件校验见 verification.json。

### Full

Checkpoint SHA256: `7ee050c946513004c3e33ce3b28c6fd246965ee4e95ff661eca3d888d5786517`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model Full --checkpoint /tmp/v7_vis_Full_seed0.pkl --p1i-root /home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1 --output /tmp/v7_visualization_20260918/Full.npz
```

### RIGNO

Checkpoint SHA256: `f44895af0a6da9b6a2f1ddf5d72cda156f86107895ac8a589c2e162b11c88e57`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model vanilla_RIGNO --checkpoint /tmp/v7_vis_RIGNO_seed0.pkl --p1i-root /home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1 --output /tmp/v7_visualization_20260918/RIGNO.npz
```

### GINO

Checkpoint SHA256: `9ae1485065a14c86cc28cfe1d279faddfbeda701b51ef418cf2b0461b83c55e8`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
conda activate g2-gino-open3d-src
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model GINO --checkpoint /home/xyh/myCodeGitOnly/heat3d-ic-g2/output/heat3d_v7_g2/gino_p13_e3/seed_0/best_valid_iid.pt --p1i-root /home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1 --upstream /home/xyh/myCodeGitOnly/external/g2/neuraloperator --output /tmp/v7_visualization_20260918/GINO.npz
```

### Transolver

Checkpoint SHA256: `f70ad7b897b2e1831a0f06b9300907ceb904a4d646cd7d7a8b8f949bdd0256eb`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
conda activate g2-baselines
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model Transolver --checkpoint /home/xyh/myCodeGitOnly/heat3d-ic-g2/output/heat3d_v7_g2/transolver/seed_0/best_valid_iid.pt --p1i-root /home/xyh/myCodeGitOnly/heat3d-ic/data/heat3d_v6_p1i_continuous_physics1024_v1 --upstream /home/xyh/myCodeGitOnly/external/g2/Transolver --output /tmp/v7_visualization_20260918/Transolver.npz
```

### Heat3D-DeepOHeat

Checkpoint SHA256: `6e4dc91188261832923f6bda13b871a666a77c41770da365bab88402416e8498`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model Heat3D-DeepOHeat --checkpoint /home/xyh/myCodeGitOnly/heat3d-ic-g2/output/heat3d_v7_g2/heat3d_on_deepoheat_v1_e600_p15/seed_0/epoch_0600.pkl --labels-root /home/xyh/myCodeGitOnly/heat3d-ic-g2/data/g2/deepoheat_v1_volumetric_labels --fs-train /home/xyh/myData/DeepOHeat-v1-main/data/fs_train_volume.npy --output /tmp/v7_visualization_20260918/Heat3D-DeepOHeat.npz
```

### DeepOHeat

Checkpoint SHA256: `446c5eed25f94601dae1e5b27352574f2a77022e504aa9bf74df18423b5b144e`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
conda activate g2-deepoheat-v1
export HEAT3D_REPO=/home/xyh/myCodeGitOnly/heat3d-ic-g2
python -u /tmp/export_v7_visualization_inference.py --model DeepOHeat --checkpoint /home/xyh/myCodeGitOnly/heat3d-ic-g2/output/heat3d_v7_g2/deepoheat_v1_matched_768_128_p14/seed_0_retry1/DeepOHeat_v1_best.eqx --labels-root /home/xyh/myCodeGitOnly/heat3d-ic-g2/data/g2/deepoheat_v1_volumetric_labels --fs-train /home/xyh/myData/DeepOHeat-v1-main/data/fs_train_volume.npy --upstream /home/xyh/myCodeGitOnly/external/g2/DeepOHeat-v1 --output /tmp/v7_visualization_20260918/DeepOHeat.npz
```

## 重绘

```bash
python3 scripts/plot_v7_validation_triptychs.py --artifacts research_artifacts/v7_validation_figures
```

依赖：NumPy、SciPy、Matplotlib。绘图入口先验证 NPZ SHA256、同组样本身份、真实值和坐标一致性，再统一按行着色。已进行实际 PNG 视觉检查。

## 仓库状态与边界

分支 `codex/v7-validation-figures`，从 `2960ab5` 建立。新增两份 inference/plot 脚本及本复现说明；个人 skill 已更新并通过 quick_validate。`AGENTS.md` 已跟踪。远端原研究工作树没有 pull/reset 或源码改写，临时推理文件仅放 `/tmp/v7_visualization_20260918/`。本地/远端 data、output、checkpoints、logs 原始目录均未写入。
后续如需进一步验证空间保真度，可单独比较 P1i 原生支撑插值与同一验证样本的 dense FVM 切片。

## 2026-09-18 reference 与模型输出复核

上一版每行 union-range 使相同 reference 显示成不同颜色，这是绘图错误。
已改为整组 union-range，每面板保留独立色条；两组所有 reference 的 RGBA
SHA256 分别逐行完全一致。样本、坐标、真实值及 layer_id 也逐数组相等。
另将 P1i 同层支撑凸包外的 37.3% 区域统一屏蔽，避免外推制造空间结构。
未改动六个模型的原始预测；核验记录在 artifact 目录的 render_audit.json、
checkpoint_probe_audit.json、deepoheat_axis_audit.json、verification.json。

| 模型 | 独立核验 | 最大差异 K |
| --- | --- | ---: |
| G1 Full | 归档同样本预测 | 0.0098724 |
| G1 vanilla RIGNO | 归档同样本预测 | 0.0303192 |
| GINO | best checkpoint 内同样本 reload probe；所有归一化张量逐元素一致 | 0.0162506 |
| Transolver | best checkpoint 内同样本 reload probe；所有归一化张量逐元素一致 | 0 |
| DeepOHeat | 独立完整网格 forward | 0.0087202 |

DeepOHeat 正式 evaluator 得到该样本 CV-relative RMSE 2.28487039%，
导出数组重新计算为 2.28490191%，差异 0.00003152 个百分点；这与图中非 CV
加权 relative L2 2.28358% 是不同指标。27 点子网格查询差异为 0.0164986 K，
保留该数值差异，不声称 bitwise deterministic。上游输出轴定义为 bijky，
坐标为 meshgrid(indexing=ij) 后 C-order 展平，没有发现转置或样本错配。

原始全样本平均偏差分别为 RIGNO -15.9692 K、GINO -35.4629 K、
Transolver -32.3599 K、DeepOHeat -0.57998 K，远大于上述重放差异。
现有证据支持这些低估存在于冻结模型输出，不能将其修饰成绘图正常的预测。
本复核不评估原训练过程的最优性，也不将单个验证样本升级为模型排名。
