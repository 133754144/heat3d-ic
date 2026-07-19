# Gate 6O launch commands

Stage 2 已使用 WSL2 上既有 V38 e543 checkpoint 完成 e40。以下命令仅
保留为可复现记录，不应重复启动：

```bash
ssh wsl2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
cd ~/myCodeGitOnly/heat3d-ic
git pull --ff-only origin research/v5
python scripts/check_heat3d_v5_gate6o.py
mkdir -p output/heat3d_v5_gate6o_stage2_logs
python scripts/run_heat3d_v4_config.py \
  --config configs/heat3d_v5/generated/V4P5_39_gate6o_e543_scale_mlp_calibration_e40.yaml \
  2>&1 | tee output/heat3d_v5_gate6o_stage2_logs/V4P5_39_gate6o_e543_scale_mlp_calibration_e40.log
```

监控：

```bash
tail -f ~/myCodeGitOnly/heat3d-ic/output/heat3d_v5_gate6o_stage2_logs/V4P5_39_gate6o_e543_scale_mlp_calibration_e40.log
```

seed1 配对配置均为 `not_started`，仅在后续明确授权后于同一 WSL2
checkout、同一 commit 分别运行。启动前必须先将两份配置所在的同一
HEAD 拉取到 WSL2，并再次运行 Gate 6O checker：

```bash
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_40_gate6o_seed1_full_graph_e600.yaml
python scripts/run_heat3d_v4_config.py --config configs/heat3d_v5/generated/V4P5_41_gate6o_seed1_r2r_mask_p005_e600.yaml
```
