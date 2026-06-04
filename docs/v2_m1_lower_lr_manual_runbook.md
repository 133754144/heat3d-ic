# Heat3D v2 M1 lower-lr 手动运行说明

本轮只准备 lower-lr ablation 配置，不训练、不运行 diagnostics、不生成 output。

## 配置

- `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml`: lr=3e-4
- `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr1e4_seed0.yaml`: lr=1e-4

两个配置均以 `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_e50.yaml` 为模板。除 `description`、`optimizer.lr`、`export.output_dir`、`export.run_name` 以及显式写入的 `run.train_metrics_schedule=half_and_final`、`run.grad_norm_report_every=10` 外，其余 dataset、model、optimizer、loss、run、export、diagnostics、baseline_reference 字段保持一致。

## 目的

lr=1e-3 M1 mini-batch e50 baseline 的 best epoch 很早。lower-lr 对照用于检查：

- best epoch 是否后移；
- valid loss / raw DeltaT 指标是否改善；
- final 与 best prediction 的 field-shape diagnostics 是否保持或改善。

## SSH WSL 手动运行

在 SSH WSL 上进入仓库后运行其中一个配置。示例：

```bash
ssh WSL
cd ~/myCodeGitOnly/heat3d-ic
conda activate rigno
git switch research/v2-training-system
git pull --ff-only
python - <<'PY'
import subprocess
from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command

config_path = "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_seed0.yaml"
config = load_v2_config(config_path)
command = build_training_command(config, python_executable="python")
print(" ".join(command), flush=True)
raise SystemExit(subprocess.call(command))
PY
```

将 `config_path` 改为 `configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr1e4_seed0.yaml` 可运行 lr=1e-4 对照。

## 输出与审查

训练完成后 runner 会自动保存：

- `run_config.json`
- `loss_summary.json`
- `predictions.npz`
- `best_predictions.npz`

diagnostics 可在训练后手动运行。`output/` 不提交。训练结束后让 Codex 读取对应 output 目录，审查 best epoch、valid 指标、prediction 文件和 diagnostics 结果。
