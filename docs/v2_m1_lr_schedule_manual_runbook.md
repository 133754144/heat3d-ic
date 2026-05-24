# Heat3D v2 M1 LR schedule manual runbook

本轮只准备 config、文档和 dry-run smoke，不训练、不运行 diagnostics、不生成 output。

## 新实验

Config:

`configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_decay_e5_to1e4_seed0.yaml`

目标是验证 M1 在 `lr=3e-4` 早期学习后，从 epoch 5 开始降到 `1e-4` 是否能缓解 early-best 和 final degradation：

- epochs 1-4: lr=3e-4
- epochs 5-50: lr=1e-4

除 LR schedule、description、run name、output dir 外，其余 dataset、model、optimizer type、batch size、loss、diagnostics 和 baseline reference 继承当前 `lr=3e-4` M1 e50 配置。

## SSH 手动训练

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

config_path = "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_decay_e5_to1e4_seed0.yaml"
config = load_v2_config(config_path)
command = build_training_command(config, python_executable="python")
print(" ".join(command), flush=True)
raise SystemExit(subprocess.call(command))
PY
```

Expected output directory:

`output/heat3d_v2_runs/m1_batch_e50_lr3e4_decay_e5_to1e4_seed0`

## Post-run checks

After manual training completes, verify these files:

- `loss_summary.json`
- `run_config.json`
- `predictions.npz`
- `best_predictions.npz`
- `train.log`

Then ask Codex to read the output directory and, if predictions are present, generate or inspect diagnostics. Do not commit `output/`.
