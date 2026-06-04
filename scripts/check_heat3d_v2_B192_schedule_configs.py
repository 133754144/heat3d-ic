#!/usr/bin/env python3
"""Dry-run checks for B192 LR schedule configs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


CASES = [
    (
        "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_rapid_decay_seed0.yaml",
        "rapid_decay",
        "output/heat3d_v2_runs/m1_B192_base_mse_rapid_decay_seed0",
        "m1_B192_base_mse_rapid_decay_seed0",
    ),
    (
        "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_warmup_cosine_seed0.yaml",
        "warmup_cosine",
        "output/heat3d_v2_runs/m1_B192_base_mse_warmup_cosine_seed0",
        "m1_B192_base_mse_warmup_cosine_seed0",
    ),
]


def _joined(command: list[str]) -> str:
    return " ".join(command)


def main() -> int:
    for path, schedule, output_dir, run_name in CASES:
        config = load_v2_config(ROOT / path)
        command = build_training_command(config, python_executable="python")
        joined = _joined(command)
        assert config["run"]["batch_size"] == 192
        assert config["run"]["validation_batch_size"] == 192
        assert config["run"]["prediction_batch_size"] == 192
        assert config["loss"]["mode"] == "mse"
        assert config["optimizer"]["lr_schedule"] == schedule
        assert config["optimizer"]["lr"] == 3.0e-4
        assert config["optimizer"]["weight_decay"] == 0.0
        assert config["export"]["output_dir"] == output_dir
        assert config["export"]["run_name"] == run_name
        assert f"--lr-schedule {schedule}" in joined
        assert "--batch-size 192" in joined
        assert "--validation-batch-size 192" in joined
        assert "--prediction-batch-size 192" in joined
        assert "--loss-mode mse" in joined
        assert "--lr 0.0003" in joined
        assert "--weight-decay 0.0" in joined
        assert "--save-predictions" in command
        assert "--save-best-predictions" in command
    print("Heat3D v2 B192 schedule config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
