"""Smoke checks for Heat3D v3 seed-decoupling YAML and command generation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


LEGACY_CONFIG = REPO_DIR / (
    "configs/heat3d_v2/"
    "frozen_v1_e400_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml"
)

NEW_CONFIGS = [
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed0_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed1_batchseed0_graphseed0_seed1.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed2_batchseed0_graphseed0_seed2.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed1_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed2_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e400_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed1_batchseed0_graphseed0_seed1.yaml"
    ),
]


def _flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise AssertionError(f"{flag} has no value")
    return command[index + 1]


def _assert_flag(command: list[str], flag: str, expected: str) -> None:
    actual = _flag_value(command, flag)
    if actual != expected:
        raise AssertionError(f"{flag}: expected {expected!r}, got {actual!r}")


def main() -> int:
    legacy = load_v2_config(LEGACY_CONFIG)
    legacy_command = build_training_command(legacy)
    if "--model-seed" in legacy_command or "--batch-order-seed" in legacy_command or "--graph-seed" in legacy_command:
        raise AssertionError("legacy config unexpectedly emits explicit seed-decoupling flags")
    _assert_flag(legacy_command, "--seed", "0")

    synthetic = copy.deepcopy(legacy)
    synthetic["optimizer"]["model_seed"] = 1
    synthetic["optimizer"]["batch_order_seed"] = 0
    synthetic["optimizer"]["graph_seed"] = 0
    validate_v2_config(synthetic)
    synthetic_command = build_training_command(synthetic)
    _assert_flag(synthetic_command, "--seed", "0")
    _assert_flag(synthetic_command, "--model-seed", "1")
    _assert_flag(synthetic_command, "--batch-order-seed", "0")
    _assert_flag(synthetic_command, "--graph-seed", "0")

    for path in NEW_CONFIGS:
        config = load_v2_config(path)
        command = build_training_command(config)
        optimizer = config["optimizer"]
        _assert_flag(command, "--seed", str(optimizer["seed"]))
        _assert_flag(command, "--model-seed", str(optimizer["model_seed"]))
        _assert_flag(command, "--batch-order-seed", str(optimizer["batch_order_seed"]))
        _assert_flag(command, "--graph-seed", str(optimizer["graph_seed"]))
        print(
            path.relative_to(REPO_DIR),
            "seed",
            optimizer["seed"],
            "model_seed",
            optimizer["model_seed"],
            "batch_order_seed",
            optimizer["batch_order_seed"],
            "graph_seed",
            optimizer["graph_seed"],
            "epochs",
            config["run"]["epochs"],
        )

    print("seed decoupling config smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
