#!/usr/bin/env python3
"""Smoke-check B192 stratified rerun configs."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


SPLIT_MAP = "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json"
CONFIGS = [
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_hotspot_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr1e4_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e5_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_full_lr1e4_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e4_wd1e8_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e4_wd0_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_rapid_decay_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_warmup_cosine_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e4_clip05_seed0_stratified.yaml",
    "configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_lr3e4_clip01_seed0_stratified.yaml",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    seen_outputs: set[str] = set()
    seen_run_names: set[str] = set()
    for config_path in CONFIGS:
        config = load_v2_config(config_path)
        dataset = config["dataset"]
        run = config["run"]
        export = config["export"]
        _require(dataset.get("split_map_path") == SPLIT_MAP, f"{config_path}: missing split_map_path")
        _require(run.get("batch_size") == 192, f"{config_path}: batch_size must be 192")
        _require(run.get("validation_batch_size") == 192, f"{config_path}: validation_batch_size must be 192")
        _require(run.get("prediction_batch_size") == 192, f"{config_path}: prediction_batch_size must be 192")
        _require(export.get("output_dir") not in seen_outputs, f"{config_path}: duplicate output_dir")
        _require(export.get("run_name") not in seen_run_names, f"{config_path}: duplicate run_name")
        seen_outputs.add(str(export.get("output_dir")))
        seen_run_names.add(str(export.get("run_name")))
        command = build_training_command(config, python_executable="python")
        _require("--split-map" in command and SPLIT_MAP in command, f"{config_path}: command missing --split-map")
        _require("--batch-size" in command and "192" in command, f"{config_path}: command missing B192")
        _require("--save-predictions" in command, f"{config_path}: command must save final predictions")
        _require("--save-best-predictions" in command, f"{config_path}: command must save best predictions")
    print("Heat3D v2 B192 stratified rerun config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
