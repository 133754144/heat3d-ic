"""Smoke-check Heat3D v2 e100 and M1.5 stratified configs.

This script only loads configs and builds commands. It does not train and does
not write outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


SPLIT_MAP = "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json"


CASES = [
    {
        "path": "configs/heat3d_v2/frozen_v1_e100_adamw_m1_B192_base_mse_lr3e4_stratified_seed0.yaml",
        "epochs": 100,
        "lr": 3.0e-4,
        "lr_schedule": "constant",
        "node_latent_size": 64,
        "edge_latent_size": 64,
        "processor_steps": 4,
        "run_name": "m1_B192_base_mse_lr3e4_e100_stratified_seed0",
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e100_adamw_m1_B192_base_mse_lr1e4_stratified_seed0.yaml",
        "epochs": 100,
        "lr": 1.0e-4,
        "lr_schedule": "constant",
        "node_latent_size": 64,
        "edge_latent_size": 64,
        "processor_steps": 4,
        "run_name": "m1_B192_base_mse_lr1e4_e100_stratified_seed0",
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e100_adamw_m1_B192_base_mse_warmup_cosine_stratified_seed0.yaml",
        "epochs": 100,
        "lr": 3.0e-4,
        "lr_schedule": "warmup_cosine",
        "node_latent_size": 64,
        "edge_latent_size": 64,
        "processor_steps": 4,
        "run_name": "m1_B192_base_mse_warmup_cosine_e100_stratified_seed0",
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e050_adamw_m15_B192_base_mse_lr3e4_stratified_seed0.yaml",
        "epochs": 50,
        "lr": 3.0e-4,
        "lr_schedule": "constant",
        "node_latent_size": 96,
        "edge_latent_size": 96,
        "processor_steps": 6,
        "run_name": "m15_B192_base_mse_lr3e4_e50_stratified_seed0",
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _has_option(command: list[str], option: str, value: str) -> bool:
    return any(command[i] == option and command[i + 1] == value for i in range(len(command) - 1))


def main() -> None:
    output_dirs: set[str] = set()
    run_names: set[str] = set()

    for case in CASES:
        config_path = ROOT / case["path"]
        _require(config_path.exists(), f"missing config: {config_path}")
        config = load_v2_config(config_path)

        dataset = config.get("dataset", {})
        model = config.get("model", {})
        optimizer = config.get("optimizer", {})
        run = config.get("run", {})
        export = config.get("export", {})
        loss = config.get("loss", {})

        _require(dataset.get("split_map_path") == SPLIT_MAP, f"{case['path']}: split_map_path mismatch")
        _require(run.get("batch_size") == 192, f"{case['path']}: batch_size must be 192")
        _require(run.get("validation_batch_size") == 192, f"{case['path']}: validation_batch_size must be 192")
        _require(run.get("prediction_batch_size") == 192, f"{case['path']}: prediction_batch_size must be 192")
        _require(run.get("epochs") == case["epochs"], f"{case['path']}: epochs mismatch")
        _require(loss.get("mode") == "mse", f"{case['path']}: loss.mode must be mse")
        _require(abs(float(optimizer.get("lr")) - case["lr"]) < 1e-12, f"{case['path']}: lr mismatch")
        _require(optimizer.get("lr_schedule") == case["lr_schedule"], f"{case['path']}: lr_schedule mismatch")
        _require(model.get("node_latent_size") == case["node_latent_size"], f"{case['path']}: node latent mismatch")
        _require(model.get("edge_latent_size") == case["edge_latent_size"], f"{case['path']}: edge latent mismatch")
        _require(model.get("processor_steps") == case["processor_steps"], f"{case['path']}: processor steps mismatch")
        _require(export.get("run_name") == case["run_name"], f"{case['path']}: run_name mismatch")
        _require(export.get("output_dir") not in output_dirs, f"{case['path']}: duplicate output_dir")
        _require(export.get("run_name") not in run_names, f"{case['path']}: duplicate run_name")
        output_dirs.add(export.get("output_dir"))
        run_names.add(export.get("run_name"))

        command = build_training_command(config, python_executable="python")
        _require(_has_option(command, "--split-map", SPLIT_MAP), f"{case['path']}: command missing split map")
        _require(_has_option(command, "--batch-size", "192"), f"{case['path']}: command missing batch-size 192")
        _require(_has_option(command, "--epochs", str(case["epochs"])), f"{case['path']}: command missing epochs")
        _require(_has_option(command, "--loss-mode", "mse"), f"{case['path']}: command missing loss mode")
        _require(_has_option(command, "--lr-schedule", case["lr_schedule"]), f"{case['path']}: command missing lr schedule")
        _require("--save-predictions" in command, f"{case['path']}: command missing save predictions")
        _require("--save-best-predictions" in command, f"{case['path']}: command missing save best predictions")

    print("Heat3D v2 e100/capacity config smoke passed.")


if __name__ == "__main__":
    main()
