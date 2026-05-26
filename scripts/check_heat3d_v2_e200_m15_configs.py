#!/usr/bin/env python3
"""Smoke-check Heat3D v2 e200 and memory-safe M1.5 configs.

This script only loads configs and builds runner commands. It does not train
and does not create output directories.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


SPLIT_MAP = "configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json"

CASES = [
    {
        "path": "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed0.yaml",
        "epochs": 200,
        "batch_size": 192,
        "node_latent_size": 64,
        "edge_latent_size": 64,
        "processor_steps": 4,
        "run_name": "m1_B192_base_mse_lr3e4_e200_stratified_seed0",
        "output_dir": "output/heat3d_v2_runs/m1_B192_base_mse_lr3e4_e200_stratified_seed0",
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e100_adamw_m15_B96_base_mse_lr3e4_stratified_seed0.yaml",
        "epochs": 100,
        "batch_size": 96,
        "node_latent_size": 96,
        "edge_latent_size": 96,
        "processor_steps": 6,
        "run_name": "m15_B96_base_mse_lr3e4_e100_stratified_seed0",
        "output_dir": "output/heat3d_v2_runs/m15_B96_base_mse_lr3e4_e100_stratified_seed0",
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _has_option(command: list[str], option: str, value: str) -> bool:
    return any(command[index] == option and command[index + 1] == value for index in range(len(command) - 1))


def main() -> int:
    output_dirs: set[str] = set()
    run_names: set[str] = set()

    for case in CASES:
        config_path = ROOT / case["path"]
        _require(config_path.exists(), f"missing config: {config_path}")
        config = load_v2_config(config_path)
        dataset = config["dataset"]
        model = config["model"]
        optimizer = config["optimizer"]
        loss = config["loss"]
        run = config["run"]
        export = config["export"]

        _require(dataset.get("split_map_path") == SPLIT_MAP, f"{case['path']}: split_map_path mismatch")
        _require(loss.get("mode") == "mse", f"{case['path']}: loss.mode must be mse")
        _require(optimizer.get("name") == "adamw", f"{case['path']}: optimizer must be adamw")
        _require(abs(float(optimizer.get("lr")) - 3.0e-4) < 1e-12, f"{case['path']}: lr mismatch")
        _require(optimizer.get("lr_schedule") == "constant", f"{case['path']}: lr_schedule mismatch")
        _require(float(optimizer.get("weight_decay")) == 1.0e-4, f"{case['path']}: weight_decay mismatch")
        _require(float(optimizer.get("gradient_clip_norm")) == 1.0, f"{case['path']}: clip mismatch")

        _require(run.get("epochs") == case["epochs"], f"{case['path']}: epochs mismatch")
        for key in ("batch_size", "validation_batch_size", "prediction_batch_size"):
            _require(run.get(key) == case["batch_size"], f"{case['path']}: {key} mismatch")

        _require(model.get("node_latent_size") == case["node_latent_size"], f"{case['path']}: node latent mismatch")
        _require(model.get("edge_latent_size") == case["edge_latent_size"], f"{case['path']}: edge latent mismatch")
        _require(model.get("processor_steps") == case["processor_steps"], f"{case['path']}: processor steps mismatch")
        _require(model.get("mlp_hidden_layers") == 2, f"{case['path']}: mlp_hidden_layers mismatch")

        _require(export.get("run_name") == case["run_name"], f"{case['path']}: run_name mismatch")
        _require(export.get("output_dir") == case["output_dir"], f"{case['path']}: output_dir mismatch")
        _require(export.get("save_final_predictions") is True, f"{case['path']}: save_final_predictions must be true")
        _require(export.get("save_best_predictions") is True, f"{case['path']}: save_best_predictions must be true")
        _require(export.get("output_dir") not in output_dirs, f"{case['path']}: duplicate output_dir")
        _require(export.get("run_name") not in run_names, f"{case['path']}: duplicate run_name")
        output_dirs.add(export.get("output_dir"))
        run_names.add(export.get("run_name"))

        command = build_training_command(config, python_executable="python")
        _require(_has_option(command, "--split-map", SPLIT_MAP), f"{case['path']}: command missing split map")
        _require(_has_option(command, "--epochs", str(case["epochs"])), f"{case['path']}: command missing epochs")
        _require(_has_option(command, "--batch-size", str(case["batch_size"])), f"{case['path']}: command missing batch size")
        _require(
            _has_option(command, "--validation-batch-size", str(case["batch_size"])),
            f"{case['path']}: command missing validation batch size",
        )
        _require(
            _has_option(command, "--prediction-batch-size", str(case["batch_size"])),
            f"{case['path']}: command missing prediction batch size",
        )
        _require(_has_option(command, "--loss-mode", "mse"), f"{case['path']}: command missing loss mode")
        _require(_has_option(command, "--lr", "0.0003"), f"{case['path']}: command missing lr")
        _require("--save-predictions" in command, f"{case['path']}: command missing final prediction export")
        _require("--save-best-predictions" in command, f"{case['path']}: command missing best prediction export")

    print("Heat3D v2 e200/M1.5 config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
