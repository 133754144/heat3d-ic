#!/usr/bin/env python3
"""Dry-run checks for Heat3D v2 one-sample fitting audit configs."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


CONFIGS = {
    "memorization_m2_B96_train1_e200_boundary_fallback_export_seed0.yaml": {
        "optimizer": "adamw",
        "lr": 3.0e-4,
        "schedule": "warmup_cosine",
        "weight_decay": 1.0e-4,
        "prediction_split": "train",
    },
    "memorization_m2_B96_train1_e200_adam_constant_lr3e4_seed0.yaml": {
        "optimizer": "adam",
        "lr": 3.0e-4,
        "schedule": "constant",
        "weight_decay": 0.0,
    },
    "memorization_m2_B96_train1_e200_adam_constant_lr1e3_seed0.yaml": {
        "optimizer": "adam",
        "lr": 1.0e-3,
        "schedule": "constant",
        "weight_decay": 0.0,
    },
    "memorization_m2_B96_train1_e200_adam_constant_lr1e4_seed0.yaml": {
        "optimizer": "adam",
        "lr": 1.0e-4,
        "schedule": "constant",
        "weight_decay": 0.0,
    },
}


def _option(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise AssertionError(f"missing command option {flag}") from exc


def main() -> int:
    output_dirs: set[str] = set()
    for filename, expected in CONFIGS.items():
        path = REPO_ROOT / "configs" / "heat3d_v2" / filename
        config = load_v2_config(path)
        command = build_training_command(config, python_executable="python3")
        dataset = config["dataset"]
        model = config["model"]
        optimizer = config["optimizer"]
        run = config["run"]
        export = config["export"]

        if not dataset.get("boundary_mask_fallback"):
            raise AssertionError(f"{path}: expected boundary fallback")
        if not dataset["split_map_path"].endswith("medium1024_gapA_memorization_train1_seed0.json"):
            raise AssertionError(f"{path}: wrong fixed one-sample split")
        if (model["node_latent_size"], model["edge_latent_size"], model["processor_steps"], model["mlp_hidden_layers"]) != (128, 128, 6, 2):
            raise AssertionError(f"{path}: expected M2 model")
        if run["epochs"] != 200 or run["batch_size"] != 96 or run["shuffle_train_batches"]:
            raise AssertionError(f"{path}: expected fixed-order e200/B96 audit")
        if optimizer["name"] != expected["optimizer"] or optimizer["lr"] != expected["lr"]:
            raise AssertionError(f"{path}: optimizer/lr mismatch")
        if optimizer["lr_schedule"] != expected["schedule"] or optimizer["weight_decay"] != expected["weight_decay"]:
            raise AssertionError(f"{path}: schedule/weight_decay mismatch")
        if _option(command, "--optimizer") != expected["optimizer"]:
            raise AssertionError(f"{path}: command optimizer mismatch")
        if float(_option(command, "--lr")) != expected["lr"]:
            raise AssertionError(f"{path}: command lr mismatch")

        prediction_split = expected.get("prediction_split")
        if prediction_split:
            if export.get("prediction_split") != prediction_split:
                raise AssertionError(f"{path}: expected train-only prediction export")
            if _option(command, "--prediction-split") != prediction_split:
                raise AssertionError(f"{path}: command prediction split mismatch")
            if "--save-predictions" not in command or "--save-best-predictions" not in command:
                raise AssertionError(f"{path}: final/best prediction export must be enabled")
        elif export["save_final_predictions"] or export["save_best_predictions"]:
            raise AssertionError(f"{path}: optimizer audit must not export predictions")

        if export["output_dir"] in output_dirs:
            raise AssertionError(f"{path}: duplicate output_dir")
        output_dirs.add(export["output_dir"])

    print("Heat3D v2 one-sample fitting config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
