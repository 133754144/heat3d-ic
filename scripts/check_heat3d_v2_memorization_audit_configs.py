#!/usr/bin/env python3
"""Dry-run checks for fixed Heat3D v2 memorization-audit configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


COUNTS = (16, 32, 96)


def _paths(count: int) -> tuple[Path, Path]:
    return (
        REPO_ROOT / f"configs/heat3d_v2/medium1024_gapA_memorization_train{count}_seed0.json",
        REPO_ROOT / f"configs/heat3d_v2/memorization_m2_B96_train{count}_e200_seed0.yaml",
    )


def main() -> int:
    previous_train_ids: set[str] = set()
    output_dirs: set[str] = set()
    run_names: set[str] = set()

    for count in COUNTS:
        split_path, config_path = _paths(count)
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        sample_splits = split_payload["sample_splits"]
        train_ids = {sample_id for sample_id, split in sample_splits.items() if split == "train"}
        valid_ids = {sample_id for sample_id, split in sample_splits.items() if split == "valid_iid"}
        if len(train_ids) != count:
            raise AssertionError(f"{split_path}: expected {count} train ids, found {len(train_ids)}")
        if len(valid_ids) != 8:
            raise AssertionError(f"{split_path}: expected 8 contract valid_iid ids")
        if previous_train_ids and not previous_train_ids.issubset(train_ids):
            raise AssertionError(f"{split_path}: train subsets must be nested")
        previous_train_ids = train_ids

        config = load_v2_config(config_path)
        command = build_training_command(config, python_executable="python3")
        model = config["model"]
        optimizer = config["optimizer"]
        run = config["run"]
        export = config["export"]

        expected_split = str(split_path.relative_to(REPO_ROOT))
        if config["dataset"].get("split_map_path") != expected_split:
            raise AssertionError(f"{config_path}: unexpected split_map_path")
        if model.get("node_latent_size") != 128 or model.get("edge_latent_size") != 128:
            raise AssertionError(f"{config_path}: expected M2 width 128")
        if model.get("processor_steps") != 6 or model.get("mlp_hidden_layers") != 2:
            raise AssertionError(f"{config_path}: expected M2 steps6/mlp2")
        if optimizer.get("name") != "adamw" or optimizer.get("lr") != 3.0e-4:
            raise AssertionError(f"{config_path}: expected AdamW lr=3e-4")
        if run.get("epochs") != 200 or run.get("batch_size") != 96:
            raise AssertionError(f"{config_path}: expected e200/B96")
        if run.get("shuffle_train_batches") is not False:
            raise AssertionError(f"{config_path}: memorization audit must keep fixed batch order")
        if run.get("train_metrics_schedule") != "every_epoch":
            raise AssertionError(f"{config_path}: train metrics must be recorded every epoch")
        if export.get("save_final_predictions") is not False or export.get("save_best_predictions") is not False:
            raise AssertionError(f"{config_path}: prediction export must be disabled")
        if "--split-map" not in command or expected_split not in command:
            raise AssertionError(f"{config_path}: command missing fixed split map")
        if "--save-predictions" in command or "--save-best-predictions" in command:
            raise AssertionError(f"{config_path}: command must not enable prediction export")

        if export["output_dir"] in output_dirs or export["run_name"] in run_names:
            raise AssertionError(f"{config_path}: duplicate output_dir or run_name")
        output_dirs.add(export["output_dir"])
        run_names.add(export["run_name"])

    print("Heat3D v2 memorization audit config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
