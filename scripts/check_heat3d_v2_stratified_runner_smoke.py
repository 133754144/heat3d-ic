#!/usr/bin/env python3
"""Dry-run checks for Heat3D v2 stratified split-map runner support."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command
from rigno.heat3d_v2_runner_command import DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP


SPLIT_MAP = Path("configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json")
STRATIFIED_CONFIG = Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_stratified_seed0.yaml")
OLD_CONFIG = Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_batch_lr3e4_B192_base_mse_seed0.yaml")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_split_map() -> dict[str, str]:
    payload = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))
    mapping = payload.get("sample_splits")
    _require(isinstance(mapping, dict), "split map must contain sample_splits")
    return {str(sample_id): str(split) for sample_id, split in mapping.items()}


def _check_split_counts() -> None:
    sample_splits = _load_split_map()
    counts: dict[str, int] = {}
    for split in sample_splits.values():
        counts[split] = counts.get(split, 0) + 1
    expected = {
        "train": 704,
        "valid_iid": 104,
        "valid_stress": 88,
        "test_id": 64,
        "test_ood_bc": 24,
        "test_ood_stack": 24,
        "test_ood_combined": 16,
    }
    _require(counts == expected, f"unexpected split counts: {counts}")


def _check_stratified_command() -> None:
    config = load_v2_config(STRATIFIED_CONFIG)
    dataset = config["dataset"]
    export = config["export"]
    _require(dataset.get("split_map_path") == str(SPLIT_MAP), "config must include dataset.split_map_path")
    _require(export.get("selection_metric") == "valid_loss", "valid_loss must map to valid_iid in split-map mode")

    command = build_training_command(config, python_executable="python")
    joined = " ".join(command)
    _require("--split-map" in command, "command must include --split-map")
    _require(str(SPLIT_MAP) in command, "command must pass the split-map path")
    _require("--batch-size" in command and "192" in command, "command must keep B192")
    _require("--loss-mode" in command and "mse" in command, "command must keep base MSE")
    _require("--selection-metric" in command and "valid_loss" in command, "command must keep valid_loss alias")
    _require("m1_B192_base_mse_stratified_seed0" in joined, "command must use stratified run name/output")


def _check_runner_default_text() -> None:
    runner_text = Path("scripts/run_heat3d_v1_medium_controlled_training_export.py").read_text(
        encoding="utf-8"
    )
    _require("DEFAULT_SPLIT_MAP" in runner_text, "runner must define DEFAULT_SPLIT_MAP")
    _require("default=None" in runner_text, "runner --split-map argparse default must stay None")
    _require(
        "args.split_map is None and _is_medium1024_gapA_subset(args.subset)" in runner_text,
        "runner must apply default split map only for medium1024 Gap-A",
    )
    _require('"configs"' in runner_text, "runner default split map must include configs/")
    _require('"heat3d_v2"' in runner_text, "runner default split map must include heat3d_v2/")
    _require(
        '"medium1024_gapA_stratified_split_seed0.json"' in runner_text,
        "runner default must point to latest split map",
    )
    _require(
        "v1_multilayer_bc_eq_physics_label_medium1024_gapA_full1024_v2" in runner_text,
        "runner default subset must be medium1024 Gap-A",
    )


def _check_old_command_uses_default_split() -> None:
    config = load_v2_config(OLD_CONFIG)
    command = build_training_command(config, python_executable="python")
    _require(
        DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP == str(SPLIT_MAP),
        "command builder default split map must match latest split map",
    )
    _require("--split-map" in command, "medium1024 config must default to --split-map")
    _require(str(SPLIT_MAP) in command, "old medium1024 config must use latest split map by default")
    _require("--batch-size" in command and "192" in command, "old B192 config must still build")


def main() -> int:
    _check_split_counts()
    _check_runner_default_text()
    _check_stratified_command()
    _check_old_command_uses_default_split()
    print("Heat3D v2 stratified runner smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
