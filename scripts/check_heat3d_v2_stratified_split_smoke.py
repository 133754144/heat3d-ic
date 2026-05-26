#!/usr/bin/env python3
"""Smoke checks for the Heat3D v2 medium1024 stratified split map."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

SPLIT_MAP = Path("configs/heat3d_v2/medium1024_gapA_stratified_split_seed0.json")
CONFIG = Path("configs/heat3d_v2/frozen_v1_e050_adamw_m1_B192_base_mse_stratified_seed0.yaml")
LEGAL_SPLITS = {
    "train",
    "valid_iid",
    "valid_stress",
    "test_id",
    "test_ood_bc",
    "test_ood_stack",
    "test_ood_combined",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return loaded


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(category_counts: dict[str, Any], key: str, split: str, value: str) -> int:
    return int(category_counts.get(key, {}).get(split, {}).get(value, 0))


def _check_split_map() -> None:
    payload = _load_json(SPLIT_MAP)
    sample_splits = payload.get("sample_splits")
    _require(isinstance(sample_splits, dict), "sample_splits must be a mapping")
    _require(len(sample_splits) == 1024, f"expected 1024 sample ids, found {len(sample_splits)}")
    _require(len(set(sample_splits)) == 1024, "duplicate sample ids found")
    split_values = {str(value) for value in sample_splits.values()}
    _require(split_values <= LEGAL_SPLITS, f"illegal split names: {sorted(split_values - LEGAL_SPLITS)}")

    counts = {split: int(count) for split, count in payload.get("split_counts", {}).items()}
    _require(sum(counts.values()) == 1024, "split_counts must sum to 1024")
    for split in ("train", "valid_iid", "valid_stress", "test_id"):
        _require(counts.get(split, 0) > 0, f"{split} must be non-empty")

    categories = payload.get("category_counts", {})
    train_low_power = _count(categories, "power_scale_category", "train", "low_power")
    train_diag3 = _count(categories, "k_field_mode", "train", "diag3")
    train_barrier = _count(categories, "k_region_mode", "train", "low_k_barrier_or_TIM_variation")
    valid_iid_low_power = _count(categories, "power_scale_category", "valid_iid", "low_power")
    valid_iid_diag3 = _count(categories, "k_field_mode", "valid_iid", "diag3")

    _require(train_low_power >= 50, f"train low_power coverage too small: {train_low_power}")
    _require(train_diag3 >= 150, f"train diag3 coverage too small: {train_diag3}")
    _require(train_barrier >= 50, f"train barrier/TIM coverage too small: {train_barrier}")
    _require(valid_iid_low_power <= 25, f"valid_iid low_power still too large: {valid_iid_low_power}")
    _require(valid_iid_diag3 <= 50, f"valid_iid diag3 still too large: {valid_iid_diag3}")


def _check_config_draft() -> None:
    try:
        from rigno.heat3d_v2_config import load_v2_config
        from rigno.heat3d_v2_runner_command import build_training_command
    except ImportError:
        text = CONFIG.read_text(encoding="utf-8")
        _require(f"split_map_path: {SPLIT_MAP}" in text, "config must point at split_map_path")
        _require("batch_size: 192" in text, "config must keep B192")
        _require("mode: mse" in text, "config must keep base MSE")
        _require(
            "output/heat3d_v2_runs/m1_B192_base_mse_stratified_seed0" in text,
            "config must use the stratified output dir",
        )
        return

    config = load_v2_config(CONFIG)
    dataset = config["dataset"]
    _require(dataset.get("split_map_path") == str(SPLIT_MAP), "config must point at the stratified split map")
    command = build_training_command(config, python_executable="python")
    _require("--batch-size" in command and "192" in command, "command must keep B192")
    _require("--loss-mode" in command and "mse" in command, "command must keep base MSE")
    _require(
        "--output-dir" in command and "output/heat3d_v2_runs/m1_B192_base_mse_stratified_seed0" in command,
        "command must use the stratified output dir",
    )
    _require("--split-map" in command and str(SPLIT_MAP) in command, "command must include split-map support")


def main() -> int:
    _check_split_map()
    _check_config_draft()
    print("Heat3D v2 stratified split smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
