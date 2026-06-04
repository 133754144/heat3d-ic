#!/usr/bin/env python3
"""Dry-run checks for Heat3D v2 boundary-mask memorization A/B configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


CONFIGS = (
    ("memorization_m2_B96_train1_e200_boundary_before_seed0.yaml", 1, False),
    ("memorization_m2_B96_train1_e200_boundary_fallback_seed0.yaml", 1, True),
    ("memorization_m2_B96_train4_e200_boundary_before_seed0.yaml", 4, False),
    ("memorization_m2_B96_train4_e200_boundary_fallback_seed0.yaml", 4, True),
    ("memorization_m2_B96_train16_e200_boundary_fallback_seed0.yaml", 16, True),
)


def main() -> int:
    output_dirs: set[str] = set()
    run_names: set[str] = set()
    previous_ids: set[str] = set()
    for count in (1, 4, 16):
        split_path = REPO_ROOT / f"configs/heat3d_v2/medium1024_gapA_memorization_train{count}_seed0.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = {sample_id for sample_id, split in payload["sample_splits"].items() if split == "train"}
        if len(train_ids) != count:
            raise AssertionError(f"{split_path}: expected {count} train samples")
        if previous_ids and not previous_ids.issubset(train_ids):
            raise AssertionError(f"{split_path}: train samples must be nested")
        previous_ids = train_ids

    for filename, count, fallback in CONFIGS:
        path = REPO_ROOT / "configs" / "heat3d_v2" / filename
        config = load_v2_config(path)
        command = build_training_command(config, python_executable="python3")
        dataset = config["dataset"]
        model = config["model"]
        run = config["run"]
        export = config["export"]
        expected_flag = "--boundary-mask-fallback" if fallback else "--no-boundary-mask-fallback"

        if dataset.get("boundary_mask_fallback") is not fallback or expected_flag not in command:
            raise AssertionError(f"{path}: boundary fallback command mismatch")
        if f"train{count}_seed0.json" not in dataset["split_map_path"]:
            raise AssertionError(f"{path}: wrong fixed split")
        if (model["node_latent_size"], model["edge_latent_size"], model["processor_steps"], model["mlp_hidden_layers"]) != (128, 128, 6, 2):
            raise AssertionError(f"{path}: expected M2 model")
        if run["epochs"] != 200 or run["batch_size"] != 96:
            raise AssertionError(f"{path}: expected e200/B96")
        if run["shuffle_train_batches"] is not False or run["train_metrics_schedule"] != "every_epoch":
            raise AssertionError(f"{path}: expected fixed order and every-epoch train metrics")
        if export["save_final_predictions"] or export["save_best_predictions"]:
            raise AssertionError(f"{path}: prediction export must be disabled")
        if export["output_dir"] in output_dirs or export["run_name"] in run_names:
            raise AssertionError(f"{path}: duplicate output_dir or run_name")
        output_dirs.add(export["output_dir"])
        run_names.add(export["run_name"])

    print("Heat3D v2 memorization boundary A/B config smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
