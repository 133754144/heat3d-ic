"""Smoke-check Heat3D v2 M2 capacity configs.

Dry-run only: load YAML, validate it, and build runner commands. This script
does not import the training runner, read datasets, create output directories,
or start training.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


CONFIGS = [
    "configs/heat3d_v2/frozen_v1_e005_adamw_m2lite_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e005_adamw_m2width_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e005_adamw_m2depthlite_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e005_adamw_m2risk_B96_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m2lite_B96_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m2depthlite_B96_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m2lite_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e300_adamw_m2risk_B96_base_mse_warmup_cosine_stratified_seed0.yaml",
]


EXPECTED_CAPACITY = {
    "M2-lite-width": (112, 112, 6),
    "M2-width": (128, 128, 6),
    "M2-lite-depth": (112, 112, 8),
    "M2-risk": (128, 128, 8),
}


def main() -> None:
    run_names: set[str] = set()
    output_dirs: set[str] = set()
    capacity_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}

    for relative_path in CONFIGS:
        config = load_v2_config(REPO_ROOT / relative_path)
        command = build_training_command(config, python_executable="python3")
        capacity_label, priority = _check_config(relative_path, config, command)

        run_name = config["export"]["run_name"]
        output_dir = config["export"]["output_dir"]
        if run_name in run_names:
            raise AssertionError(f"{relative_path}: duplicate run_name {run_name!r}")
        if output_dir in output_dirs:
            raise AssertionError(f"{relative_path}: duplicate output_dir {output_dir!r}")
        run_names.add(run_name)
        output_dirs.add(output_dir)

        capacity_counts[capacity_label] = capacity_counts.get(capacity_label, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    print(f"Checked {len(CONFIGS)} Heat3D v2 M2 capacity configs.")
    print(
        "Capacity counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(capacity_counts.items()))
    )
    print(
        "Priority counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(priority_counts.items()))
    )
    print("Heat3D v2 M2 capacity config smoke passed.")


def _check_config(relative_path: str, config: dict[str, Any], command: list[str]) -> tuple[str, str]:
    dataset = config["dataset"]
    model = config["model"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    run = config["run"]
    export = config["export"]
    metadata = config.get("metadata", {})

    if config.get("config_role") != "controlled":
        raise AssertionError(f"{relative_path}: expected controlled config_role")
    if run.get("epochs") not in {5, 300, 400}:
        raise AssertionError(f"{relative_path}: expected epochs=5, 300, or 400")
    if run.get("batch_size") != 96:
        raise AssertionError(f"{relative_path}: batch_size must be 96")
    if run.get("validation_batch_size") != 96:
        raise AssertionError(f"{relative_path}: validation_batch_size must be 96")
    if run.get("prediction_batch_size") != 96:
        raise AssertionError(f"{relative_path}: prediction_batch_size must be 96")

    if optimizer.get("name") != "adamw":
        raise AssertionError(f"{relative_path}: expected optimizer=adamw")
    if float(optimizer.get("lr")) != 0.0003:
        raise AssertionError(f"{relative_path}: expected lr=3e-4")
    if optimizer.get("lr_schedule") != "warmup_cosine":
        raise AssertionError(f"{relative_path}: expected warmup_cosine schedule")
    if int(optimizer.get("warmup_epochs")) != 10:
        raise AssertionError(f"{relative_path}: expected warmup_epochs=10")
    if float(optimizer.get("min_lr")) != 1.0e-6:
        raise AssertionError(f"{relative_path}: expected min_lr=1e-6")
    if float(optimizer.get("gradient_clip_norm")) != 1.0:
        raise AssertionError(f"{relative_path}: expected gradient_clip_norm=1.0")

    if loss.get("mode") != "mse":
        raise AssertionError(f"{relative_path}: expected loss mode mse")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_best_predictions must be true")
    if export.get("selection_metric") != "valid_loss":
        raise AssertionError(f"{relative_path}: expected selection_metric=valid_loss")
    if not str(export.get("output_dir", "")).startswith("output/heat3d_v2_runs/"):
        raise AssertionError(f"{relative_path}: output_dir must be under output/heat3d_v2_runs")

    split_map_path = dataset.get("split_map_path")
    if not split_map_path or not (REPO_ROOT / split_map_path).exists():
        raise AssertionError(f"{relative_path}: split_map_path missing or not found")

    capacity_label = metadata.get("capacity_label")
    expected_shape = EXPECTED_CAPACITY.get(capacity_label)
    if expected_shape is None:
        raise AssertionError(f"{relative_path}: unexpected capacity_label {capacity_label!r}")
    actual_shape = (
        model.get("node_latent_size"),
        model.get("edge_latent_size"),
        model.get("processor_steps"),
    )
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{relative_path}: expected {capacity_label} shape {expected_shape}, got {actual_shape}"
        )
    if int(model.get("mlp_hidden_layers")) != 2:
        raise AssertionError(f"{relative_path}: expected mlp_hidden_layers=2")

    for flag in (
        "--split-map",
        "--epochs",
        "--node-latent-size",
        "--edge-latent-size",
        "--processor-steps",
        "--mlp-hidden-layers",
        "--batch-size",
        "--validation-batch-size",
        "--prediction-batch-size",
        "--optimizer",
        "--lr",
        "--lr-schedule",
        "--warmup-epochs",
        "--min-lr",
        "--save-predictions",
        "--save-best-predictions",
        "--selection-metric",
    ):
        if flag not in command:
            raise AssertionError(f"{relative_path}: command missing {flag}")

    priority = str(metadata.get("priority", "unknown"))
    return str(capacity_label), priority


if __name__ == "__main__":
    main()
