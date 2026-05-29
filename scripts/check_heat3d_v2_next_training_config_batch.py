"""Smoke-check the next Heat3D v2 training config batch.

This script is intentionally dry-run only. It loads YAML configs, builds
commands, and checks config consistency without importing the runner, reading
datasets, creating output directories, or starting training.
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
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr1e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e5_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_rapid_decay_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_clip0p5_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_clip0p1_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_wd1e3_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_wd1e2_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_light_bg_bias_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_light_bg_over_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_light_bg_l1_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_light_bg_bias_over_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m125_B128_base_mse_lr3e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m125_B128_base_mse_lr1e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m125_B128_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m125_B96_base_mse_lr3e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m125_B96_base_mse_lr1e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed1.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr3e4_stratified_seed2.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_lr1e4_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m1_B192_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed1.yaml",
    "configs/heat3d_v2/frozen_v1_e200_adamw_m15_B96_base_mse_lr3e4_stratified_seed2.yaml",
]


def main() -> None:
    run_names: set[str] = set()
    output_dirs: set[str] = set()
    priority_counts: dict[str, int] = {}

    for relative_path in CONFIGS:
        config_path = REPO_ROOT / relative_path
        config = load_v2_config(config_path)
        command = build_training_command(config, python_executable="python3")
        _check_config(relative_path, config, command)

        export = config["export"]
        run_name = export["run_name"]
        output_dir = export["output_dir"]
        if run_name in run_names:
            raise AssertionError(f"{relative_path}: duplicate run_name {run_name!r}")
        if output_dir in output_dirs:
            raise AssertionError(f"{relative_path}: duplicate output_dir {output_dir!r}")
        run_names.add(run_name)
        output_dirs.add(output_dir)

        priority = config.get("metadata", {}).get("priority", "unknown")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    print(f"Checked {len(CONFIGS)} Heat3D v2 next-training configs.")
    print(
        "Priority counts: "
        + ", ".join(
            f"{priority}={count}" for priority, count in sorted(priority_counts.items())
        )
    )
    print("Heat3D v2 next training config batch smoke passed.")


def _check_config(relative_path: str, config: dict[str, Any], command: list[str]) -> None:
    dataset = config["dataset"]
    model = config["model"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    run = config["run"]
    export = config["export"]

    if config.get("config_role") != "controlled":
        raise AssertionError(f"{relative_path}: expected controlled config_role")
    if run.get("epochs") != 200:
        raise AssertionError(f"{relative_path}: expected epochs=200")
    if optimizer.get("name") != "adamw":
        raise AssertionError(f"{relative_path}: expected optimizer=adamw")
    if optimizer.get("gradient_clip_norm") is None:
        raise AssertionError(f"{relative_path}: missing gradient_clip_norm")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_best_predictions must be true")
    if export.get("selection_metric") != "valid_loss":
        raise AssertionError(f"{relative_path}: expected selection_metric=valid_loss")
    if not str(export.get("output_dir", "")).startswith("output/heat3d_v2_runs/"):
        raise AssertionError(f"{relative_path}: output_dir must be under output/heat3d_v2_runs")

    split_map_path = dataset.get("split_map_path")
    if not split_map_path:
        raise AssertionError(f"{relative_path}: missing dataset.split_map_path")
    if not (REPO_ROOT / split_map_path).exists():
        raise AssertionError(f"{relative_path}: split_map_path does not exist: {split_map_path}")

    batch_size = run.get("batch_size")
    if batch_size != run.get("validation_batch_size"):
        raise AssertionError(f"{relative_path}: validation_batch_size must match batch_size")
    if batch_size != run.get("prediction_batch_size"):
        raise AssertionError(f"{relative_path}: prediction_batch_size must match batch_size")

    model_key = _model_key(model)
    if model_key == "m15" and batch_size != 96:
        raise AssertionError(f"{relative_path}: M1.5 configs must use batch_size=96")
    if model_key == "m125" and batch_size not in {96, 128}:
        raise AssertionError(f"{relative_path}: M1.25 configs must use batch_size=96 or 128")
    if model_key == "m1" and batch_size != 192:
        raise AssertionError(f"{relative_path}: M1 configs must use batch_size=192")

    if loss.get("mode") == "mse":
        for key in (
            "background_l1_weight",
            "background_bias_weight",
            "background_over_weight",
            "background_relative_weight",
        ):
            if float(loss.get(key, 0.0)) != 0.0:
                raise AssertionError(f"{relative_path}: base MSE config has nonzero {key}")
    elif loss.get("mode") == "background_l1_bias":
        weights = [
            float(loss.get("background_l1_weight", 0.0)),
            float(loss.get("background_bias_weight", 0.0)),
            float(loss.get("background_over_weight", 0.0)),
        ]
        if max(weights) > 0.02 or sum(weights) <= 0.0:
            raise AssertionError(f"{relative_path}: light background weights are not conservative")
    else:
        raise AssertionError(f"{relative_path}: unexpected loss mode {loss.get('mode')!r}")

    required_flags = [
        "--split-map",
        "--epochs",
        "--batch-size",
        "--validation-batch-size",
        "--prediction-batch-size",
        "--optimizer",
        "--lr",
        "--save-predictions",
        "--save-best-predictions",
        "--selection-metric",
    ]
    for flag in required_flags:
        if flag not in command:
            raise AssertionError(f"{relative_path}: command missing {flag}")


def _model_key(model: dict[str, Any]) -> str:
    shape = (
        model.get("node_latent_size"),
        model.get("edge_latent_size"),
        model.get("processor_steps"),
        model.get("mlp_hidden_layers"),
    )
    if shape == (96, 96, 6, 2):
        return "m15"
    if shape == (80, 80, 5, 2):
        return "m125"
    if shape == (64, 64, 4, 2):
        return "m1"
    raise AssertionError(f"unexpected model shape {shape}")


if __name__ == "__main__":
    main()
