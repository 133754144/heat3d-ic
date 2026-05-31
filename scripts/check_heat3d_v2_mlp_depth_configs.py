"""Smoke-check Heat3D v2 MLP-depth configs.

Dry-run only: load YAML and build runner commands. This does not import the
training runner, read datasets, create output directories, or start training.
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
    "configs/heat3d_v2/frozen_v1_e002_adamw_m15_B96_base_mse_warmup_cosine_mlp3_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m15_B96_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml",
]


def main() -> None:
    run_names: set[str] = set()
    output_dirs: set[str] = set()

    for relative_path in CONFIGS:
        config = load_v2_config(REPO_ROOT / relative_path)
        command = build_training_command(config, python_executable="python3")
        _check_config(relative_path, config, command)

        run_name = config["export"]["run_name"]
        output_dir = config["export"]["output_dir"]
        if run_name in run_names:
            raise AssertionError(f"{relative_path}: duplicate run_name {run_name!r}")
        if output_dir in output_dirs:
            raise AssertionError(f"{relative_path}: duplicate output_dir {output_dir!r}")
        run_names.add(run_name)
        output_dirs.add(output_dir)

    print(f"Checked {len(CONFIGS)} Heat3D v2 MLP-depth configs.")
    print("Heat3D v2 MLP-depth config smoke passed.")


def _check_config(relative_path: str, config: dict[str, Any], command: list[str]) -> None:
    dataset = config["dataset"]
    model = config["model"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    run = config["run"]
    export = config["export"]

    if config.get("config_role") != "controlled":
        raise AssertionError(f"{relative_path}: expected controlled config_role")
    if run.get("epochs") not in {2, 400}:
        raise AssertionError(f"{relative_path}: expected epochs=2 or 400")
    if run.get("batch_size") != 96:
        raise AssertionError(f"{relative_path}: batch_size must be 96")
    if run.get("validation_batch_size") != 96:
        raise AssertionError(f"{relative_path}: validation_batch_size must be 96")
    if run.get("prediction_batch_size") != 96:
        raise AssertionError(f"{relative_path}: prediction_batch_size must be 96")

    actual_shape = (
        model.get("node_latent_size"),
        model.get("edge_latent_size"),
        model.get("processor_steps"),
        model.get("mlp_hidden_layers"),
    )
    if actual_shape != (96, 96, 6, 3):
        raise AssertionError(
            f"{relative_path}: expected M1.5 mlp3 shape, got {actual_shape}"
        )

    if optimizer.get("name") != "adamw":
        raise AssertionError(f"{relative_path}: expected optimizer=adamw")
    if float(optimizer.get("lr")) != 0.0003:
        raise AssertionError(f"{relative_path}: expected lr=3e-4")
    if optimizer.get("lr_schedule") != "warmup_cosine":
        raise AssertionError(f"{relative_path}: expected warmup_cosine schedule")
    if int(optimizer.get("warmup_epochs")) != 10:
        raise AssertionError(f"{relative_path}: expected warmup_epochs=10")
    if float(optimizer.get("weight_decay")) != 0.0001:
        raise AssertionError(f"{relative_path}: expected weight_decay=1e-4")
    if float(optimizer.get("gradient_clip_norm")) != 1.0:
        raise AssertionError(f"{relative_path}: expected gradient_clip_norm=1.0")

    if loss.get("mode") != "mse":
        raise AssertionError(f"{relative_path}: expected loss mode mse")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_best_predictions must be true")
    if not str(export.get("output_dir", "")).startswith("output/heat3d_v2_runs/"):
        raise AssertionError(f"{relative_path}: output_dir must be under output/heat3d_v2_runs")

    split_map_path = dataset.get("split_map_path")
    if not split_map_path or not (REPO_ROOT / split_map_path).exists():
        raise AssertionError(f"{relative_path}: split_map_path missing or not found")

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
        "--save-predictions",
        "--save-best-predictions",
    ):
        if flag not in command:
            raise AssertionError(f"{relative_path}: command missing {flag}")


if __name__ == "__main__":
    main()
