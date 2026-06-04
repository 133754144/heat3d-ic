"""Smoke-check Heat3D v2 large B48 capacity probe/long configs.

Dry-run only: load YAML and build runner commands. This does not import the
training runner, read datasets, create output directories, or start training.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config
from rigno.heat3d_v2_runner_command import build_training_command


CONFIGS = [
    "configs/heat3d_v2/frozen_v1_e002_adamw_m3width_B48_base_mse_warmup_cosine_mlp3_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e002_adamw_m25depth_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e002_adamw_m3depth_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e002_adamw_m35width_B48_base_mse_warmup_cosine_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e002_adamw_m35width_B48_base_mse_warmup_cosine_mlp3_capacity_probe_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m3width_B48_base_mse_warmup_cosine_mlp3_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m25depth_B48_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m3depth_B48_base_mse_warmup_cosine_stratified_seed0.yaml",
    "configs/heat3d_v2/frozen_v1_e400_adamw_m35width_B48_base_mse_warmup_cosine_stratified_seed0.yaml",
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

    print(f"Checked {len(CONFIGS)} Heat3D v2 large B48 capacity configs.")
    print("Heat3D v2 large B48 capacity config smoke passed.")


def _check_config(relative_path: str, config: dict, command: list[str]) -> None:
    dataset = config["dataset"]
    model = config["model"]
    run = config["run"]
    export = config["export"]

    if config.get("config_role") != "controlled":
        raise AssertionError(f"{relative_path}: expected controlled config_role")
    if int(run.get("batch_size")) != 48:
        raise AssertionError(f"{relative_path}: expected batch_size=48")
    if int(run.get("validation_batch_size")) != 48:
        raise AssertionError(f"{relative_path}: expected validation_batch_size=48")
    if int(run.get("prediction_batch_size")) != 48:
        raise AssertionError(f"{relative_path}: expected prediction_batch_size=48")
    if int(run.get("epochs")) not in {2, 400}:
        raise AssertionError(f"{relative_path}: expected epochs=2 or 400")
    if int(model.get("processor_steps")) > 8:
        raise AssertionError(f"{relative_path}: processor_steps>8 is not allowed")
    if int(model.get("mlp_hidden_layers")) > 3:
        raise AssertionError(f"{relative_path}: mlp_hidden_layers>3 is not allowed")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_best_predictions must be true")

    split_map_path = dataset.get("split_map_path")
    if not split_map_path or not (REPO_ROOT / split_map_path).exists():
        raise AssertionError(f"{relative_path}: split_map_path missing or not found")

    for flag in (
        "--epochs",
        "--node-latent-size",
        "--edge-latent-size",
        "--processor-steps",
        "--mlp-hidden-layers",
        "--batch-size",
        "--validation-batch-size",
        "--prediction-batch-size",
        "--save-predictions",
        "--save-best-predictions",
    ):
        if flag not in command:
            raise AssertionError(f"{relative_path}: command missing {flag}")


if __name__ == "__main__":
    main()
