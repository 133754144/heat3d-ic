"""Smoke-check Heat3D v2 B96 follow-up configs.

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
from rigno.heat3d_v2_runner_command import DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP
from rigno.heat3d_v2_runner_command import build_training_command


CONFIGS = [
    {
        "path": "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed1.yaml",
        "seed": 1,
        "weight_decay": 1.0e-4,
        "loss_mode": "mse",
        "background_bias_weight": 0.0,
        "background_over_weight": 0.0,
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_stratified_seed2.yaml",
        "seed": 2,
        "weight_decay": 1.0e-4,
        "loss_mode": "mse",
        "background_bias_weight": 0.0,
        "background_over_weight": 0.0,
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_base_mse_warmup_cosine_wd1e3_stratified_seed0.yaml",
        "seed": 0,
        "weight_decay": 1.0e-3,
        "loss_mode": "mse",
        "background_bias_weight": 0.0,
        "background_over_weight": 0.0,
    },
    {
        "path": "configs/heat3d_v2/frozen_v1_e400_adamw_m2width_B96_light_bg_bias_over_warmup_cosine_stratified_seed0.yaml",
        "seed": 0,
        "weight_decay": 1.0e-4,
        "loss_mode": "background_l1_bias",
        "background_bias_weight": 0.01,
        "background_over_weight": 0.01,
    },
]

M25_B96_LONG_CONFIG = (
    "configs/heat3d_v2/"
    "frozen_v1_e400_adamw_m25width_B96_base_mse_warmup_cosine_stratified_seed0.yaml"
)


def main() -> None:
    run_names: set[str] = set()
    output_dirs: set[str] = set()

    for item in CONFIGS:
        relative_path = item["path"]
        config = load_v2_config(REPO_ROOT / relative_path)
        command = build_training_command(config, python_executable="python3")
        _check_config(relative_path, config, command, item)

        run_name = config["export"]["run_name"]
        output_dir = config["export"]["output_dir"]
        if run_name in run_names:
            raise AssertionError(f"{relative_path}: duplicate run_name {run_name!r}")
        if output_dir in output_dirs:
            raise AssertionError(f"{relative_path}: duplicate output_dir {output_dir!r}")
        run_names.add(run_name)
        output_dirs.add(output_dir)

    if (REPO_ROOT / M25_B96_LONG_CONFIG).exists():
        raise AssertionError(
            "M2.5 B96 e400 config must not exist because the two-epoch probe OOMed"
        )

    print(f"Checked {len(CONFIGS)} Heat3D v2 B96 follow-up configs.")
    print("Heat3D v2 B96 follow-up config smoke passed.")


def _check_config(
    relative_path: str,
    config: dict,
    command: list[str],
    expected: dict,
) -> None:
    dataset = config["dataset"]
    model = config["model"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    run = config["run"]
    export = config["export"]

    if config.get("config_role") != "controlled":
        raise AssertionError(f"{relative_path}: expected controlled config_role")
    if dataset.get("name") != "medium1024_gapA_full1024_v2":
        raise AssertionError(f"{relative_path}: expected medium1024 Gap-A dataset")
    if dataset.get("split_map_path") != DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP:
        raise AssertionError(f"{relative_path}: unexpected split_map_path")
    if not (REPO_ROOT / DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP).exists():
        raise AssertionError("default stratified split map does not exist")
    if model.get("node_latent_size") != 128 or model.get("edge_latent_size") != 128:
        raise AssertionError(f"{relative_path}: expected node=edge=128")
    if model.get("processor_steps") != 6:
        raise AssertionError(f"{relative_path}: expected processor_steps=6")
    if model.get("mlp_hidden_layers") != 2:
        raise AssertionError(f"{relative_path}: expected mlp_hidden_layers=2")
    if run.get("epochs") != 400:
        raise AssertionError(f"{relative_path}: expected epochs=400")
    for field in ("batch_size", "validation_batch_size", "prediction_batch_size"):
        if run.get(field) != 96:
            raise AssertionError(f"{relative_path}: expected run.{field}=96")
    if optimizer.get("name") != "adamw":
        raise AssertionError(f"{relative_path}: expected optimizer.name=adamw")
    if optimizer.get("lr") != 3.0e-4:
        raise AssertionError(f"{relative_path}: expected lr=3e-4")
    if optimizer.get("lr_schedule") != "warmup_cosine":
        raise AssertionError(f"{relative_path}: expected warmup_cosine schedule")
    if optimizer.get("warmup_epochs") != 10:
        raise AssertionError(f"{relative_path}: expected warmup_epochs=10")
    if optimizer.get("min_lr") != 1.0e-6:
        raise AssertionError(f"{relative_path}: expected min_lr=1e-6")
    if optimizer.get("seed") != expected["seed"]:
        raise AssertionError(f"{relative_path}: unexpected optimizer.seed")
    if optimizer.get("weight_decay") != expected["weight_decay"]:
        raise AssertionError(f"{relative_path}: unexpected weight_decay")
    if optimizer.get("gradient_clip_norm") != 1.0:
        raise AssertionError(f"{relative_path}: expected gradient_clip_norm=1.0")
    if loss.get("mode") != expected["loss_mode"]:
        raise AssertionError(f"{relative_path}: unexpected loss.mode")
    if loss.get("background_bias_weight") != expected["background_bias_weight"]:
        raise AssertionError(f"{relative_path}: unexpected background_bias_weight")
    if loss.get("background_over_weight") != expected["background_over_weight"]:
        raise AssertionError(f"{relative_path}: unexpected background_over_weight")
    if export.get("save_final_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_final_predictions must be true")
    if export.get("save_best_predictions") is not True:
        raise AssertionError(f"{relative_path}: save_best_predictions must be true")

    for flag in (
        "--split-map",
        "--epochs",
        "--optimizer",
        "--lr-schedule",
        "--weight-decay",
        "--gradient-clip-norm",
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
    split_index = command.index("--split-map") + 1
    if command[split_index] != DEFAULT_MEDIUM1024_GAPA_SPLIT_MAP:
        raise AssertionError(f"{relative_path}: command must use latest stratified split map")


if __name__ == "__main__":
    main()
