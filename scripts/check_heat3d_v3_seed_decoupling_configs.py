"""Smoke checks for Heat3D v3 seed-decoupling YAML and command generation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v2_config import load_v2_config, summarize_v2_config, validate_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import build_training_command  # noqa: E402


LEGACY_CONFIG = REPO_DIR / (
    "configs/heat3d_v2/"
    "frozen_v1_e400_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_seed0.yaml"
)

NEW_CONFIGS = [
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed0_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed1_batchseed0_graphseed0_seed1.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed2_batchseed0_graphseed0_seed2.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed1_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e020_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed0_batchseed2_graphseed0_seed0.yaml"
    ),
    REPO_DIR / (
        "configs/heat3d_v2/"
        "frozen_v1_e400_adamw_latent96_s6_mlp2_B96_base_mse_warmup_cosine_nearest_repair_"
        "model_seed1_batchseed0_graphseed0_seed1.yaml"
    ),
]

B88_SAMPLE_SHUFFLE_CONFIGS = [
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_legacy_model_seed0_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "legacy",
        0,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_nearest_repair_model_seed0_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "nearest_repair",
        0,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_nearest_repair_model_seed1_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "nearest_repair",
        1,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_nearest_repair_model_seed2_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "nearest_repair",
        2,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_discrete_radius_model_seed0_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "discrete_radius",
        0,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_discrete_radius_model_seed1_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "discrete_radius",
        1,
    ),
    (
        REPO_DIR / (
            "configs/heat3d_v2/"
            "frozen_v1_e400_adamw_latent96_s6_mlp2_B88_sample_shuffle_base_mse_"
            "warmup_cosine_discrete_radius_model_seed2_batchbuild0_batchorder0_graphseed0.yaml"
        ),
        "discrete_radius",
        2,
    ),
]


def _flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise AssertionError(f"{flag} has no value")
    return command[index + 1]


def _assert_flag(command: list[str], flag: str, expected: str) -> None:
    actual = _flag_value(command, flag)
    if actual != expected:
        raise AssertionError(f"{flag}: expected {expected!r}, got {actual!r}")


def _assert_b88_sample_shuffle_config(path: Path, policy: str, model_seed: int) -> None:
    config = load_v2_config(path)
    command = build_training_command(config)
    optimizer = config["optimizer"]
    run = config["run"]
    graph = config["graph"]
    summary = summarize_v2_config(config)

    _assert_flag(command, "--batch-plan", "sample_shuffle")
    _assert_flag(command, "--batch-build-seed", "0")
    _assert_flag(command, "--batch-size", "88")
    _assert_flag(command, "--validation-batch-size", "88")
    _assert_flag(command, "--prediction-batch-size", "88")
    _assert_flag(command, "--model-seed", str(model_seed))
    _assert_flag(command, "--batch-order-seed", "0")
    _assert_flag(command, "--graph-seed", "0")
    if int(run["batch_build_seed"]) != 0:
        raise AssertionError(f"{path}: expected run.batch_build_seed=0")
    if int(optimizer["batch_order_seed"]) != 0 or int(optimizer["graph_seed"]) != 0:
        raise AssertionError(f"{path}: expected optimizer batch_order_seed=0 and graph_seed=0")
    if int(optimizer["model_seed"]) != model_seed:
        raise AssertionError(f"{path}: expected optimizer.model_seed={model_seed}")
    if summary.get("batch_plan") != "sample_shuffle" or summary.get("batch_build_seed") != 0:
        raise AssertionError(f"{path}: summary missing B88 batch fields")
    if summary.get("model_seed") != model_seed or summary.get("batch_order_seed") != 0 or summary.get("graph_seed") != 0:
        raise AssertionError(f"{path}: summary missing seed fields")

    expected_graph = {
        "legacy": ("legacy_kdtree_mean4", "none"),
        "nearest_repair": ("legacy_kdtree_mean4", "nearest_rnode"),
        "discrete_radius": ("discrete_physical_coverage", "none"),
    }[policy]
    if (graph["radius_policy"], graph["coverage_repair_policy"]) != expected_graph:
        raise AssertionError(
            f"{path}: unexpected graph policy "
            f"{graph['radius_policy']}/{graph['coverage_repair_policy']}"
        )
    if policy == "discrete_radius" and graph["coverage_repair_policy"] != "none":
        raise AssertionError(f"{path}: discrete_radius must remain pure discrete without nearest repair")


def main() -> int:
    legacy = load_v2_config(LEGACY_CONFIG)
    legacy_command = build_training_command(legacy)
    if "--model-seed" in legacy_command or "--batch-order-seed" in legacy_command or "--graph-seed" in legacy_command:
        raise AssertionError("legacy config unexpectedly emits explicit seed-decoupling flags")
    _assert_flag(legacy_command, "--seed", "0")

    synthetic = copy.deepcopy(legacy)
    synthetic["optimizer"]["model_seed"] = 1
    synthetic["optimizer"]["batch_order_seed"] = 0
    synthetic["optimizer"]["graph_seed"] = 0
    validate_v2_config(synthetic)
    synthetic_command = build_training_command(synthetic)
    _assert_flag(synthetic_command, "--seed", "0")
    _assert_flag(synthetic_command, "--model-seed", "1")
    _assert_flag(synthetic_command, "--batch-order-seed", "0")
    _assert_flag(synthetic_command, "--graph-seed", "0")

    for path in NEW_CONFIGS:
        config = load_v2_config(path)
        command = build_training_command(config)
        optimizer = config["optimizer"]
        _assert_flag(command, "--seed", str(optimizer["seed"]))
        _assert_flag(command, "--model-seed", str(optimizer["model_seed"]))
        _assert_flag(command, "--batch-order-seed", str(optimizer["batch_order_seed"]))
        _assert_flag(command, "--graph-seed", str(optimizer["graph_seed"]))
        print(
            path.relative_to(REPO_DIR),
            "seed",
            optimizer["seed"],
            "model_seed",
            optimizer["model_seed"],
            "batch_order_seed",
            optimizer["batch_order_seed"],
            "graph_seed",
            optimizer["graph_seed"],
            "epochs",
            config["run"]["epochs"],
        )

    for path, policy, model_seed in B88_SAMPLE_SHUFFLE_CONFIGS:
        _assert_b88_sample_shuffle_config(path, policy, model_seed)
        print(
            path.relative_to(REPO_DIR),
            "policy",
            policy,
            "model_seed",
            model_seed,
            "batch_plan",
            "sample_shuffle",
            "batch_size",
            88,
        )

    print("seed decoupling config smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
