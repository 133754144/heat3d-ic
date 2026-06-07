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


def _b88_seed_stability_path(
    *,
    optimizer_name: str,
    policy: str,
    variant: str,
    model_seed: int,
    graph_seed: int,
    lr_tag: str,
    warmup_epochs: int,
    min_lr_tag: str,
    weight_decay_tag: str,
) -> Path:
    return REPO_DIR / (
        "configs/heat3d_v2/"
        f"frozen_v1_e400_{optimizer_name}_latent96_s6_mlp2_B88_sample_shuffle_"
        f"{policy}_{variant}_model_seed{model_seed}_batchbuild0_batchorder0_"
        f"graphseed{graph_seed}_{lr_tag}_warmup{warmup_epochs}_{min_lr_tag}_"
        f"{weight_decay_tag}.yaml"
    )


def _seed_stability_spec(
    *,
    variant: str,
    policy: str,
    model_seed: int,
    graph_seed: int = 0,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    lr_tag: str = "lr3e-4",
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    min_lr_tag: str = "minlr1e-6",
    weight_decay: float = 1e-4,
    weight_decay_tag: str = "wd1e-4",
) -> dict[str, object]:
    return {
        "path": _b88_seed_stability_path(
            optimizer_name=optimizer_name,
            policy=policy,
            variant=variant,
            model_seed=model_seed,
            graph_seed=graph_seed,
            lr_tag=lr_tag,
            warmup_epochs=warmup_epochs,
            min_lr_tag=min_lr_tag,
            weight_decay_tag=weight_decay_tag,
        ),
        "variant": variant,
        "policy": policy,
        "model_seed": model_seed,
        "graph_seed": graph_seed,
        "optimizer_name": optimizer_name,
        "lr": lr,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
    }


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

B88_SEED_STABILITY_CONFIGS = [
    *[
        _seed_stability_spec(
            variant=f"A{index}",
            policy="nearest_repair",
            model_seed=model_seed,
        )
        for index, model_seed in enumerate(range(3, 8), start=1)
    ],
    *[
        _seed_stability_spec(
            variant=f"B{index}",
            policy="discrete_radius",
            model_seed=model_seed,
        )
        for index, model_seed in enumerate(range(3, 8), start=1)
    ],
    _seed_stability_spec(
        variant="C1",
        policy="nearest_repair",
        model_seed=1,
        warmup_epochs=50,
    ),
    _seed_stability_spec(
        variant="C2",
        policy="nearest_repair",
        model_seed=1,
        warmup_epochs=100,
    ),
    _seed_stability_spec(
        variant="C3",
        policy="nearest_repair",
        model_seed=1,
        min_lr=1e-5,
        min_lr_tag="minlr1e-5",
    ),
    _seed_stability_spec(
        variant="C4",
        policy="nearest_repair",
        model_seed=1,
        min_lr=3e-5,
        min_lr_tag="minlr3e-5",
    ),
    _seed_stability_spec(
        variant="C5",
        policy="nearest_repair",
        model_seed=1,
        lr=1e-4,
        lr_tag="lr1e-4",
    ),
    _seed_stability_spec(
        variant="C6",
        policy="nearest_repair",
        model_seed=1,
        lr=1e-4,
        lr_tag="lr1e-4",
        warmup_epochs=50,
    ),
    _seed_stability_spec(
        variant="D1",
        policy="nearest_repair",
        model_seed=1,
        weight_decay=0.0,
        weight_decay_tag="wd0",
    ),
    _seed_stability_spec(
        variant="D2",
        policy="nearest_repair",
        model_seed=1,
        weight_decay=1e-5,
        weight_decay_tag="wd1e-5",
    ),
    _seed_stability_spec(
        variant="D3",
        policy="nearest_repair",
        model_seed=1,
        optimizer_name="adam",
        weight_decay=0.0,
        weight_decay_tag="wd0",
    ),
    _seed_stability_spec(
        variant="G1",
        policy="nearest_repair",
        model_seed=1,
        graph_seed=1,
    ),
    _seed_stability_spec(
        variant="G3",
        policy="nearest_repair",
        model_seed=0,
        graph_seed=1,
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


def _assert_float_close(actual: object, expected: float, label: str) -> None:
    try:
        actual_float = float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label}: expected float {expected!r}, got {actual!r}") from exc
    if abs(actual_float - expected) > max(1e-12, abs(expected) * 1e-9):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual_float!r}")


def _assert_b88_sample_shuffle_config(
    path: Path,
    policy: str,
    model_seed: int,
    *,
    variant: str | None = None,
    graph_seed: int = 0,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
) -> None:
    config = load_v2_config(path)
    command = build_training_command(config)
    optimizer = config["optimizer"]
    run = config["run"]
    model = config["model"]
    loss = config["loss"]
    graph = config["graph"]
    summary = summarize_v2_config(config)

    _assert_flag(command, "--epochs", "400")
    _assert_flag(command, "--batch-plan", "sample_shuffle")
    _assert_flag(command, "--batch-build-seed", "0")
    _assert_flag(command, "--batch-size", "88")
    _assert_flag(command, "--validation-batch-size", "88")
    _assert_flag(command, "--prediction-batch-size", "88")
    _assert_flag(command, "--model-seed", str(model_seed))
    _assert_flag(command, "--batch-order-seed", "0")
    _assert_flag(command, "--graph-seed", str(graph_seed))
    _assert_flag(command, "--optimizer", optimizer_name)
    _assert_flag(command, "--warmup-epochs", str(warmup_epochs))
    _assert_float_close(_flag_value(command, "--lr"), lr, f"{path}: --lr")
    _assert_float_close(_flag_value(command, "--min-lr"), min_lr, f"{path}: --min-lr")
    _assert_float_close(
        _flag_value(command, "--weight-decay"),
        weight_decay,
        f"{path}: --weight-decay",
    )
    if int(run["batch_build_seed"]) != 0:
        raise AssertionError(f"{path}: expected run.batch_build_seed=0")
    if int(run["epochs"]) != 400 or int(run["batch_size"]) != 88:
        raise AssertionError(f"{path}: expected e400 B88 run settings")
    if int(optimizer["batch_order_seed"]) != 0 or int(optimizer["graph_seed"]) != graph_seed:
        raise AssertionError(
            f"{path}: expected optimizer batch_order_seed=0 and graph_seed={graph_seed}"
        )
    if int(optimizer["model_seed"]) != model_seed:
        raise AssertionError(f"{path}: expected optimizer.model_seed={model_seed}")
    if optimizer["name"] != optimizer_name:
        raise AssertionError(f"{path}: expected optimizer.name={optimizer_name}")
    _assert_float_close(optimizer["lr"], lr, f"{path}: optimizer.lr")
    if int(optimizer["warmup_epochs"]) != warmup_epochs:
        raise AssertionError(f"{path}: expected warmup_epochs={warmup_epochs}")
    _assert_float_close(optimizer["min_lr"], min_lr, f"{path}: optimizer.min_lr")
    _assert_float_close(
        optimizer["weight_decay"],
        weight_decay,
        f"{path}: optimizer.weight_decay",
    )
    if (
        int(model["node_latent_size"]) != 96
        or int(model["edge_latent_size"]) != 96
        or int(model["processor_steps"]) != 6
        or int(model["mlp_hidden_layers"]) != 2
    ):
        raise AssertionError(f"{path}: expected latent96_s6_mlp2 model settings")
    if loss["mode"] != "mse":
        raise AssertionError(f"{path}: expected loss.mode=mse")
    if summary.get("batch_plan") != "sample_shuffle" or summary.get("batch_build_seed") != 0:
        raise AssertionError(f"{path}: summary missing B88 batch fields")
    if (
        summary.get("model_seed") != model_seed
        or summary.get("batch_order_seed") != 0
        or summary.get("graph_seed") != graph_seed
    ):
        raise AssertionError(f"{path}: summary missing seed fields")
    if variant is not None:
        metadata = config.get("metadata", {})
        if metadata.get("variant_label") != variant:
            raise AssertionError(f"{path}: expected metadata.variant_label={variant}")

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

    if len(B88_SEED_STABILITY_CONFIGS) != 21:
        raise AssertionError(
            f"expected 21 B88 seed-stability configs, got {len(B88_SEED_STABILITY_CONFIGS)}"
        )
    seen_variants = set()
    for spec in B88_SEED_STABILITY_CONFIGS:
        path = spec["path"]
        if not isinstance(path, Path):
            raise AssertionError(f"invalid path spec: {spec}")
        variant = str(spec["variant"])
        if variant in seen_variants:
            raise AssertionError(f"duplicate B88 seed-stability variant: {variant}")
        seen_variants.add(variant)
        _assert_b88_sample_shuffle_config(
            path,
            str(spec["policy"]),
            int(spec["model_seed"]),
            variant=variant,
            graph_seed=int(spec["graph_seed"]),
            optimizer_name=str(spec["optimizer_name"]),
            lr=float(spec["lr"]),
            warmup_epochs=int(spec["warmup_epochs"]),
            min_lr=float(spec["min_lr"]),
            weight_decay=float(spec["weight_decay"]),
        )
        print(
            path.relative_to(REPO_DIR),
            "variant",
            variant,
            "policy",
            spec["policy"],
            "optimizer",
            spec["optimizer_name"],
            "model_seed",
            spec["model_seed"],
            "graph_seed",
            spec["graph_seed"],
            "lr",
            spec["lr"],
            "warmup",
            spec["warmup_epochs"],
            "min_lr",
            spec["min_lr"],
            "weight_decay",
            spec["weight_decay"],
        )

    print("seed decoupling config smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
