#!/usr/bin/env python3
"""V4 controlled runner wrapper for semantic normalization experiments.

This wrapper preserves the legacy V1 medium runner as the default baseline
runner. It only swaps the train-only normalization helper when the selected
profile is `semantic_normalization_v1`.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as legacy_runner  # noqa: E402
from rigno.heat3d_v1_normalization import (  # noqa: E402
    CONDITION_FEATURE_TRANSFORMS,
    CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL,
    training_normalization_stats,
)
from rigno.heat3d_v1_training_semantics import (  # noqa: E402
    COORD_POLICIES,
    COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    EXTENT_FEATURE_POLICIES,
    EXTENT_FEATURE_POLICY_NONE,
    INPUT_FEATURE_SCHEMAS,
    INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS,
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
    NORMALIZATION_PROFILES,
    build_configured_zero_delta_bridge,
)


DEFAULT_NORMALIZATION_PROFILE = NORMALIZATION_PROFILE_SEMANTIC_V1
DEFAULT_CONDITION_FEATURE_TRANSFORM = CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL
DEFAULT_INPUT_FEATURE_SCHEMA = INPUT_FEATURE_SCHEMA_LEGACY_BC_FLAGS
DEFAULT_COORD_POLICY = COORD_POLICY_TRAIN_MINMAX_UNIT_BOX
DEFAULT_EXTENT_FEATURE_POLICY = EXTENT_FEATURE_POLICY_NONE


def main() -> int:
    (
        profile,
        condition_feature_transform,
        input_feature_schema,
        coord_policy,
        extent_feature_policy,
    ) = _pop_v4_profile_args(
        sys.argv,
        default_profile=DEFAULT_NORMALIZATION_PROFILE,
        default_condition_feature_transform=DEFAULT_CONDITION_FEATURE_TRANSFORM,
        default_input_feature_schema=DEFAULT_INPUT_FEATURE_SCHEMA,
        default_coord_policy=DEFAULT_COORD_POLICY,
        default_extent_feature_policy=DEFAULT_EXTENT_FEATURE_POLICY,
    )
    _install_profile_hooks(
        profile,
        condition_feature_transform,
        input_feature_schema,
        coord_policy,
        extent_feature_policy,
    )
    return legacy_runner.main()


def _pop_v4_profile_args(
    argv: list[str],
    *,
    default_profile: str,
    default_condition_feature_transform: str,
    default_input_feature_schema: str,
    default_coord_policy: str,
    default_extent_feature_policy: str,
) -> tuple[str, str, str, str, str]:
    profile = default_profile
    condition_feature_transform = default_condition_feature_transform
    input_feature_schema = default_input_feature_schema
    coord_policy = default_coord_policy
    extent_feature_policy = default_extent_feature_policy
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--normalization-profile":
            if index + 1 >= len(argv):
                raise ValueError("--normalization-profile requires a value")
            profile = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--normalization-profile="):
            profile = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--condition-feature-transform":
            if index + 1 >= len(argv):
                raise ValueError("--condition-feature-transform requires a value")
            condition_feature_transform = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--condition-feature-transform="):
            condition_feature_transform = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--input-feature-schema":
            if index + 1 >= len(argv):
                raise ValueError("--input-feature-schema requires a value")
            input_feature_schema = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--input-feature-schema="):
            input_feature_schema = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--coord-policy":
            if index + 1 >= len(argv):
                raise ValueError("--coord-policy requires a value")
            coord_policy = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--coord-policy="):
            coord_policy = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--extent-feature-policy":
            if index + 1 >= len(argv):
                raise ValueError("--extent-feature-policy requires a value")
            extent_feature_policy = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--extent-feature-policy="):
            extent_feature_policy = arg.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(arg)
        index += 1

    if profile not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"--normalization-profile must be one of {NORMALIZATION_PROFILES}, "
            f"found {profile!r}"
        )
    if condition_feature_transform not in CONDITION_FEATURE_TRANSFORMS:
        raise ValueError(
            "--condition-feature-transform must be one of "
            f"{CONDITION_FEATURE_TRANSFORMS}, found {condition_feature_transform!r}"
        )
    if input_feature_schema not in INPUT_FEATURE_SCHEMAS:
        raise ValueError(
            f"--input-feature-schema must be one of {INPUT_FEATURE_SCHEMAS}, "
            f"found {input_feature_schema!r}"
        )
    if coord_policy not in COORD_POLICIES:
        raise ValueError(
            f"--coord-policy must be one of {COORD_POLICIES}, found {coord_policy!r}"
        )
    if extent_feature_policy not in EXTENT_FEATURE_POLICIES:
        raise ValueError(
            "--extent-feature-policy must be one of "
            f"{EXTENT_FEATURE_POLICIES}, found {extent_feature_policy!r}"
        )
    argv[:] = cleaned
    return (
        profile,
        condition_feature_transform,
        input_feature_schema,
        coord_policy,
        extent_feature_policy,
    )


def _install_profile_hooks(
    profile: str,
    condition_feature_transform: str,
    input_feature_schema: str,
    coord_policy: str,
    extent_feature_policy: str,
) -> None:
    legacy_runner.HIDE_MISSING_STRESS_COMPACT_LOG = True
    if (
        profile == NORMALIZATION_PROFILE_LEGACY_ZSCORE
        and input_feature_schema == DEFAULT_INPUT_FEATURE_SCHEMA
        and coord_policy == DEFAULT_COORD_POLICY
        and extent_feature_policy == DEFAULT_EXTENT_FEATURE_POLICY
    ):
        return

    def _bridge_for(example: Any) -> Any:
        return build_configured_zero_delta_bridge(
            example,
            input_feature_schema=input_feature_schema,
            coord_policy=coord_policy,
            extent_feature_policy=extent_feature_policy,
        )

    def _train_only_stats(examples: list[Any]) -> dict[str, Any]:
        return training_normalization_stats(
            examples,
            normalization_profile=profile,
            condition_feature_transform=condition_feature_transform,
            input_feature_schema=input_feature_schema,
            coord_policy=coord_policy,
            extent_feature_policy=extent_feature_policy,
        )

    original_stats_payload = legacy_runner._stats_payload
    original_checkpoint_run_metadata = legacy_runner._checkpoint_run_metadata
    original_write_json = legacy_runner._write_json

    def _stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
        payload = original_stats_payload(stats)
        payload["normalization_profile"] = stats.get("normalization_profile", profile)
        payload["condition_feature_transform"] = stats.get(
            "condition_feature_transform", condition_feature_transform
        )
        payload["input_feature_schema"] = stats.get(
            "input_feature_schema", input_feature_schema
        )
        payload["coord_policy"] = stats.get("coord_policy", coord_policy)
        payload["extent_feature_policy"] = stats.get(
            "extent_feature_policy", extent_feature_policy
        )
        payload["condition_feature_transforms"] = list(
            stats.get("condition_feature_transforms", ())
        )
        for key in (
            "physical_extent_min",
            "physical_extent_max",
            "physical_extent_mean",
        ):
            if key in stats:
                payload[key] = [float(value) for value in stats[key].reshape(-1)]
        for key in ("aspect_ratio_min", "aspect_ratio_max", "aspect_ratio_mean"):
            if key in stats:
                payload[key] = float(stats[key])
        return payload

    def _checkpoint_run_metadata(**kwargs: Any) -> dict[str, Any]:
        payload = original_checkpoint_run_metadata(**kwargs)
        payload["normalization_profile"] = profile
        payload["condition_feature_transform"] = condition_feature_transform
        payload["input_feature_schema"] = input_feature_schema
        payload["coord_policy"] = coord_policy
        payload["extent_feature_policy"] = extent_feature_policy
        payload["runner"] = "scripts/run_heat3d_v4_controlled_training.py"
        return payload

    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name in {"run_config.json", "loss_summary.json"}:
            payload = dict(payload)
            payload.setdefault("normalization_profile", profile)
            payload.setdefault("condition_feature_transform", condition_feature_transform)
            payload.setdefault("input_feature_schema", input_feature_schema)
            payload.setdefault("coord_policy", coord_policy)
            payload.setdefault("extent_feature_policy", extent_feature_policy)
            payload.setdefault("runner", "scripts/run_heat3d_v4_controlled_training.py")
        original_write_json(path, payload)

    legacy_runner._bridge_for = _bridge_for
    legacy_runner._train_only_stats = _train_only_stats
    legacy_runner._stats_payload = _stats_payload
    legacy_runner._checkpoint_run_metadata = _checkpoint_run_metadata
    legacy_runner._write_json = _write_json


if __name__ == "__main__":
    raise SystemExit(main())
