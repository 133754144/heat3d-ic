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
    NORMALIZATION_PROFILE_LEGACY_ZSCORE,
    NORMALIZATION_PROFILE_SEMANTIC_V1,
    NORMALIZATION_PROFILES,
)


DEFAULT_NORMALIZATION_PROFILE = NORMALIZATION_PROFILE_SEMANTIC_V1
DEFAULT_CONDITION_FEATURE_TRANSFORM = CONDITION_FEATURE_TRANSFORM_SEMANTIC_FULL


def main() -> int:
    profile, condition_feature_transform = _pop_v4_profile_args(
        sys.argv,
        default_profile=DEFAULT_NORMALIZATION_PROFILE,
        default_condition_feature_transform=DEFAULT_CONDITION_FEATURE_TRANSFORM,
    )
    _install_profile_hooks(profile, condition_feature_transform)
    return legacy_runner.main()


def _pop_v4_profile_args(
    argv: list[str],
    *,
    default_profile: str,
    default_condition_feature_transform: str,
) -> tuple[str, str]:
    profile = default_profile
    condition_feature_transform = default_condition_feature_transform
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
    argv[:] = cleaned
    return profile, condition_feature_transform


def _install_profile_hooks(profile: str, condition_feature_transform: str) -> None:
    if profile == NORMALIZATION_PROFILE_LEGACY_ZSCORE:
        return

    def _train_only_stats(examples: list[Any]) -> dict[str, Any]:
        return training_normalization_stats(
            examples,
            normalization_profile=profile,
            condition_feature_transform=condition_feature_transform,
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
        payload["runner"] = "scripts/run_heat3d_v4_controlled_training.py"
        return payload

    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name in {"run_config.json", "loss_summary.json"}:
            payload = dict(payload)
            payload.setdefault("normalization_profile", profile)
            payload.setdefault("condition_feature_transform", condition_feature_transform)
            payload.setdefault("runner", "scripts/run_heat3d_v4_controlled_training.py")
        original_write_json(path, payload)

    legacy_runner._train_only_stats = _train_only_stats
    legacy_runner._stats_payload = _stats_payload
    legacy_runner._checkpoint_run_metadata = _checkpoint_run_metadata
    legacy_runner._write_json = _write_json


if __name__ == "__main__":
    raise SystemExit(main())
