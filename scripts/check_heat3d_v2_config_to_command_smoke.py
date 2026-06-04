#!/usr/bin/env python3
"""Smoke-check v2 config to v1 command dry-run generation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import load_v2_config  # noqa: E402
from rigno.heat3d_v2_runner_command import (  # noqa: E402
    build_v2_command_plan,
    summarize_command_plan,
)


CONFIG_PATHS = (
    Path("configs/heat3d_v2/smoke_minimal.yaml"),
    Path("configs/heat3d_v2/medium1024_gapA_controlled.yaml"),
)
REQUIRED_CONTROLLED_KINDS = {
    "baseline_comparison",
    "error_bins",
    "run_summary",
    "condition_diagnostics",
    "field_shape_diagnostics",
}


def main() -> int:
    for relative_path in CONFIG_PATHS:
        config = load_v2_config(REPO_ROOT / relative_path)
        plan = build_v2_command_plan(config, python_executable="python3")
        print(f"config path: {relative_path}")
        print(summarize_command_plan(plan))
        _print_field_summary(plan)
        _assert_dry_run_only(plan)

        if relative_path.name == "medium1024_gapA_controlled.yaml":
            _assert_controlled_final_best_diagnostics(plan)
            _assert_controlled_model_capacity(plan)

    print("Heat3D v2 config-to-command smoke passed.")
    return 0


def _print_field_summary(plan: dict) -> None:
    mapped = plan["mapped_fields"]
    unmapped = plan["unmapped_fields"]
    print("mapped fields sample:")
    for item in mapped[:8]:
        print(f"  {item['field']} -> {item['target']}")
    print("unmapped fields:")
    for item in unmapped:
        print(f"  {item['field']}: {item['reason']}")
    if plan["warnings"]:
        print("warnings:")
        for warning in plan["warnings"]:
            print(f"  {warning}")


def _assert_dry_run_only(plan: dict) -> None:
    note = plan.get("non_execution_note", "")
    if "not executed" not in note:
        raise AssertionError("command plan must explicitly say commands are not executed")


def _assert_controlled_final_best_diagnostics(plan: dict) -> None:
    groups = {}
    for entry in plan["diagnostics_commands"]:
        groups.setdefault(entry["prediction_label"], set()).add(entry["kind"])

    for label in ("final", "best"):
        kinds = groups.get(label)
        if kinds != REQUIRED_CONTROLLED_KINDS:
            raise AssertionError(
                f"controlled config missing {label} diagnostics: {kinds}"
            )

    expected_count = len(REQUIRED_CONTROLLED_KINDS) * 2
    actual_count = len(plan["diagnostics_commands"])
    if actual_count != expected_count:
        raise AssertionError(
            f"controlled config expected {expected_count} diagnostics commands, "
            f"got {actual_count}"
        )


def _assert_controlled_model_capacity(plan: dict) -> None:
    command = plan["training_command"]
    _assert_option(command, "--node-latent-size", "64")
    _assert_option(command, "--edge-latent-size", "64")
    _assert_option(command, "--processor-steps", "4")
    _assert_option(command, "--mlp-hidden-layers", "2")


def _assert_option(command: list[str], flag: str, expected: str) -> None:
    if flag not in command:
        raise AssertionError(f"command missing {flag}")
    index = command.index(flag)
    try:
        actual = command[index + 1]
    except IndexError as exc:
        raise AssertionError(f"command flag {flag} has no value") from exc
    if actual != expected:
        raise AssertionError(f"{flag} expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
