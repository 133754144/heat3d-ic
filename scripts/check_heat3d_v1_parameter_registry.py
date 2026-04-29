#!/usr/bin/env python3
"""Validate and summarize the Heat3D v1 parameter registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_parameter_registry import (  # noqa: E402
    load_registry,
    summarize_registry,
    validate_registry,
)


DEFAULT_REGISTRY = (
    REPO_ROOT / "configs" / "heat3d_v1" / "parameter_registry_v1.json"
)


def _print_list(title: str, values: list[str]) -> None:
    print(f"{title}: {len(values)}")
    for value in values:
        print(f"  - {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Heat3D v1 parameter registry."
    )
    parser.add_argument(
        "registry",
        nargs="?",
        default=str(DEFAULT_REGISTRY),
        help="Path to parameter registry JSON.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    result = validate_registry(registry)
    summary = summarize_registry(registry)

    print(f"registry_path: {registry_path}")
    print(f"registry_version: {summary['registry_version']}")
    print(f"parameter_group_count: {summary['group_count']}")
    print(f"group_entry_counts: {summary['group_entry_counts']}")
    print(f"source_category_counts: {summary['source_category_counts']}")
    print(f"allowed_use_counts: {summary['allowed_use_counts']}")
    _print_list(
        "requires_user_confirmation",
        summary["requires_user_confirmation"],
    )
    _print_list(
        "provisional_engineering_assumption",
        summary["provisional_engineering_assumption"],
    )
    _print_list("unresolved_parameters", summary["unresolved"])

    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  - {error}")
        print("registry_ok: False")
        return 1

    print("registry_ok: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
