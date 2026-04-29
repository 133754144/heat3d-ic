"""Utilities for validating the Heat3D v1 parameter registry.

This module is intentionally lightweight and standard-library only. The
registry is planning / smoke infrastructure; it does not drive generation yet.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_SOURCE_CATEGORIES = {
    "literature_backed",
    "provisional_engineering_assumption",
    "requires_user_confirmation",
}

ALLOWED_USES = {
    "smoke",
    "diagnostic",
    "benchmark_candidate",
    "deprecated",
}


@dataclass(frozen=True)
class RegistryValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a JSON registry from disk."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    if not isinstance(registry, dict):
        raise ValueError(f"Registry root must be an object: {registry_path}")
    return registry


def validate_registry(registry: dict[str, Any]) -> RegistryValidationResult:
    """Validate registry structure and parameter-source tagging."""

    errors: list[str] = []
    warnings: list[str] = []

    if not registry.get("registry_version"):
        errors.append("top-level registry_version is required")

    groups = registry.get("parameter_groups")
    if not isinstance(groups, dict):
        errors.append("top-level parameter_groups object is required")
        return RegistryValidationResult(tuple(errors), tuple(warnings))

    if not groups:
        errors.append("parameter_groups must not be empty")

    declared_sources = set(registry.get("allowed_source_categories", []))
    if declared_sources and declared_sources != ALLOWED_SOURCE_CATEGORIES:
        errors.append(
            "allowed_source_categories must exactly match "
            f"{sorted(ALLOWED_SOURCE_CATEGORIES)}"
        )

    declared_uses = set(registry.get("allowed_uses", []))
    if declared_uses and declared_uses != ALLOWED_USES:
        errors.append(f"allowed_uses must exactly match {sorted(ALLOWED_USES)}")

    for group_name, group in groups.items():
        if not isinstance(group, dict):
            errors.append(f"{group_name}: group must be an object")
            continue

        planned_empty = bool(group.get("planned_empty", False))
        entries = group.get("entries")

        if planned_empty:
            if not group.get("planned_empty_reason"):
                errors.append(f"{group_name}: planned_empty requires reason")
            continue

        if not isinstance(entries, list) or not entries:
            errors.append(
                f"{group_name}: entries must be non-empty unless planned_empty is true"
            )
            continue

        for index, entry in enumerate(entries):
            location = f"{group_name}[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{location}: entry must be an object")
                continue

            key = entry.get("key") or entry.get("name")
            if not key:
                errors.append(f"{location}: key or name is required")

            source_category = entry.get("source_category")
            if source_category not in ALLOWED_SOURCE_CATEGORIES:
                errors.append(
                    f"{location}: invalid source_category {source_category!r}"
                )

            allowed_use = entry.get("allowed_use")
            if allowed_use not in ALLOWED_USES:
                errors.append(f"{location}: invalid allowed_use {allowed_use!r}")

            unresolved = bool(entry.get("unresolved", False))
            value = entry.get("value")

            if unresolved:
                if not entry.get("unresolved_reason"):
                    errors.append(f"{location}: unresolved entry requires reason")
            else:
                if value is None:
                    errors.append(f"{location}: resolved entry requires value")
                if not entry.get("unit"):
                    errors.append(f"{location}: resolved entry requires unit")

            if source_category == "literature_backed":
                if not (entry.get("citation") or entry.get("reference")):
                    errors.append(
                        f"{location}: literature_backed entry requires citation "
                        "or reference"
                    )

            if source_category in {None, "unknown", "implicit", "untagged"}:
                errors.append(f"{location}: untagged source category is forbidden")

            if allowed_use == "benchmark_candidate":
                warnings.append(
                    f"{location}: benchmark_candidate is not a formal benchmark"
                )

    return RegistryValidationResult(tuple(errors), tuple(warnings))


def summarize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Return simple counts and unresolved/provisional item lists."""

    groups = registry.get("parameter_groups", {})
    source_counts: Counter[str] = Counter()
    allowed_use_counts: Counter[str] = Counter()
    requires_user_confirmation: list[str] = []
    provisional: list[str] = []
    unresolved: list[str] = []
    group_entry_counts: dict[str, int] = {}
    entries_by_source: defaultdict[str, list[str]] = defaultdict(list)

    for group_name, group in groups.items():
        entries = group.get("entries", []) if isinstance(group, dict) else []
        group_entry_counts[group_name] = len(entries)
        for entry in entries:
            key = entry.get("key") or entry.get("name") or "<missing-key>"
            qualified = f"{group_name}.{key}"
            source_category = entry.get("source_category", "<missing>")
            allowed_use = entry.get("allowed_use", "<missing>")
            source_counts[source_category] += 1
            allowed_use_counts[allowed_use] += 1
            entries_by_source[source_category].append(qualified)
            if source_category == "requires_user_confirmation":
                requires_user_confirmation.append(qualified)
            if source_category == "provisional_engineering_assumption":
                provisional.append(qualified)
            if entry.get("unresolved", False):
                unresolved.append(qualified)

    return {
        "registry_version": registry.get("registry_version"),
        "group_count": len(groups),
        "group_entry_counts": group_entry_counts,
        "source_category_counts": dict(sorted(source_counts.items())),
        "allowed_use_counts": dict(sorted(allowed_use_counts.items())),
        "requires_user_confirmation": requires_user_confirmation,
        "provisional_engineering_assumption": provisional,
        "unresolved": unresolved,
        "entries_by_source": dict(entries_by_source),
    }
