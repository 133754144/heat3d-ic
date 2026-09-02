"""Machine guard for the V7 G1 H2 native closeout governance boundary.

Gate A and Gate B are geometry-only.  This module makes that boundary
explicit: their input paths cannot name result-bearing artifacts, and their
output must pass the existing geometry-only sanitizer before it is written.
The guard is intentionally independent of the model/evaluation runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rigno.heat3d_v7_h2_geometry_sanitizer import (
    BANNED_TOKENS,
    assert_geometry_only_output,
)


GATE_AB_BANNED_INPUT_TOKENS = frozenset(
    set(BANNED_TOKENS)
    | {
        "checkpoint",
        "checkpoints",
        "forward",
        "model",
        "optimizer",
        "training",
        "truth",
        "label",
        "labels",
        "test_iid",
        "sealed",
        "result",
        "results",
        "output",
        "pred",
    }
)

GATE_AB_ALLOWED_INPUT_ROLES = frozenset({"geometry", "support", "config"})


def _path_tokens(value: str | Path) -> set[str]:
    normalized = (
        str(value)
        .lower()
        .replace("\\", "/")
        .replace("/", "_")
        .replace(".", "_")
    )
    return {token for token in normalized.replace("-", "_").split("_") if token}


def assert_gate_ab_input_paths(
    paths_by_role: Mapping[str, Sequence[str | Path]],
) -> None:
    """Reject Gate A/B paths that could carry model/result information.

    The caller must classify each path as ``geometry``, ``support`` or
    ``config``.  A co-located archive is permitted only when the calling code
    explicitly reads geometry datasets and does not open label datasets.
    """

    unknown_roles = set(paths_by_role) - GATE_AB_ALLOWED_INPUT_ROLES
    if unknown_roles:
        raise ValueError(f"Gate A/B input role is not geometry/support/config: {sorted(unknown_roles)}")
    for role, paths in paths_by_role.items():
        for value in paths:
            tokens = _path_tokens(value)
            banned = tokens & GATE_AB_BANNED_INPUT_TOKENS
            if banned:
                raise ValueError(
                    f"Gate A/B {role} input path contains a result-bearing token: {sorted(banned)}"
                )
            if Path(value).suffix.lower() in {".pkl", ".pickle", ".npz"} and role != "support":
                raise ValueError(f"Gate A/B {role} input cannot be a model/result binary: {value}")


def assert_gate_ab_output(payload: Mapping[str, Any]) -> None:
    """Enforce the allowlisted geometry-only Gate A/B output schema."""

    assert_geometry_only_output(payload)
    if "sample_id" not in payload and "provenance" not in payload:
        raise ValueError("Gate A/B output must identify its geometry sample/provenance")
