"""Machine-enforced sanitizer for V7 G1 H2 provenance recovery.

Only geometry/support/graph/dependency/provenance fields are returned.  This
module may inspect a mixed historical JSON internally, but banned fields and
their values are dropped before any caller can observe, log, or use them.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


BANNED_TOKENS = frozenset({
    "accuracy",
    "metric",
    "rmse",
    "mae",
    "temperature",
    "prediction",
    "target",
    "loss",
})

ALLOWED_TOP_LEVEL_KEYS = frozenset({
    "sample_id",
    "geometry",
    "support",
    "graph",
    "dependency",
    "provenance",
})

ALLOWED_SECTION_KEYS = {
    "geometry": frozenset({
        "coordinate_sha256",
        "coordinates_sha256",
        "normalized_coordinates_sha256",
        "rnodes_sha256",
        "domain_sha256",
        "layer_id_sha256",
        "control_volume_sha256",
        "boundary_sha256",
        "node_count",
        "support_count",
        "resolution",
        "coordinate_policy",
        "normalization_policy",
        "domain",
    }),
    "support": frozenset({
        "support_sha256",
        "support_indices_sha256",
        "support_coordinates_sha256",
        "support_count",
        "support_source",
        "support_order",
        "selection_seed",
        "selection_rule",
        "algorithm",
        "source",
        "provenance",
    }),
    "graph": frozenset({
        "config",
        "config_sha256",
        "graph_config",
        "graph_config_sha256",
        "backend",
        "discrete_graph_backend",
        "coverage_repair_policy",
        "repair_p2r",
        "repair_r2p",
        "min_physical_coverage",
        "rmesh_levels",
        "subsample_factor",
        "overlap_factor_p2r",
        "overlap_factor_r2p",
        "discrete_graph_chunk_size",
        "discrete_coverage_multiplier",
        "radius_policy",
        "radius_sha256",
        "radii_sha256",
        "raw_radius_sha256",
        "raw_p2r_sha256",
        "repair_edge_sha256",
        "repair_p2r_sha256",
        "final_p2r_sha256",
        "final_r2r_sha256",
        "final_r2r_domains_sha256",
        "p2r_count",
        "r2p_count",
        "r2r_count",
        "p2r_edge_count",
        "r2p_edge_count",
        "r2r_edge_count",
        "raw_p2r_count",
        "repair_edge_count",
        "final_p2r_count",
        "final_r2r_count",
    }),
    "dependency": frozenset({
        "python",
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "flax",
        "jraph",
        "backend",
        "platform",
        "machine",
        "runtime",
        "version",
    }),
    "provenance": frozenset({
        "commit",
        "git_commit",
        "code_sha256",
        "path",
        "source",
        "implementation",
        "route",
        "schema_version",
        "input_sha256",
    }),
}

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _normalized_key(value: Any) -> str:
    return _TOKEN_RE.sub("_", str(value).strip().lower()).strip("_")


def _contains_banned_text(value: str) -> bool:
    normalized = _normalized_key(value)
    tokens = set(filter(None, normalized.split("_")))
    return any(token in tokens or token in normalized for token in BANNED_TOKENS)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and not _contains_banned_text(value):
        return value
    return None


def _sanitize_value(value: Any, section: str, *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        allowed = ALLOWED_TOP_LEVEL_KEYS if top_level else ALLOWED_SECTION_KEYS[section]
        for raw_key, raw_value in value.items():
            key = _normalized_key(raw_key)
            if key in BANNED_TOKENS or any(token in key for token in BANNED_TOKENS):
                continue
            if key not in allowed:
                continue
            if key in ALLOWED_SECTION_KEYS:
                nested_section = key
            else:
                nested_section = section
            sanitized = _sanitize_value(raw_value, nested_section)
            if sanitized is not None and sanitized != {} and sanitized != []:
                result[key] = sanitized
        return result
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        for item in value:
            clean = _sanitize_value(item, section)
            if clean is not None:
                sanitized_items.append(clean)
        return sanitized_items
    return _safe_scalar(value)


def _assert_safe_output(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if any(token in normalized for token in BANNED_TOKENS):
                raise ValueError("sanitizer output contains a banned key")
            _assert_safe_output(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_safe_output(child)
    elif isinstance(value, str) and _contains_banned_text(value):
        raise ValueError("sanitizer output contains a banned string")


def sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an allowlisted geometry-only projection of a JSON object."""

    if not isinstance(payload, Mapping):
        raise ValueError("sanitizer input must be a JSON object")
    result: dict[str, Any] = {"schema_version": "geometry_only_sanitized_v1"}
    for raw_key, raw_value in payload.items():
        key = _normalized_key(raw_key)
        if key in BANNED_TOKENS or any(token in key for token in BANNED_TOKENS):
            continue
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            continue
        section = "provenance" if key in {"sample_id"} else key
        sanitized = _sanitize_value(raw_value, section, top_level=(key == "sample_id"))
        if sanitized is not None and sanitized != {} and sanitized != []:
            result[key] = sanitized
    _assert_safe_output(result)
    return result


def sanitize_json_file(path: str | Path) -> dict[str, Any]:
    """Parse a JSON file without exposing banned fields or values."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return sanitize_payload(payload)


def guard_geometry_input_paths(paths: Sequence[str | Path]) -> None:
    """Reject direct geometry inputs whose path names indicate result data."""

    for value in paths:
        path_text = str(value).lower()
        if any(token in path_text for token in BANNED_TOKENS):
            raise ValueError("geometry-only input path contains a banned token")


def assert_geometry_only_output(payload: Mapping[str, Any]) -> None:
    """Enforce the output schema at the boundary of a Gate A tool."""

    if not isinstance(payload, Mapping):
        raise ValueError("geometry-only output must be an object")
    allowed = {"schema_version", *ALLOWED_TOP_LEVEL_KEYS}
    unknown = {_normalized_key(key) for key in payload} - allowed
    if unknown:
        raise ValueError("geometry-only output contains an unknown top-level key")
    _assert_safe_output(payload)
