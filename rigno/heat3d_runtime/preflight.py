"""Fail-closed semantic-contract validation for the V7 Heat3D runtime."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any


EXECUTION_ROLES = frozenset(
    {"publication_training", "production_inference", "compatibility_audit"}
)

_MODEL_FIELDS = (
    "qk_region_feature_version",
    "decoder_bypass_mode",
    "decoder_bypass_features",
    "decoder_bypass_feature_source",
    "decoder_bypass_output_space",
    "decoder_bypass_num_features",
    "decoder_bypass_local_feature_names",
    "native_output_mode",
    "global_context_mode",
    "global_context_feature_dim",
    "scale_context_mode",
    "scale_pooling",
    "shape_attention_mode",
    "scale_attention_mode",
    "scale_deepsets_mode",
)
_STATS_FIELDS = (
    "input_feature_schema",
    "coord_policy",
    "extent_feature_policy",
    "normalization_profile",
    "feature_names",
    "condition_feature_transforms",
    "condition_mean",
    "condition_std",
    "coord_min",
    "coord_span",
)
_RUN_FIELDS = (
    "graph_config",
    "graph_seed",
    "global_context",
    "scale_context",
    "input_feature_schema",
    "coord_policy",
    "extent_feature_policy",
    "normalization_profile",
    "condition_feature_transform",
)
_GRAPH_FIELDS = (
    "discrete_graph_backend",
    "coverage_repair_policy",
    "discrete_coverage_multiplier",
    "discrete_graph_chunk_size",
    "min_physical_coverage",
    "node_coordinate_encoding",
    "node_coordinate_freqs",
    "radius_policy",
    "repair_p2r",
    "repair_r2p",
)
_ROUTE_FIELDS = (
    "route_id",
    "strategy_name",
    "anchor_context_resolution",
    "encoder_input_resolution",
    "output_query_resolution",
    "direct_query",
    "reconstruction_resolution",
    "fixed_edge_targets",
)


class SemanticContractError(ValueError):
    """Raised when a semantic-critical runtime field is absent or invalid."""


def _require(mapping: Mapping[str, Any], field: str, scope: str) -> Any:
    if field not in mapping or mapping[field] is None:
        raise SemanticContractError(f"{scope}.{field} is required; no legacy default is allowed")
    return mapping[field]


def _require_mapping(mapping: Mapping[str, Any], field: str, scope: str) -> Mapping[str, Any]:
    value = _require(mapping, field, scope)
    if not isinstance(value, Mapping):
        raise SemanticContractError(f"{scope}.{field} must be a mapping")
    return value


def _require_fields(mapping: Mapping[str, Any], fields: tuple[str, ...], scope: str) -> None:
    for field in fields:
        _require(mapping, field, scope)


def _validate_route(route_contract: Mapping[str, Any]) -> None:
    deprecated = {
        "conditioning_resolution",
        "conditioning_input_resolution",
        "query_resolution",
    }
    present_deprecated = sorted(deprecated.intersection(route_contract))
    if present_deprecated:
        raise SemanticContractError(
            "route_contract contains deprecated ambiguous fields: "
            f"{present_deprecated}"
        )
    _require_fields(route_contract, _ROUTE_FIELDS, "route_contract")
    route_id = route_contract["route_id"]
    if not isinstance(route_id, str) or not route_id:
        raise SemanticContractError("route_contract.route_id must be a non-empty string")
    strategy = str(route_contract["strategy_name"])
    if strategy not in {"E", "U-v2"}:
        raise SemanticContractError(f"route_contract.strategy_name is unsupported: {strategy!r}")
    resolutions = (
        "anchor_context_resolution",
        "encoder_input_resolution",
        "output_query_resolution",
        "reconstruction_resolution",
    )
    for field in resolutions:
        value = route_contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SemanticContractError(f"route_contract.{field} must be positive")
    if strategy == "E" and (
        route_contract["encoder_input_resolution"]
        != route_contract["output_query_resolution"]
    ):
        raise SemanticContractError("E route requires encoder_input_resolution == output_query_resolution")
    if strategy == "U-v2" and (
        route_contract["anchor_context_resolution"] != 1024
        or route_contract["encoder_input_resolution"]
        != route_contract["anchor_context_resolution"]
    ):
        raise SemanticContractError(
            "U-v2 route requires encoder_input_resolution == "
            "anchor_context_resolution == 1024"
        )
    if not isinstance(route_contract["direct_query"], bool):
        raise SemanticContractError("route_contract.direct_query must be boolean")
    if int(route_contract["output_query_resolution"]) > 1024:
        targets = route_contract["fixed_edge_targets"]
        if not isinstance(targets, Mapping) or not targets:
            raise SemanticContractError(
                "high-resolution route requires explicit fixed_edge_targets"
            )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item) for item in value)
    return value


def _resolve_fixed_edge_targets(contract_path: Path, route: dict[str, Any]) -> dict[str, Any]:
    """Resolve a registered padding reference without accepting arbitrary defaults."""

    value = route["fixed_edge_targets"]
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or ":" not in value:
        raise SemanticContractError(
            "registered route fixed_edge_targets is unresolved; production binding is refused"
        )
    relative_path, selector = value.split(":", 1)
    root = contract_path.resolve().parents[2]
    source = (root / relative_path).resolve()
    if not source.is_file() or root not in source.parents:
        raise SemanticContractError(f"registered padding source is unavailable: {relative_path}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected: Any = payload
    for component in selector.split("."):
        if not isinstance(selected, Mapping) or component not in selected:
            raise SemanticContractError(
                f"registered padding selector is unresolved: {relative_path}:{selector}"
            )
        selected = selected[component]
    if not isinstance(selected, Mapping) or not selected:
        raise SemanticContractError(
            f"registered padding envelope is not a non-empty mapping: {relative_path}:{selector}"
        )
    return dict(selected)


def load_registered_route(contract_path: str | Path, route_id: str) -> dict[str, Any]:
    """Load one route by explicit ID; unknown routes fail closed."""

    path = Path(contract_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategies = payload["strategies"]
    if not isinstance(strategies, Mapping) or route_id not in strategies:
        raise SemanticContractError(f"route_id is not registered: {route_id!r}")
    raw_route = strategies[route_id]
    if not isinstance(raw_route, Mapping):
        raise SemanticContractError(f"registered route is not an object: {route_id!r}")
    route = dict(raw_route)
    if route.get("route_id") != route_id:
        raise SemanticContractError(
            f"registered route_id mismatch: key={route_id!r}, field={route.get('route_id')!r}"
        )
    route["fixed_edge_targets"] = _resolve_fixed_edge_targets(path, route)
    _validate_route(route)
    return route


def bind_registered_route(
    *,
    contract_path: str | Path,
    route_id: str,
    requested_strategy: str,
    anchor_context_resolution: int,
    encoder_input_resolution: int,
    output_query_resolution: int,
    reconstruction_resolution: int,
    fixed_edge_targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a production request to equal one registered route exactly."""

    route = load_registered_route(contract_path, route_id)
    requested = {
        "strategy_name": requested_strategy,
        "anchor_context_resolution": anchor_context_resolution,
        "encoder_input_resolution": encoder_input_resolution,
        "output_query_resolution": output_query_resolution,
        "reconstruction_resolution": reconstruction_resolution,
        "fixed_edge_targets": fixed_edge_targets,
    }
    if requested_strategy != route["strategy_name"]:
        raise SemanticContractError(
            f"route strategy mismatch: requested={requested_strategy!r}, "
            f"registered={route['strategy_name']!r}"
        )
    for field in (
        "anchor_context_resolution",
        "encoder_input_resolution",
        "output_query_resolution",
        "reconstruction_resolution",
    ):
        if requested[field] != route[field]:
            raise SemanticContractError(
                f"route {field} mismatch: requested={requested[field]!r}, "
                f"registered={route[field]!r}"
            )
    if output_query_resolution != route["output_query_resolution"]:
        raise SemanticContractError("requested output resolution is not the registered route output")
    if _canonical(fixed_edge_targets) != _canonical(route["fixed_edge_targets"]):
        raise SemanticContractError("route fixed padding envelope mismatch")
    return route


def validate_semantic_contract(
    *,
    run_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    stats: Mapping[str, Any],
    execution_role: str,
    route_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate semantic-critical fields and return an auditable summary.

    This function is deliberately side-effect free.  Compatibility callers may
    supply a legacy-derived mapping, but the source remains explicit in the
    returned receipt and missing fields still fail closed.
    """

    if execution_role not in EXECUTION_ROLES:
        raise SemanticContractError(
            f"execution_role must be one of {sorted(EXECUTION_ROLES)}; got {execution_role!r}"
        )
    _require_fields(run_config, _RUN_FIELDS, "run_config")
    _require_fields(model_config, _MODEL_FIELDS, "model_config")
    _require_fields(stats, _STATS_FIELDS, "checkpoint_stats")

    graph_config = _require_mapping(run_config, "graph_config", "run_config")
    _require_fields(graph_config, _GRAPH_FIELDS, "run_config.graph_config")
    if not isinstance(run_config["graph_seed"], int):
        raise SemanticContractError("run_config.graph_seed must be an integer")

    global_context = _require_mapping(run_config, "global_context", "run_config")
    scale_context = _require_mapping(run_config, "scale_context", "run_config")
    global_mode = str(model_config["global_context_mode"])
    native_mode = str(model_config["native_output_mode"])
    if global_mode != "none" or native_mode == "native_shape_scale":
        standardizer = _require_mapping(global_context, "standardizer", "run_config.global_context")
        _require(standardizer, "feature_names", "run_config.global_context.standardizer")
        if _require(global_context, "target_or_label_derived_inputs", "run_config.global_context") is not False:
            raise SemanticContractError(
                "run_config.global_context.target_or_label_derived_inputs must be false"
            )
    if str(model_config["scale_context_mode"]) != "none":
        _require_mapping(global_context, "scale_standardizer", "run_config.global_context")
        if _require(scale_context, "target_or_label_derived_inputs", "run_config.scale_context") is not False:
            raise SemanticContractError(
                "run_config.scale_context.target_or_label_derived_inputs must be false"
            )

    feature_names = stats["feature_names"]
    if not isinstance(feature_names, (tuple, list)) or not feature_names:
        raise SemanticContractError("checkpoint_stats.feature_names must be a non-empty sequence")
    if stats["condition_feature_transforms"] is None:
        raise SemanticContractError(
            "checkpoint_stats.condition_feature_transforms is required; no transform default is allowed"
        )

    if route_contract is not None:
        _validate_route(route_contract)

    return {
        "execution_role": execution_role,
        "critical_fields_checked": {
            "run_config": list(_RUN_FIELDS),
            "graph_config": list(_GRAPH_FIELDS),
            "model_config": list(_MODEL_FIELDS),
            "checkpoint_stats": list(_STATS_FIELDS),
            "route_contract": list(_ROUTE_FIELDS) if route_contract is not None else [],
        },
        "route_contract_present": route_contract is not None,
        "legacy_defaults_allowed": False,
        "source": "checkpoint/run_config/route contract",
    }
