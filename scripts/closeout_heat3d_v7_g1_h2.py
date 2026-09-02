#!/usr/bin/env python3
"""V7 G1 H2 full-field closeout for the two frozen U-v2 routes.

This script is evaluation-only.  It reconstructs the train-only runtime
preparation needed by the already-produced V7 G1 checkpoints, evaluates only
the 128 ``valid_iid`` samples, and writes evidence below an explicitly passed
temporary/output root.  It never constructs an optimizer, calls a training
loop, opens test/sealed labels, or changes a checkpoint.

The ``execute`` mode evaluates the nine selected-best checkpoints for one or
both registered U-v2 routes.  The ``analyze`` mode consumes only the resulting
per-sample metric rows and performs the preregistered two-level bootstrap.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np
from scipy.spatial import cKDTree


FORMAL_CODE_SHA = "191a7a06a681556f575a1c04e2b61cb13363efe1"
PREREG_SHA = "03be1617b78f2e1f41431411e601a54136a59e363c8321457a19b717249ad31e"
GOVERNANCE_AMENDMENT_SHA = "168966f6f9091f46ea831ad08a6c014de6d38541766f2419d5fcac3cab4cbd52"
GOVERNANCE_AMENDMENT_SCHEMA = "heat3d_v7_g1_h2_native_closeout_governance_amendment_v1"
NATIVE_CAPACITY_SCHEMA = "heat3d_v7_g1_native_geometry_capacity_manifest_v1"
H2_PRIMARY_DECISION_SCHEMA = "heat3d_v7_g1_h2_route_primary_decision_v1"
COMMON_DOMAIN_ID = "heat3d_v6_p1i_full_field_240825"
FULL_FIELD_RESOLUTION = 240825
NATIVE_RESOLUTION = 1024
U16384_RESOLUTION = 16384
VALID_COUNT = 128
SEEDS = (0, 1, 2)
VARIANTS = ("Full", "layout_agnostic_stratified_support", "cv_only_support")
H2_ROUTES = ("U_v2_16384_reconstruction", "U_v2_direct240825")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260829
METRICS = (
    "source_region_RMSE_K",
    "point_global_relative_rmse_pct",
    "sample_first_relative_rmse_pct",
    "raw_K_CV_RMSE_K",
    "peak_RMSE_K",
    "interface_RMSE_K",
)
H2_PRIMARY_METRIC = "source_region_RMSE_K"
SUPPORT_PROVIDER_BY_VARIANT = {
    "layout_agnostic_stratified_support": "generic_stratified_v2",
    "cv_only_support": "cv_only_v1",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "analyze"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--full-fields", type=Path, default=None)
    parser.add_argument("--launch-manifest", type=Path, default=None)
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--eu-contract", type=Path, default=None)
    parser.add_argument("--governance-amendment", type=Path, default=None)
    parser.add_argument("--capacity-manifest", type=Path, default=None)
    parser.add_argument("--native-anchor", type=Path, default=None)
    parser.add_argument("--primary-decision", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--route-id", choices=H2_ROUTES, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--allow-existing-complete",
        action="store_true",
        help="skip a route/run directory only when its complete receipt is valid",
    )
    return parser.parse_args()


def _repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _resolve_execute_paths(args: argparse.Namespace) -> dict[str, Path]:
    repo = args.repo.resolve()
    launch = (args.launch_manifest or repo / "configs/heat3d_v7/v7_g1_formal_launch_manifest.json").resolve()
    launch_payload = _load_json(launch)
    dataset = _load_json(repo / "configs/heat3d_v7/v7_g1_full_p1i.json")["dataset"]
    return {
        "repo": repo,
        "launch": launch,
        "prereg": (args.preregistration or repo / "configs/heat3d_v7/v7_g1_statistical_preregistration.json").resolve(),
        "eu_contract": (args.eu_contract or repo / "configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json").resolve(),
        "governance": (args.governance_amendment or repo / "configs/heat3d_v7/v7_g1_h2_native_closeout_governance_amendment.json").resolve(),
        "capacity_manifest": (args.capacity_manifest or Path("/tmp/v7_g1_h2_native_closeout/g1_native_geometry_capacity_manifest.json")).resolve(),
        "native_anchor": (args.native_anchor or Path("/tmp/v7_g1_h2_native_closeout/g1_native_anchor_Full_seed0_v6p1if1_0993.json")).resolve(),
        "v6_binding": (repo / "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json").resolve(),
        "decision": (args.primary_decision or repo / "configs/heat3d_v7/v7_g1_h2_route_primary_decision.json").resolve(),
        "subset": (args.subset or _repo_path(repo, str(dataset["subset_path"]))),
        "manifest": (args.manifest or _repo_path(repo, str(dataset["manifest_path"]))),
        "full_fields": (args.full_fields or _repo_path(repo, str(dataset["full_field_archive_path"]))),
        "checkpoint_root": (args.checkpoint_root or Path("/tmp/v7_g1_formal_runs")).resolve(),
        "launch_payload": launch_payload,
    }


def _verify_governance_amendment(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen H2 governance amendment: {path}")
    sha = _sha256(path)
    if sha != GOVERNANCE_AMENDMENT_SHA:
        raise ValueError(f"H2 governance amendment SHA drifted: {sha}")
    amendment = _load_json(path)
    if amendment.get("schema_version") != GOVERNANCE_AMENDMENT_SCHEMA:
        raise ValueError("H2 governance amendment schema drifted")
    if amendment.get("status") != "FROZEN_BEFORE_H2_ACCURACY":
        raise ValueError("H2 governance amendment is not frozen before accuracy")
    historical = amendment.get("historical_3074")
    if not isinstance(historical, Mapping) or historical.get("status") != "HISTORICAL_REPRODUCIBILITY_DIAGNOSTIC_ONLY" or historical.get("is_h2_scientific_gate") is not False:
        raise ValueError("V6 3074 role is not diagnostic-only")
    gates = amendment.get("gates")
    if not isinstance(gates, Mapping) or not isinstance(gates.get("gate_a"), Mapping) or not isinstance(gates.get("gate_b"), Mapping):
        raise ValueError("Gate A/B governance sections are missing")
    if any(
        gates[name].get(field) is not False
        for name in ("gate_a", "gate_b")
        for field in ("checkpoint_load", "model_forward", "truth_access", "accuracy_or_metric_calculation", "target_or_loss_access")
    ):
        raise ValueError("Gate A/B governance boundary drifted")
    if not isinstance(gates.get("gate_c"), Mapping) or gates["gate_c"].get("model_forward") is not True or gates["gate_c"].get("valid_iid_truth_access") is not True or gates["gate_c"].get("parameter_update") is not False:
        raise ValueError("Gate C governance boundary drifted")
    return amendment, sha


def _verify_native_capacity_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing geometry-only capacity manifest: {path}")
    sha = _sha256(path)
    payload = _load_json(path)
    if payload.get("schema_version") != NATIVE_CAPACITY_SCHEMA:
        raise ValueError("native geometry capacity manifest schema drifted")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("formal_code_sha") != FORMAL_CODE_SHA:
        raise ValueError("native geometry formal code provenance drifted")
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("native geometry graph section is missing")
    if graph.get("adapter_native_exact_all_records") is not True:
        raise ValueError("native/U adapter exactness gate is not PASS")
    if graph.get("padding_invariance", {}).get("status") != "PASS":
        raise ValueError("native geometry padding invariance gate is not PASS")
    route_caps = graph.get("route_edge_capacities")
    if not isinstance(route_caps, Mapping) or set(route_caps) != set(H2_ROUTES):
        raise ValueError("native geometry route capacity population drifted")
    for route_id in H2_ROUTES:
        route = route_caps[route_id]
        if not isinstance(route, Mapping) or not isinstance(route.get("native"), Mapping) or not isinstance(route.get("query"), Mapping):
            raise ValueError(f"{route_id}: native/query capacity scopes are missing")
        for scope in ("native", "query"):
            for field in ("p2r_edge_indices", "r2p_edge_indices", "r2r_edge_indices", "r2r_edge_domains"):
                value = route[scope].get(field)
                if value is not None and (isinstance(value, bool) or int(value) < 1):
                    raise ValueError(f"{route_id}/{scope}/{field}: invalid frozen capacity")
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("formal_native_resolution") != NATIVE_RESOLUTION or geometry.get("u_query_resolutions") != {"U16384_query": U16384_RESOLUTION, "U240825_query": FULL_FIELD_RESOLUTION}:
        raise ValueError("native geometry resolution contract drifted")
    return payload, sha


def _verify_native_anchor(path: Path, capacity_payload: Mapping[str, Any]) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing native anchor receipt: {path}")
    sha = _sha256(path)
    payload = _load_json(path)
    if payload.get("schema_version") != "heat3d_v7_g1_native_anchor_geometry_only_v1":
        raise ValueError("native anchor schema drifted")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("formal_code_sha") != FORMAL_CODE_SHA or provenance.get("anchor_role") != "Full_seed0/v6p1if1_0993":
        raise ValueError("native anchor identity drifted")
    geometry = payload.get("geometry")
    support = payload.get("support")
    graph = payload.get("graph")
    if not isinstance(geometry, Mapping) or not isinstance(support, Mapping) or not isinstance(graph, Mapping) or graph.get("real_edge_counts", {}).get("p2r") is None or graph.get("real_edge_counts", {}).get("r2r") is None:
        raise ValueError("native anchor graph counts are missing")
    records = capacity_payload.get("graph", {}).get("records", [])
    if not any(
        isinstance(row, Mapping)
        and row.get("provenance", {}).get("run_id") == "Full_seed0"
        and row.get("provenance", {}).get("sample_id") == "v6p1if1_0993"
        and row.get("geometry") == geometry
        and row.get("support") == support
        and row.get("graph") == graph
        for row in records
    ):
        raise ValueError("native anchor does not exactly match the capacity manifest")
    return sha


def _verify_execute_contract(paths: Mapping[str, Path | dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    launch = paths["launch_payload"]
    assert isinstance(launch, dict)
    prereg = _load_json(Path(paths["prereg"]))
    contract = _load_json(Path(paths["eu_contract"]))
    decision = _load_json(Path(paths["decision"]))
    amendment, amendment_sha = _verify_governance_amendment(Path(paths["governance"]))
    capacity_manifest, capacity_sha = _verify_native_capacity_manifest(Path(paths["capacity_manifest"]))
    anchor_sha = _verify_native_anchor(Path(paths["native_anchor"]), capacity_manifest)
    if launch.get("g1_formal_code_sha") != FORMAL_CODE_SHA:
        raise ValueError("formal launch code SHA drifted")
    if launch.get("test_iid_access") or launch.get("sealed_access"):
        raise ValueError("formal launch manifest opens test/sealed access")
    if prereg.get("preregistration_sha256") != PREREG_SHA:
        raise ValueError("statistical preregistration SHA drifted")
    if decision.get("schema_version") != H2_PRIMARY_DECISION_SCHEMA:
        raise ValueError("H2 route primary decision schema drifted")
    if decision.get("status") != "RESOLVED_BEFORE_RESULT_CALCULATION":
        raise ValueError("H2 primary route decision is not pre-result resolved")
    if decision.get("accuracy_was_read_before_decision") is not False:
        raise ValueError("H2 route decision was not recorded as accuracy-blind")
    if decision.get("primary_route_id") != "U_v2_16384_reconstruction":
        raise ValueError("H2 primary route is not the frozen 16384 reconstruction route")
    if decision.get("robustness_route_id") != "U_v2_direct240825":
        raise ValueError("H2 robustness route is not the frozen direct 240825 route")
    runs = list(launch.get("runs") or [])
    if len(runs) != 21:
        raise ValueError("formal launch matrix is not 21 runs")
    selected = [
        row for row in runs
        if str(row.get("variant")) in VARIANTS
        and int(row.get("seed")) in SEEDS
    ]
    if len(selected) != 9 or len({str(row["run_id"]) for row in selected}) != 9:
        raise ValueError("H2 checkpoint population is not exactly 9 runs")
    if not Path(paths["subset"]).is_dir():
        raise FileNotFoundError(f"missing frozen subset: {paths['subset']}")
    if not Path(paths["manifest"]).is_file() or not Path(paths["full_fields"]).is_file():
        raise FileNotFoundError("missing frozen manifest/full-field archive")
    return launch, prereg, contract, decision, capacity_manifest, amendment_sha, capacity_sha, anchor_sha


def _registered_route(contract_path: Path, route_id: str) -> dict[str, Any]:
    from rigno.heat3d_runtime.preflight import bind_registered_route, load_registered_route

    route = load_registered_route(contract_path, route_id)
    query_resolution = int(route["output_query_resolution"])
    bound = bind_registered_route(
        contract_path=contract_path,
        route_id=route_id,
        requested_strategy=str(route["strategy_name"]),
        anchor_context_resolution=int(route["anchor_context_resolution"]),
        encoder_input_resolution=int(route["encoder_input_resolution"]),
        output_query_resolution=query_resolution,
        reconstruction_resolution=int(route["reconstruction_resolution"]),
        fixed_edge_targets=route["fixed_edge_targets"],
    )
    if query_resolution not in {U16384_RESOLUTION, FULL_FIELD_RESOLUTION}:
        raise ValueError(f"unsupported H2 route output resolution: {query_resolution}")
    return bound


def _amend_route_capacity(
    route: Mapping[str, Any],
    capacity_manifest: Mapping[str, Any],
    capacity_sha: str,
) -> dict[str, Any]:
    route_id = str(route["route_id"])
    capacities = capacity_manifest["graph"]["route_edge_capacities"]
    amended = deepcopy(dict(route))
    amended["fixed_edge_targets"] = deepcopy(dict(capacities[route_id]))
    amended["capacity_manifest_sha256"] = capacity_sha
    amended["capacity_semantics"] = "complete 9x128 geometry-only observed maximum plus mandatory dummy"
    return amended


def _load_examples(
    *,
    repo: Path,
    subset: Path,
    manifest: Path,
    full_fields: Path,
    variant: str,
    seed: int,
) -> tuple[list[Any], list[Any], dict[str, dict[str, float]], str]:
    from rigno.heat3d_training.full_field import load_alternative_p1i_examples
    from rigno.heat3d_training.p1i import load_selected_p1i_examples
    from rigno.heat3d_v6_global_context import global_context_from_v6_inputs

    if variant in SUPPORT_PROVIDER_BY_VARIANT:
        prepared = load_alternative_p1i_examples(
            subset=subset,
            manifest_path=manifest,
            full_field_archive_path=full_fields,
            provider_id=SUPPORT_PROVIDER_BY_VARIANT[variant],
            seed=seed,
        )
        train = list(prepared.train_examples)
        valid = list(prepared.valid_examples)
        context_rows = {str(key): dict(value) for key, value in prepared.context_by_id.items()}
        provider = SUPPORT_PROVIDER_BY_VARIANT[variant]
    elif variant == "Full":
        loaded = load_selected_p1i_examples(subset, manifest)
        train = list(loaded["train"])
        valid = list(loaded["valid_iid"])
        context_rows = {
            str(example.sample_id): global_context_from_v6_inputs(
                **example.v6_global_context_inputs()
            )
            for example in [*train, *valid]
        }
        provider = "historical_v6_stored_support"
    else:
        raise ValueError(f"unsupported H2 variant: {variant}")
    if len(train) != 768 or len(valid) != VALID_COUNT:
        raise ValueError(f"{variant}/seed{seed}: train/valid population drifted")
    ids = [str(example.sample_id) for example in [*train, *valid]]
    if len(ids) != len(set(ids)) or set(ids) != set(context_rows):
        raise ValueError(f"{variant}/seed{seed}: context population drifted")
    if any(str(example.meta.get("v6_adapter", {}).get("manifest_split_role")) not in {"train", "valid_iid"} for example in [*train, *valid]):
        raise ValueError(f"{variant}/seed{seed}: non-train/valid example entered H2 preparation")
    return train, valid, context_rows, provider


def _build_session(
    *,
    repo: Path,
    run: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    raw_checkpoint_path: Path,
    source_run_config: Mapping[str, Any],
    train_examples: list[Any],
    context_rows_by_id: Mapping[str, Mapping[str, float]],
    route: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from rigno.heat3d_runtime.checkpoint import (
        CheckpointBundle,
        device_params,
        materialize_checkpoint_stats,
        resolve_model_config,
    )
    from rigno.heat3d_runtime.features import FeatureTransform
    from rigno.heat3d_runtime.grouping import GroupBuilder
    from rigno.heat3d_runtime.preflight import validate_semantic_contract
    from rigno.heat3d_runtime.session import RuntimeSession
    from rigno.heat3d_training.p1i import COORD_POLICY_TRAIN_MINMAX_UNIT_BOX, legacy_train_only_stats
    from rigno.heat3d_v6_global_context import fit_train_only_v6_standardizer
    from rigno.models.rigno import RIGNO

    variant = str(run["variant"])
    seed = int(run["seed"])
    stats_raw = legacy_train_only_stats(
        list(train_examples),
        coord_policy=COORD_POLICY_TRAIN_MINMAX_UNIT_BOX,
    )
    train_ids = [str(example.sample_id) for example in train_examples]
    standardizer = fit_train_only_v6_standardizer(
        [context_rows_by_id[sample_id] for sample_id in train_ids],
        fit_sample_ids=train_ids,
    )
    stats = materialize_checkpoint_stats(stats_raw, strict_semantic_contract=True)
    model_config = resolve_model_config(
        dict(source_run_config["model_resolved_for_variant"]),
        stats,
        strict_semantic_contract=True,
    )
    # The frozen parent contract predates the explicit runtime field.  The
    # RIGNO constructor's frozen default is ``none``; materialize that default
    # only to satisfy the no-legacy-default runtime preflight, without changing
    # the trained architecture.
    model_config.setdefault("shape_attention_mode", "none")
    runtime_config = {
        "schema_version": "heat3d_v7_g1_h2_runtime_config_v1",
        "graph_config": dict(source_run_config["graph"]),
        "graph_seed": seed,
        "global_context": {
            "schema": "GLOBAL_CONTEXT_FEATURES_V6",
            "standardizer": standardizer,
            "target_or_label_derived_inputs": False,
        },
        "scale_context": {"target_or_label_derived_inputs": False},
        "input_feature_schema": stats_raw["input_feature_schema"],
        "coord_policy": stats_raw["coord_policy"],
        "extent_feature_policy": stats_raw["extent_feature_policy"],
        "normalization_profile": stats_raw["normalization_profile"],
        "condition_feature_transform": stats_raw["condition_feature_transform"],
    }
    semantic_contract = validate_semantic_contract(
        run_config=runtime_config,
        model_config=model_config,
        stats=stats,
        execution_role="production_inference",
        route_contract=route,
    )
    with raw_checkpoint_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{raw_checkpoint_path}: params-only checkpoint payload invalid")
    expected_epoch = raw_receipt.get("checkpoint_selection", {}).get("best_epoch")
    if expected_epoch is None or int(payload.get("epoch")) != int(expected_epoch):
        raise ValueError(f"{run['run_id']}: selected-best checkpoint epoch drifted")
    checkpoint = CheckpointBundle(
        path=raw_checkpoint_path.resolve(),
        sha256=_sha256(raw_checkpoint_path),
        params=payload["params"],
        model_config=model_config,
        stats=stats,
        epoch=int(payload["epoch"]),
        checkpoint_kind="best_sample_first",
        git_commit=(None if raw_receipt.get("git_commit") is None else str(raw_receipt["git_commit"])),
        payload_metadata={
            "raw_checkpoint_schema_version": payload.get("schema_version"),
            "raw_checkpoint_variant": payload.get("variant"),
            "raw_checkpoint_execution_role": payload.get("execution_role"),
        },
    )
    feature_transform = FeatureTransform(
        stats,
        context_rows_by_id={
            str(sample_id): dict(row) for sample_id, row in context_rows_by_id.items()
        },
    )
    session = RuntimeSession(
        checkpoint=checkpoint,
        run_config=runtime_config,
        model_config=model_config,
        graph_config=dict(runtime_config["graph_config"]),
        feature_transform=feature_transform,
        group_builder=GroupBuilder(
            feature_transform=feature_transform,
            graph_config=dict(runtime_config["graph_config"]),
            graph_seed=seed,
        ),
        model=RIGNO(**model_config),
        params=device_params(payload["params"]),
        execution_role="production_inference",
        semantic_contract=semantic_contract,
    )
    del payload
    return session, runtime_config, stats_raw, model_config


def _continuous_fields(meta: Mapping[str, Any], mesh: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Call the frozen V6/P1i high-N k/q implementation without labels."""

    # Imported from the fingerprinted V6 benchmark module at execution time;
    # this duplicate guard makes the exact high-N fallback explicit in the
    # H2 receipt while avoiding any training entrypoint call.
    from scripts import benchmark_heat3d_v6_p1i_resolution as resolution_base

    return resolution_base._continuous_fields(dict(meta), mesh)


def _prepare_fixture(
    *,
    example: Any,
    geometry: Any,
    resolution: int,
    selection_seed: int,
) -> dict[str, Any]:
    from scripts import benchmark_heat3d_v6_p1i_resolution as resolution_base
    from rigno.heat3d_runtime.high_n import SupportArtifact
    from rigno.heat3d_v6_p1i_anchor_query import (
        conservative_selected_control_volume,
        deterministic_nested_query_order,
    )

    distance, anchor_indices = cKDTree(geometry.coords).query(
        np.asarray(example.condition.coords, dtype=np.float64), k=1
    )
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    if (
        anchor_indices.shape != (NATIVE_RESOLUTION,)
        or len(np.unique(anchor_indices)) != NATIVE_RESOLUTION
        or float(np.max(distance)) > 1.0e-14
    ):
        raise ValueError(f"{example.sample_id}: native anchor is not an exact full-mesh subset")
    meta = deepcopy(example.meta)
    meta.pop("v6_adapter", None)
    meta["sample_id"] = str(example.sample_id)
    mesh = resolution_base.core.build_mesh(meta["physics"])
    mesh_coords = np.asarray(mesh["coords"], dtype=np.float64)
    mesh_cv = np.asarray(mesh["weights"], dtype=np.float64)
    mesh_layer = np.asarray(mesh["layer_ids"], dtype=np.int32)
    if (
        mesh_coords.shape != geometry.coords.shape
        or float(np.max(np.abs(mesh_coords - geometry.coords))) > 1.0e-14
        or not np.array_equal(mesh_layer, geometry.layer_id)
        or not np.allclose(mesh_cv, geometry.control_volume, rtol=0.0, atol=1.0e-30)
    ):
        raise ValueError(f"{example.sample_id}: frozen mesh does not match shared full-field geometry")
    full_k, full_q, power_audit = _continuous_fields(meta, mesh)
    full_k = np.asarray(full_k, dtype=np.float64)
    full_q = np.asarray(full_q, dtype=np.float64)
    if (
        full_k.shape != (FULL_FIELD_RESOLUTION, 3)
        or full_q.shape != (FULL_FIELD_RESOLUTION,)
        or not np.all(np.isfinite(full_k))
        or not np.all(np.isfinite(full_q))
        or np.any(full_k <= 0.0)
        or np.any(full_q < 0.0)
        or float(power_audit["relative_power_error"]) > 1.0e-12
    ):
        raise ValueError(f"{example.sample_id}: frozen full-field k/q audit failed")
    order, order_audit = deterministic_nested_query_order(
        sample_id=str(example.sample_id),
        anchor_indices=anchor_indices,
        full_coords=geometry.coords,
        full_control_volume=geometry.control_volume,
        full_layer_id=geometry.layer_id,
        full_q=full_q,
        layer_boundaries_m=np.asarray(mesh["boundaries"], dtype=np.float64),
        selection_seed=int(selection_seed),
    )
    selected_indices = np.asarray(order[:resolution], dtype=np.int64)
    effective_cv, cv_audit = conservative_selected_control_volume(
        full_coords=geometry.coords,
        full_control_volume=geometry.control_volume,
        full_layer_id=geometry.layer_id,
        selected_indices=selected_indices,
    )
    if float(cv_audit["relative_volume_error"]) > 1.0e-12:
        raise ValueError(f"{example.sample_id}: selected support CV conservation failed")
    support_sha = _canonical_sha(
        {
            "selected_indices": _array_sha256(selected_indices.astype(np.int32)),
            "operator_control_volume": _array_sha256(effective_cv),
            "k_xyz": _array_sha256(full_k[selected_indices]),
            "q_W_m3": _array_sha256(full_q[selected_indices]),
            "layer_id": _array_sha256(geometry.layer_id[selected_indices]),
        }
    )
    support = SupportArtifact.from_arrays(
        selected_indices=selected_indices,
        operator_control_volume=effective_cv,
        k_xyz=full_k[selected_indices],
        q_W_m3=full_q[selected_indices],
        layer_id=geometry.layer_id[selected_indices],
        path=f"<generated-frozen-nested-support:{example.sample_id}:{resolution}>",
        sha256=support_sha,
    )
    return {
        "sample_id": str(example.sample_id),
        "mesh": mesh,
        "full_k": full_k,
        "full_q": full_q,
        "power_audit": power_audit,
        "anchor_indices": anchor_indices,
        "order": np.asarray(order, dtype=np.int64),
        "order_audit": order_audit,
        "selected_indices": selected_indices,
        "effective_cv": np.asarray(effective_cv, dtype=np.float64),
        "cv_audit": cv_audit,
        "support": support,
        "layer_boundaries": np.asarray(mesh["boundaries"], dtype=np.float64),
        "source_mask": full_q > 0.0,
    }


def _checkpoint_and_receipt(
    *,
    checkpoint_root: Path,
    run: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    run_id = str(run["run_id"])
    raw_dir = checkpoint_root / run_id
    receipt_path = raw_dir / "v7_g1_formal_receipt.json"
    checkpoint_path = raw_dir / "params_best_sample_first.pkl"
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"{run_id}: selected-best checkpoint or formal receipt missing")
    receipt = _load_json(receipt_path)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("g1_formal") is not True
        or receipt.get("formal_run_id") != run_id
        or receipt.get("g1_formal_code_sha") != FORMAL_CODE_SHA
        or receipt.get("test_and_sealed_access") != "closed"
        or receipt.get("split_counts") != {"train": 768, "valid_iid": 128}
    ):
        raise ValueError(f"{run_id}: formal receipt safety/provenance guard failed")
    receipt_seed = receipt.get("seed")
    if receipt.get("variant") != run["variant"] or (
        receipt_seed is not None and int(receipt_seed) != int(run["seed"])
    ):
        raise ValueError(f"{run_id}: formal receipt variant/seed drifted")
    if receipt.get("checkpoint_selection", {}).get("metric") != "sample_first_relative_rmse_pct":
        raise ValueError(f"{run_id}: selected-best metric drifted")
    if launch.get("test_iid_access") or launch.get("sealed_access"):
        raise ValueError("launch manifest safety guard failed")
    return checkpoint_path, receipt


def _source_run_config(repo: Path, run_id: str) -> dict[str, Any]:
    # The persisted native export config is a read-only provenance source.  It
    # contains the exact resolved model/graph contract of the formal run; H2
    # adds only the explicit runtime fields required by RuntimeSession.
    path = repo / "research_artifacts/v7_g1_formal_archive/derived_1024" / run_id / "run_config.json"
    if path.is_file():
        payload = _load_json(path)
        if payload.get("run_id") != run_id:
            raise ValueError(f"{run_id}: source run config identity drifted")
        return payload
    # The ignored native archive is intentionally not required on devbox.
    # Reconstruct only the immutable model/graph portion from the tracked
    # parent contract and the registered variant; no checkpoint or training
    # state is changed by this fallback.
    config = _load_json(repo / "configs/heat3d_v7/v7_g1_full_p1i.json")
    run_parts = str(run_id).rsplit("_seed", 1)
    if len(run_parts) != 2:
        raise ValueError(f"cannot resolve source run config for {run_id}")
    variant = run_parts[0]
    seed = int(run_parts[1])
    from scripts.run_heat3d_v7_formal_p1i_training import _resolve_model_config, _variant_model_config

    variant_model = _variant_model_config(dict(config["model"]), variant)
    # Feature names are immutable in the parent contract; the resulting
    # decoder indices are finalized again after train-only stats are fitted.
    feature_names = (
        "k_x", "k_y", "k_z", "q", "is_top", "is_bottom", "is_side", "is_interior",
        "top_h", "bottom_h", "top_T_inf_minus_T_ref", "bottom_T_fixed_minus_T_ref",
        "bottom_T_inf_minus_T_ref",
    )
    model_resolved = _resolve_model_config(variant_model, feature_names)
    return {
        "schema_version": "heat3d_v7_g1_h2_source_run_config_from_tracked_parent_v1",
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "graph": dict(config["graph"]),
        "model_parent": dict(config["model"]),
        "model_resolved_for_variant": model_resolved,
        "source": "tracked v7_g1_full_p1i.json plus registered variant delta; ignored native export config unavailable on this host",
    }


def _write_run_config(
    *,
    path: Path,
    run: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    raw_checkpoint_path: Path,
    source_run_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    stats_raw: Mapping[str, Any],
    provider: str,
    route: Mapping[str, Any],
    route_path: Path,
    decision_path: Path,
    governance_path: Path,
    capacity_manifest_path: Path,
    native_anchor_path: Path,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "heat3d_v7_g1_h2_run_config_provenance_v1",
            "run_id": str(run["run_id"]),
            "variant": str(run["variant"]),
            "seed": int(run["seed"]),
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "source_formal_receipt": str(raw_checkpoint_path.parent / "v7_g1_formal_receipt.json"),
            "source_formal_receipt_sha256": _sha256(raw_checkpoint_path.parent / "v7_g1_formal_receipt.json"),
            "source_best_checkpoint": str(raw_checkpoint_path),
            "source_best_checkpoint_sha256": _sha256(raw_checkpoint_path),
            "source_best_epoch": int(raw_receipt["checkpoint_selection"]["best_epoch"]),
            "source_run_config_sha256": _canonical_sha(source_run_config),
            "runtime_config": runtime_config,
            "resolved_model_config": model_config,
            "train_only_stats_sha256": _canonical_sha(stats_raw),
            "support_provider_id": provider,
            "route": route,
            "route_config_sha256": _sha256(route_path),
            "governance_amendment_path": str(governance_path),
            "governance_amendment_sha256": _sha256(governance_path),
            "capacity_manifest_path": str(capacity_manifest_path),
            "capacity_manifest_sha256": _sha256(capacity_manifest_path),
            "native_anchor_path": str(native_anchor_path),
            "native_anchor_sha256": _sha256(native_anchor_path),
            "primary_decision_path": str(decision_path),
            "primary_decision_sha256": _sha256(decision_path),
            "training_performed_by_h2": False,
            "optimizer_called_by_h2": False,
            "solver_called_by_h2": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )


def _execute_one(
    *,
    paths: Mapping[str, Path | dict[str, Any]],
    run: Mapping[str, Any],
    route: Mapping[str, Any],
    route_path: Path,
    decision: Mapping[str, Any],
    output_root: Path,
    max_samples: int | None,
    allow_existing_complete: bool,
) -> dict[str, Any]:
    from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample, _metrics_from_statistics, _sum_sufficient_statistics
    from rigno.heat3d_runtime.high_n import FullFieldGeometry, _mapping_sha256
    from rigno.heat3d_v6_full_field import build_reconstruction_map
    from rigno.heat3d_runtime.u_split import UHighNRuntime
    from rigno.heat3d_v6_p1i_anchor_query import HIGH_N_SELECTION_SEED
    from rigno.heat3d_training.p1i import legacy_train_only_stats

    repo = Path(paths["repo"])
    checkpoint_root = Path(paths["checkpoint_root"])
    run_id = str(run["run_id"])
    route_id = str(route["route_id"])
    run_output = output_root / route_id / run_id
    receipt_path = run_output / "evaluation_receipt.json"
    if allow_existing_complete and receipt_path.is_file():
        existing = _load_json(receipt_path)
        if existing.get("status") == "COMPLETE" and int(existing.get("sample_count", -1)) == VALID_COUNT:
            return existing
    if run_output.exists() and receipt_path.is_file():
        existing = _load_json(receipt_path)
        if existing.get("status") == "COMPLETE" and int(existing.get("sample_count", -1)) != VALID_COUNT:
            raise ValueError(f"{route_id}/{run_id}: refusing to mix partial/smoke output with full output")
    run_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path, raw_receipt = _checkpoint_and_receipt(
        checkpoint_root=checkpoint_root, run=run, launch=paths["launch_payload"]
    )
    source_config = _source_run_config(repo, run_id)
    train, valid, context_rows, provider = _load_examples(
        repo=repo,
        subset=Path(paths["subset"]),
        manifest=Path(paths["manifest"]),
        full_fields=Path(paths["full_fields"]),
        variant=str(run["variant"]),
        seed=int(run["seed"]),
    )
    from rigno.heat3d_runtime.high_n import FullFieldGeometry

    geometry = FullFieldGeometry.load(Path(paths["full_fields"]))
    valid_ids = [str(example.sample_id) for example in valid]
    if len(valid_ids) != VALID_COUNT or len(set(valid_ids)) != VALID_COUNT:
        raise ValueError(f"{run_id}: valid_iid population drifted")
    if any(str(geometry.split_roles[geometry.sample_ids.index(sample_id)]) != "valid_iid" for sample_id in valid_ids):
        raise ValueError(f"{run_id}: full-field truth role guard failed")
    runtime_session, runtime_config, stats_raw, model_config = _build_session(
        repo=repo,
        run=run,
        raw_receipt=raw_receipt,
        raw_checkpoint_path=checkpoint_path,
        source_run_config=source_config,
        train_examples=train,
        context_rows_by_id=context_rows,
        route=route,
    )
    capacity_manifest = paths["capacity_payload"]
    if not isinstance(capacity_manifest, Mapping):
        raise ValueError("native geometry capacity payload is missing")
    expected_native_config_sha = capacity_manifest["graph"]["formal_native_config_sha256"]
    if _canonical_sha(runtime_session.graph_config) != expected_native_config_sha:
        raise ValueError(f"{run_id}: H2 native graph config is not the frozen formal graph config")
    graph_sha = _sha256(repo / "rigno/graphBuilder_Heat3D.py")
    query_sha = _sha256(repo / "rigno/heat3d_v6_p1i_anchor_query.py")
    reconstruction_sha = _sha256(repo / "rigno/heat3d_v6_full_field.py")
    u_runtime_sha = _sha256(repo / "rigno/heat3d_runtime/u_split.py")
    high_n_sha = _sha256(repo / "rigno/heat3d_runtime/high_n.py")
    evaluation_sha = _sha256(repo / "rigno/heat3d_runtime/evaluation.py")
    benchmark_sha = _sha256(repo / "scripts/benchmark_heat3d_v6_p1i_resolution.py")
    selection_seed = int(_load_json(Path(paths["v6_binding"]))["nested_support"]["selection_seed"])
    if selection_seed != int(HIGH_N_SELECTION_SEED):
        raise ValueError("nested support selection seed drifted")
    _write_run_config(
        path=run_output / "run_config.json",
        run=run,
        raw_receipt=raw_receipt,
        raw_checkpoint_path=checkpoint_path,
        source_run_config=source_config,
        runtime_config=runtime_config,
        model_config=model_config,
        stats_raw=stats_raw,
        provider=provider,
        route=route,
        route_path=Path(paths["eu_contract"]),
        decision_path=Path(paths["decision"]),
        governance_path=Path(paths["governance"]),
        capacity_manifest_path=Path(paths["capacity_manifest"]),
        native_anchor_path=Path(paths["native_anchor"]),
    )
    _write_json(
        run_output / "implementation_provenance.json",
        {
            "schema_version": "heat3d_v7_g1_h2_implementation_provenance_v1",
            "formal_code_sha": FORMAL_CODE_SHA,
            "historical_v6_binding_path": "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json",
            "historical_v6_binding_sha256": _sha256(repo / "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json"),
            "historical_code_fingerprints": {
                "adapter_and_selector": "db2cc1f59a61419862ba8c58077a261ce51cb18e3561fae9a272a3c20f7a69c2",
                "full_kq_reconstruction": "9818b25f45210ff573e68338d85933d15d5545d8e93bde34ee6d9a47e523c59c",
                "graph_builder": "fce189e90aa3e182a418cd1ef50a9b5d24558fc3d24e50f9d6d1e734c3129cc3",
                "mesh_core": "d9a16ad59ffc4bb2c0bbf6457aaa53c1f5ce6d916d75a753b83fe15d8df9145a",
                "reconstruction": "8ffaa7680d1463c40fc57da9f3171cee75d0034e999854487b12a2f17be0e6d8",
            },
            "executed_checkout_sha256": {
                "anchor_query": query_sha,
                "benchmark_full_kq": benchmark_sha,
                "graph_builder": graph_sha,
                "reconstruction": reconstruction_sha,
                "u_split_runtime": u_runtime_sha,
                "high_n_runtime": high_n_sha,
                "evaluation_core": evaluation_sha,
            },
            "historical_binary_reconciliation_status": "historical high-resolution binary artifacts unavailable; executed frozen V7 semantic/runtime binding",
            "native_graph_anchor_sha256": str(paths["native_anchor_sha"]),
            "capacity_manifest_sha256": str(paths["capacity_sha"]),
            "governance_amendment_sha256": str(paths["governance_sha"]),
            "training_executed": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )
    u_runtime = UHighNRuntime.from_session(
        runtime_session,
        geometry,
        graph_builder_fingerprint=graph_sha,
    )
    count = VALID_COUNT if max_samples is None else int(max_samples)
    if count < 1 or count > VALID_COUNT:
        raise ValueError("--max-samples must be between 1 and 128")
    if max_samples is not None and count != VALID_COUNT:
        # A smoke run is deliberately kept outside the persistent archive and
        # can never be mistaken for a formal 128-sample result.
        pass
    query_resolution = int(route["output_query_resolution"])
    targets = route["fixed_edge_targets"]
    if not isinstance(targets, Mapping) or "native" not in targets or "query" not in targets:
        raise ValueError(f"{route_id}: U route fixed targets are not nested native/query mappings")
    native_targets = dict(targets["native"])
    query_targets = dict(targets["query"])
    full_predictions = np.empty((count, FULL_FIELD_RESOLUTION), dtype=np.float32)
    query_predictions = np.empty((count, query_resolution), dtype=np.float32)
    support_indices_matrix = np.empty((count, query_resolution), dtype=np.int32)
    metric_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    evaluation_core = EvaluationCore()
    started = time.perf_counter()
    for row_index, example in enumerate(valid[:count]):
        sample_started = time.perf_counter()
        fixture = _prepare_fixture(
            example=example,
            geometry=geometry,
            resolution=query_resolution,
            selection_seed=selection_seed,
        )
        support_indices = fixture["selected_indices"]
        support_indices_matrix[row_index] = support_indices.astype(np.int32)
        case = u_runtime.build_case(
            example,
            query_resolution,
            support=fixture["support"],
            native_edge_targets=native_targets,
            query_edge_targets=query_targets,
        )
        output = u_runtime.apply(case)
        jax.block_until_ready(output["raw_temperature"])
        raw = np.asarray(output["raw_temperature"], dtype=np.float64).reshape(-1)
        if raw.shape != (query_resolution,) or not np.all(np.isfinite(raw)):
            raise ValueError(f"{run_id}/{route_id}/{example.sample_id}: non-finite query prediction")
        query_delta = raw - 300.0
        if not np.all(np.isfinite(query_delta)):
            raise ValueError(f"{run_id}/{route_id}/{example.sample_id}: non-finite query deltaT")
        query_predictions[row_index] = query_delta.astype(np.float32)
        if route_id == "U_v2_direct240825":
            if query_resolution != FULL_FIELD_RESOLUTION or len(np.unique(support_indices)) != FULL_FIELD_RESOLUTION:
                raise ValueError("direct U route did not produce a full permutation query")
            full_prediction = np.empty(FULL_FIELD_RESOLUTION, dtype=np.float64)
            full_prediction[support_indices] = query_delta
            reconstruction_audit: dict[str, Any] = {
                "applied": False,
                "algorithm": "none; direct query is already 240825 common-domain field",
            }
        else:
            if query_resolution != U16384_RESOLUTION:
                raise ValueError("U16384 reconstruction route resolution drifted")
            mapping, reconstruction_audit = build_reconstruction_map(
                coords=geometry.coords,
                layer_id=geometry.layer_id,
                boundaries=fixture["layer_boundaries"],
                support_indices=support_indices.astype(np.int32),
                empty_domain_fallback="same_layer",
            )
            reconstruction_audit = dict(reconstruction_audit)
            reconstruction_audit.update(
                {
                    "mapping_hash": _mapping_sha256(mapping),
                    "support_hash": _array_sha256(support_indices.astype(np.int32)),
                    "boundary_hash": _array_sha256(fixture["layer_boundaries"]),
                }
            )
            full_prediction = mapping.reconstruct(query_delta)
            if full_prediction.shape != (FULL_FIELD_RESOLUTION,):
                raise ValueError("reconstructed full prediction shape drifted")
        if not np.all(np.isfinite(full_prediction)) or np.all(full_prediction == 0.0):
            raise ValueError(f"{run_id}/{route_id}/{example.sample_id}: invalid/zero-filled full prediction")
        full_predictions[row_index] = full_prediction.astype(np.float32)
        truth = u_runtime.geometry.valid_truth([str(example.sample_id)])[str(example.sample_id)]
        q = fixture["full_q"]
        source_mask = q > 0.0
        source_volume = float(np.sum(geometry.control_volume[source_mask]))
        if source_volume <= 0.0:
            raise ValueError(f"{example.sample_id}: source region is not estimable")
        evaluated = evaluation_core.evaluate(
            [EvaluationSample(
                sample_id=str(example.sample_id),
                prediction_deltaT_K=full_prediction,
                truth_deltaT_K=truth,
                control_volumes_m3=geometry.control_volume,
                coords=geometry.coords,
                layer_id=geometry.layer_id,
                q_W_m3=q,
                split="valid_iid",
            )]
        )
        row = dict(evaluated["per_sample"][0])
        if row["split"] != "valid_iid" or int(row["point_count"]) != FULL_FIELD_RESOLUTION:
            raise ValueError(f"{example.sample_id}: evaluation row contract drifted")
        metric_rows.append(row)
        support_rows.append(
            {
                "sample_id": str(example.sample_id),
                "anchor_indices_sha256": _array_sha256(fixture["anchor_indices"].astype(np.int32)),
                "nested_order_sha256": str(fixture["order_audit"]["order_sha256"]),
                "selected_indices_sha256": _array_sha256(support_indices.astype(np.int32)),
                "support_artifact_sha256": fixture["support"].sha256,
                "selected_resolution": int(query_resolution),
                "source_mask_sha256": _array_sha256(source_mask),
                "q_sha256": _array_sha256(q),
                "k_sha256": _array_sha256(fixture["full_k"]),
                "operator_control_volume_sha256": _array_sha256(fixture["effective_cv"]),
                "power_audit": fixture["power_audit"],
                "cv_audit": fixture["cv_audit"],
                "reconstruction_audit": reconstruction_audit,
                "source_node_count": int(np.count_nonzero(source_mask)),
                "source_volume_m3": source_volume,
                "prediction_all_finite": True,
                "prediction_zero_fill_detected": False,
                "elapsed_seconds": float(time.perf_counter() - sample_started),
            }
        )
        print(
            f"{route_id}/{run_id}: {row_index + 1}/{count} {example.sample_id} "
            f"source={np.sqrt(row['source_sse'] / row['source_volume']):.6g} "
            f"elapsed={time.perf_counter() - sample_started:.2f}s",
            flush=True,
        )
        del fixture, case, output, raw, query_delta, full_prediction, truth
        gc.collect()
    if len(metric_rows) != count or [str(row["sample_id"]) for row in metric_rows] != valid_ids[:count]:
        raise ValueError(f"{run_id}/{route_id}: metric population/order drifted")
    sufficient = _sum_sufficient_statistics(metric_rows)
    metrics = _metrics_from_statistics(sufficient)
    truth_hashes = {}
    for row in metric_rows:
        truth = u_runtime.geometry.valid_truth([str(row["sample_id"])])[str(row["sample_id"])]
        truth_hashes[str(row["sample_id"])] = _array_sha256(truth)
    common_contract = {
        "domain_id": COMMON_DOMAIN_ID,
        "point_count_per_sample": FULL_FIELD_RESOLUTION,
        "sample_count": count,
        "evaluation_split": "valid_iid",
        "sample_ids": valid_ids[:count],
        "coords_sha256": _array_sha256(geometry.coords),
        "control_volume_sha256": _array_sha256(geometry.control_volume),
        "layer_id_sha256": _array_sha256(geometry.layer_id),
        "q_sha256_by_sample": {row["sample_id"]: support_rows[index]["q_sha256"] for index, row in enumerate(metric_rows)},
        "source_mask_sha256_by_sample": {row["sample_id"]: support_rows[index]["source_mask_sha256"] for index, row in enumerate(metric_rows)},
        "truth_delta_sha256_by_sample": truth_hashes,
        "zero_fill_detected": False,
        "row_deletion_detected": False,
        "all_source_region_estimable": all(float(row["source_volume"]) > 0.0 for row in metric_rows),
        "exact_same_coordinates_masks_truth_contract": True,
    }
    common_contract_sha = _canonical_sha(common_contract)
    _write_json(run_output / "evaluation_contract.json", {**common_contract, "contract_sha256": common_contract_sha})
    _write_json(
        run_output / "per_sample_metrics.json",
        {
            "schema_version": "heat3d_v7_g1_h2_per_sample_metrics_v1",
            "run_id": run_id,
            "variant": str(run["variant"]),
            "seed": int(run["seed"]),
            "route_id": route_id,
            "domain_id": COMMON_DOMAIN_ID,
            "metric_contract": {
                "primary": H2_PRIMARY_METRIC,
                "secondary": [metric for metric in METRICS if metric != H2_PRIMARY_METRIC],
                "effect_definition": "ablation_error - Full_error",
            },
            "rows": metric_rows,
        },
    )
    _write_json(
        run_output / "support_reconstruction_provenance.json",
        {
            "schema_version": "heat3d_v7_g1_h2_support_reconstruction_provenance_v1",
            "route": route,
            "route_id": route_id,
            "query_generation": {
                "algorithm": "anchored_stratified_deficit_round_robin_v1",
                "selection_seed": selection_seed,
                "anchor_count": NATIVE_RESOLUTION,
                "query_resolution": query_resolution,
                "anchor_order": "original frozen 1024 solver-node order",
                "nested_prefix_rule": "first N entries after original anchors",
                "target_or_temperature_used": False,
            },
            "support_operator_measure": {
                "algorithm": "same_layer_nearest_solver_cv_partition_v1",
                "conservation_relative_tolerance": 1.0e-12,
                "all_selected_supports_positive_measure": True,
            },
            "query_normalization": {
                "algorithm": "native 1024 anchor bounding-box affine normalization",
                "clamped": False,
                "numerical_tolerance": 1.0e-6,
                "maximum_normalized_overshoot_cap": 0.25,
            },
            "reconstruction": {
                "algorithm": "layer_interface_aware_inverse_distance_v1",
                "resolution": FULL_FIELD_RESOLUTION,
                "empty_domain_fallback": "same_layer",
                "top_bottom_interface_neighbors": 4,
                "interior_layer_neighbors": 8,
                "distance_power": 2,
                "partition_of_unity": True,
                "direct_route_reconstruction_applied": route_id != "U_v2_direct240825",
            },
            "rows": support_rows,
        },
    )
    np.savez_compressed(
        run_output / "predictions_best.npz",
        sample_ids=np.asarray(valid_ids[:count], dtype="U128"),
        prediction_deltaT_K=full_predictions,
        split=np.asarray("valid_iid"),
        domain_id=np.asarray(COMMON_DOMAIN_ID),
        point_count=np.asarray(FULL_FIELD_RESOLUTION, dtype=np.int32),
    )
    np.savez_compressed(
        run_output / "query_support_indices_best.npz",
        sample_ids=np.asarray(valid_ids[:count], dtype="U128"),
        selected_indices=support_indices_matrix,
        query_resolution=np.asarray(query_resolution, dtype=np.int32),
        selection_seed=np.asarray(selection_seed, dtype=np.int64),
        route_id=np.asarray(route_id),
    )
    if route_id == "U_v2_16384_reconstruction":
        np.savez_compressed(
            run_output / "query_predictions_best.npz",
            sample_ids=np.asarray(valid_ids[:count], dtype="U128"),
            prediction_deltaT_K=query_predictions,
            query_resolution=np.asarray(query_resolution, dtype=np.int32),
            route_id=np.asarray(route_id),
        )
    _write_json(
        run_output / "evaluation_receipt.json",
        {
            "schema_version": "heat3d_v7_g1_h2_evaluation_receipt_v1",
            "status": "COMPLETE",
            "run_id": run_id,
            "variant": str(run["variant"]),
            "seed": int(run["seed"]),
            "checkpoint": "best_sample_first",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "selected_best_epoch": int(raw_receipt["checkpoint_selection"]["best_epoch"]),
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "training_git_commit": raw_receipt.get("git_commit"),
            "route": route,
            "route_id": route_id,
            "route_config_sha256": _sha256(Path(paths["eu_contract"])),
            "route_primary_decision_path": str(paths["decision"]),
            "route_primary_decision_sha256": _sha256(Path(paths["decision"])),
            "governance_amendment_sha256": str(paths["governance_sha"]),
            "capacity_manifest_sha256": str(paths["capacity_sha"]),
            "native_anchor_sha256": str(paths["native_anchor_sha"]),
            "domain_id": COMMON_DOMAIN_ID,
            "point_count_per_sample": FULL_FIELD_RESOLUTION,
            "query_resolution": query_resolution,
            "sample_count": count,
            "evaluation_split": "valid_iid",
            "valid_iid_sample_ids": valid_ids[:count],
            "common_domain_contract_sha256": common_contract_sha,
            "dataset_manifest_sha256": _sha256(Path(paths["manifest"])),
            "full_field_archive_sha256": _sha256(Path(paths["full_fields"])),
            "implementation_provenance": {
                "anchor_query_sha256": query_sha,
                "benchmark_full_kq_sha256": benchmark_sha,
                "graph_builder_sha256": graph_sha,
                "reconstruction_sha256": reconstruction_sha,
                "u_split_runtime_sha256": u_runtime_sha,
                "high_n_runtime_sha256": high_n_sha,
                "evaluation_core_sha256": evaluation_sha,
                "nested_selection_seed": selection_seed,
            },
            "metrics": metrics,
            "sufficient_statistics": sufficient,
            "metric_contract": {
                "primary": H2_PRIMARY_METRIC,
                "secondary": [metric for metric in METRICS if metric != H2_PRIMARY_METRIC],
                "source_mask": "q_W_m3 > 0",
                "truth_source": "frozen full-field archive valid_iid rows only",
                "coordinates": "shared frozen 240825 physical coordinates",
                "control_volumes": "shared frozen full-field control volumes",
            },
            "artifacts": {
                "per_sample_metrics": "per_sample_metrics.json",
                "predictions_best": "predictions_best.npz",
                "query_support_indices_best": "query_support_indices_best.npz",
                "query_predictions_best": "query_predictions_best.npz" if route_id == "U_v2_16384_reconstruction" else "predictions_best.npz; direct query is the common full field",
                "support_reconstruction_provenance": "support_reconstruction_provenance.json",
                "evaluation_contract": "evaluation_contract.json",
            },
            "integrity": {
                "all_source_region_estimable": True,
                "zero_fill_detected": False,
                "row_deletion_detected": False,
                "all_predictions_finite": True,
                "exact_common_coordinates_masks_truth": True,
            },
            "training_performed": False,
            "optimizer_called": False,
            "solver_called": False,
            "test_iid_access": False,
            "sealed_access": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    print(f"COMPLETE {route_id}/{run_id}: {count} samples in {time.perf_counter() - started:.1f}s", flush=True)
    return _load_json(receipt_path)


def _execute(args: argparse.Namespace) -> int:
    paths = _resolve_execute_paths(args)
    launch, prereg, _contract_payload, decision, capacity_payload, governance_sha, capacity_sha, anchor_sha = _verify_execute_contract(paths)
    del prereg
    route_ids = [args.route_id] if args.route_id else list(H2_ROUTES)
    route_payloads = {
        route_id: _amend_route_capacity(
            _registered_route(Path(paths["eu_contract"]), route_id),
            capacity_payload,
            capacity_sha,
        )
        for route_id in route_ids
    }
    paths = {
        **paths,
        "capacity_payload": capacity_payload,
        "governance_sha": governance_sha,
        "capacity_sha": capacity_sha,
        "native_anchor_sha": anchor_sha,
    }
    runs = [
        row for row in launch["runs"]
        if str(row.get("variant")) in VARIANTS and int(row.get("seed")) in SEEDS
    ]
    if args.run_id is not None:
        runs = [row for row in runs if str(row.get("run_id")) == str(args.run_id)]
        if not runs:
            raise ValueError(f"unknown H2 run id: {args.run_id}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "h2_execution_scope.json",
        {
            "schema_version": "heat3d_v7_g1_h2_execution_scope_v1",
            "status": "RUNNING",
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "preregistration_sha256": PREREG_SHA,
            "governance_amendment_sha256": governance_sha,
            "capacity_manifest_sha256": capacity_sha,
            "native_anchor_sha256": anchor_sha,
            "primary_route_id": decision["primary_route_id"],
            "robustness_route_id": decision["robustness_route_id"],
            "routes_requested": route_ids,
            "runs_requested": [str(row["run_id"]) for row in runs],
            "checkpoint_policy": "selected-best only; valid_iid.sample_first_relative_rmse_pct; no re-selection",
            "training_executed": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )
    completed = []
    for route_id in route_ids:
        route = route_payloads[route_id]
        route_dir = output_root / route_id
        route_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            completed.append(
                _execute_one(
                    paths=paths,
                    run=run,
                    route=route,
                    route_path=Path(paths["eu_contract"]),
                    decision=decision,
                    output_root=output_root,
                    max_samples=args.max_samples,
                    allow_existing_complete=args.allow_existing_complete,
                )
            )
    full_complete = all(int(row.get("sample_count", -1)) == VALID_COUNT for row in completed) and len(completed) == len(route_ids) * len(runs)
    _write_json(
        output_root / "h2_execution_scope.json",
        {
            "schema_version": "heat3d_v7_g1_h2_execution_scope_v1",
            "status": "COMPLETE" if full_complete else "SMOKE_ONLY",
            "formal_training_code_sha": FORMAL_CODE_SHA,
            "preregistration_sha256": PREREG_SHA,
            "primary_route_id": decision["primary_route_id"],
            "robustness_route_id": decision["robustness_route_id"],
            "routes_requested": route_ids,
            "run_count": len(runs),
            "route_count": len(route_ids),
            "completed_evaluation_count": len(completed),
            "full_128_sample_formal_count": sum(int(row.get("sample_count", -1)) == VALID_COUNT for row in completed),
            "checkpoint_policy": "selected-best only; valid_iid.sample_first_relative_rmse_pct; no re-selection",
            "training_executed": False,
            "optimizer_called": False,
            "solver_called": False,
            "test_iid_access": False,
            "sealed_access": False,
        },
    )
    return 0


def _metric_components(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "point_global_relative_rmse_pct":
        return np.asarray([float(row["point_sse"]) for row in rows]), np.asarray([float(row["point_truth_energy"]) for row in rows])
    if metric == "sample_first_relative_rmse_pct":
        return np.asarray([float(row["sample_cv_relative_rmse"]) for row in rows]), np.ones(len(rows))
    if metric == "raw_K_CV_RMSE_K":
        return np.asarray([float(row["cv_sse"]) for row in rows]), np.asarray([float(row["cv_volume"]) for row in rows])
    if metric == "source_region_RMSE_K":
        return np.asarray([float(row["source_sse"]) for row in rows]), np.asarray([float(row["source_volume"]) for row in rows])
    if metric == "peak_RMSE_K":
        return np.asarray([float(row["peak_error_squared"]) for row in rows]), np.ones(len(rows))
    if metric == "interface_RMSE_K":
        return np.asarray([float(row["interface_error_sum_squared"]) for row in rows]), np.asarray([float(row["interface_error_count"]) for row in rows])
    raise ValueError(f"unsupported H2 metric: {metric}")


def _aggregate(numerator: np.ndarray, denominator: np.ndarray, metric: str) -> float:
    n = float(np.sum(numerator, dtype=np.float64))
    d = float(np.sum(denominator, dtype=np.float64))
    if d <= 0.0:
        raise ValueError(f"{metric}: non-positive denominator")
    if metric == "point_global_relative_rmse_pct":
        return float(np.sqrt(n / d) * 100.0)
    if metric == "sample_first_relative_rmse_pct":
        return float(n / d * 100.0)
    return float(np.sqrt(n / d))


def _sample_metric(row: Mapping[str, Any], metric: str) -> float:
    n, d = _metric_components([row], metric)
    return _aggregate(n, d, metric)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("H2 distribution values must be non-empty and finite")
    order = np.argsort(-array, kind="mergesort")
    top = array[order[: min(10, array.size)]]
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "worst_10_mean": float(np.mean(top)),
        "worst_10_values_descending": [float(value) for value in top],
    }


def _bootstrap(
    full_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    ablation_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    full_num, full_den, abl_num, abl_den = [], [], [], []
    for seed in SEEDS:
        fn, fd = _metric_components(full_by_seed[seed], metric)
        an, ad = _metric_components(ablation_by_seed[seed], metric)
        if len(fn) != VALID_COUNT or len(an) != VALID_COUNT:
            raise ValueError(f"{metric}: bootstrap population drifted at seed {seed}")
        full_num.append(fn)
        full_den.append(fd)
        abl_num.append(an)
        abl_den.append(ad)
    full_num = np.stack(full_num)
    full_den = np.stack(full_den)
    abl_num = np.stack(abl_num)
    abl_den = np.stack(abl_den)
    effects = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        seed_draw = rng.integers(0, len(SEEDS), size=len(SEEDS))
        sample_draw = rng.integers(0, VALID_COUNT, size=(len(SEEDS), VALID_COUNT))
        full_value = _aggregate(
            full_num[seed_draw[:, None], sample_draw],
            full_den[seed_draw[:, None], sample_draw],
            metric,
        )
        abl_value = _aggregate(
            abl_num[seed_draw[:, None], sample_draw],
            abl_den[seed_draw[:, None], sample_draw],
            metric,
        )
        effects[index] = abl_value - full_value
    return {
        "enabled": True,
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "resampling_levels": ["seed", "valid_iid_sample_within_seed"],
        "seed_resampling_with_replacement": True,
        "sample_resampling_with_replacement": True,
        "interval": "percentile_95_percent_CI",
        "ci_low": float(np.percentile(effects, 2.5)),
        "ci_high": float(np.percentile(effects, 97.5)),
        "bootstrap_effect_mean": float(np.mean(effects)),
        "bootstrap_effect_median": float(np.median(effects)),
        "replicate_effect_sha256": _array_sha256(np.asarray(effects, dtype="<f8")),
    }


def _claim(bootstrap: Mapping[str, Any], median: float, seed_effects: Sequence[float]) -> str:
    if float(bootstrap["ci_low"]) > 0.0 and float(median) > 0.0 and all(float(value) > 0.0 for value in seed_effects):
        return "SUPERIORITY_SUPPORTED"
    return "DESCRIPTIVE_ONLY"


def _load_metric_payload(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("route_id") != expected["route_id"] or payload.get("variant") != expected["variant"] or int(payload.get("seed")) != int(expected["seed"]):
        raise ValueError(f"metric provenance drifted: {path}")
    if payload.get("domain_id") != COMMON_DOMAIN_ID:
        raise ValueError(f"metric domain drifted: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != VALID_COUNT:
        raise ValueError(f"metric sample count drifted: {path}")
    if any(row.get("split") != "valid_iid" or int(row.get("point_count", -1)) != FULL_FIELD_RESOLUTION for row in rows):
        raise ValueError(f"metric row contract drifted: {path}")
    return payload


def _analyze(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    prereg_path = (args.preregistration or repo / "configs/heat3d_v7/v7_g1_statistical_preregistration.json").resolve()
    decision_path = (args.primary_decision or repo / "configs/heat3d_v7/v7_g1_h2_route_primary_decision.json").resolve()
    governance_path = (args.governance_amendment or repo / "configs/heat3d_v7/v7_g1_h2_native_closeout_governance_amendment.json").resolve()
    capacity_path = (args.capacity_manifest or Path("/tmp/v7_g1_h2_native_closeout/g1_native_geometry_capacity_manifest.json")).resolve()
    native_anchor_path = (args.native_anchor or Path("/tmp/v7_g1_h2_native_closeout/g1_native_anchor_Full_seed0_v6p1if1_0993.json")).resolve()
    _amendment, governance_sha = _verify_governance_amendment(governance_path)
    capacity_payload, capacity_sha = _verify_native_capacity_manifest(capacity_path)
    native_anchor_sha = _verify_native_anchor(native_anchor_path, capacity_payload)
    prereg = _load_json(prereg_path)
    decision = _load_json(decision_path)
    if prereg.get("preregistration_sha256") != PREREG_SHA:
        raise ValueError("statistical preregistration SHA drifted")
    if decision.get("status") != "RESOLVED_BEFORE_RESULT_CALCULATION":
        raise ValueError("H2 route decision is not pre-result resolved")
    launch = _load_json((args.launch_manifest or repo / "configs/heat3d_v7/v7_g1_formal_launch_manifest.json").resolve())
    if launch.get("g1_formal_code_sha") != FORMAL_CODE_SHA or launch.get("test_iid_access") or launch.get("sealed_access"):
        raise ValueError("formal H2 analysis launch boundary drifted")
    runs = [row for row in launch.get("runs", []) if str(row.get("variant")) in VARIANTS and int(row.get("seed")) in SEEDS]
    if len(runs) != 9:
        raise ValueError("H2 analysis run population is not 9")
    route_ids = [args.route_id] if args.route_id else list(H2_ROUTES)
    if set(route_ids) != set(H2_ROUTES):
        raise ValueError("formal H2 analysis requires both frozen U routes")
    payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    domain_hashes: set[str] = set()
    for route_id in route_ids:
        for run in runs:
            run_id = str(run["run_id"])
            receipt_path = output_root / route_id / run_id / "evaluation_receipt.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"missing H2 evaluation receipt: {receipt_path}")
            receipt = _load_json(receipt_path)
            if receipt.get("status") != "COMPLETE" or int(receipt.get("sample_count", -1)) != VALID_COUNT:
                raise ValueError(f"incomplete H2 evaluation: {receipt_path}")
            if receipt.get("training_performed") or receipt.get("optimizer_called") or receipt.get("solver_called") or receipt.get("test_iid_access") or receipt.get("sealed_access"):
                raise ValueError(f"H2 safety flags drifted: {receipt_path}")
            if (
                receipt.get("governance_amendment_sha256") != governance_sha
                or receipt.get("capacity_manifest_sha256") != capacity_sha
                or receipt.get("native_anchor_sha256") != native_anchor_sha
            ):
                raise ValueError(f"H2 governance/native-capacity provenance drifted: {receipt_path}")
            metric_payload = _load_metric_payload(output_root / route_id / run_id / "per_sample_metrics.json", {"route_id": route_id, "variant": run["variant"], "seed": run["seed"]})
            contract = _load_json(output_root / route_id / run_id / "evaluation_contract.json")
            if not contract.get("all_source_region_estimable") or contract.get("zero_fill_detected") or contract.get("row_deletion_detected") or not contract.get("exact_same_coordinates_masks_truth_contract"):
                raise ValueError(f"H2 evaluation contract failed: {run_id}/{route_id}")
            domain_hashes.add(str(contract.get("contract_sha256")))
            payloads[(route_id, str(run["variant"]), int(run["seed"]))] = metric_payload
    if len(domain_hashes) != 1:
        raise ValueError(f"H2 route/variant common domain contract mismatch: {sorted(domain_hashes)}")
    common_domain_sha = next(iter(domain_hashes))
    effect_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    variant_summary: list[dict[str, Any]] = []
    route_comparison: list[dict[str, Any]] = []
    for route_id in route_ids:
        for variant in VARIANTS:
            for metric in METRICS:
                values = []
                per_seed_metric: dict[int, float] = {}
                for seed in SEEDS:
                    rows = payloads[(route_id, variant, seed)]["rows"]
                    value = _aggregate(*_metric_components(rows, metric), metric)
                    values.append(value)
                    per_seed_metric[seed] = value
                variant_summary.append({
                    "route_id": route_id,
                    "variant": variant,
                    "metric": metric,
                    "seed_values": [float(value) for value in values],
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "domain": COMMON_DOMAIN_ID,
                })
            if route_id == "U_v2_direct240825":
                for metric in (H2_PRIMARY_METRIC, "point_global_relative_rmse_pct", "peak_RMSE_K"):
                    direct_values = [next(row for row in variant_summary if row["route_id"] == route_id and row["variant"] == variant and row["metric"] == metric)["seed_values"][index] for index in range(len(SEEDS))]
                    recon_values = [next(row for row in variant_summary if row["route_id"] == "U_v2_16384_reconstruction" and row["variant"] == variant and row["metric"] == metric)["seed_values"][index] for index in range(len(SEEDS))]
                    route_comparison.append({
                        "variant": variant,
                        "metric": metric,
                        "definition": "U_v2_direct240825 - U_v2_16384_reconstruction; positive means direct route has higher error",
                        "direct_seed_values": direct_values,
                        "reconstruction_seed_values": recon_values,
                        "route_difference_seed_values": [float(a - b) for a, b in zip(direct_values, recon_values, strict=True)],
                        "route_difference_mean": float(np.mean(np.asarray(direct_values) - np.asarray(recon_values))),
                        "route_difference_sample_sd": float(np.std(np.asarray(direct_values) - np.asarray(recon_values), ddof=1)),
                    })
        if route_id not in route_ids:
            continue
        for ablation_variant in VARIANTS[1:]:
            for metric in METRICS:
                full_by_seed = {seed: payloads[(route_id, "Full", seed)]["rows"] for seed in SEEDS}
                abl_by_seed = {seed: payloads[(route_id, ablation_variant, seed)]["rows"] for seed in SEEDS}
                paired = []
                seed_effects = []
                per_seed_distribution = []
                for seed in SEEDS:
                    full_rows = full_by_seed[seed]
                    abl_rows = abl_by_seed[seed]
                    full_map = {str(row["sample_id"]): row for row in full_rows}
                    abl_map = {str(row["sample_id"]): row for row in abl_rows}
                    if set(full_map) != set(abl_map) or len(full_map) != VALID_COUNT:
                        raise ValueError(f"{route_id}/{ablation_variant}/seed{seed}: paired IDs drifted")
                    seed_pair = []
                    for sample_id in sorted(full_map):
                        full_error = _sample_metric(full_map[sample_id], metric)
                        abl_error = _sample_metric(abl_map[sample_id], metric)
                        effect = float(abl_error - full_error)
                        seed_pair.append(effect)
                        paired.append({
                            "route_id": route_id,
                            "variant": ablation_variant,
                            "metric": metric,
                            "seed": seed,
                            "sample_id": sample_id,
                            "full_error": full_error,
                            "ablation_error": abl_error,
                            "effect_ablation_minus_full": effect,
                            "same_sample_id": True,
                            "same_coordinate_grid": True,
                        })
                    full_value = _aggregate(*_metric_components(full_rows, metric), metric)
                    abl_value = _aggregate(*_metric_components(abl_rows, metric), metric)
                    seed_effect = float(abl_value - full_value)
                    seed_effects.append(seed_effect)
                    per_seed_distribution.append({
                        "seed": seed,
                        "full_value": full_value,
                        "ablation_value": abl_value,
                        "effect_ablation_minus_full": seed_effect,
                        "paired_sample_distribution": _distribution(seed_pair),
                    })
                pooled_full_rows = [row for seed in SEEDS for row in full_by_seed[seed]]
                pooled_abl_rows = [row for seed in SEEDS for row in abl_by_seed[seed]]
                pooled_full = _aggregate(*_metric_components(pooled_full_rows, metric), metric)
                pooled_abl = _aggregate(*_metric_components(pooled_abl_rows, metric), metric)
                paired_values = [row["effect_ablation_minus_full"] for row in paired]
                paired_distribution = _distribution(paired_values)
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                # Each formal route/metric comparison starts from the
                # preregistered seed; analysis order is therefore irrelevant
                # to any individual CI.
                bootstrap = _bootstrap(full_by_seed, abl_by_seed, metric, rng)
                claim = _claim(bootstrap, paired_distribution["median"], seed_effects) if metric == H2_PRIMARY_METRIC else "DESCRIPTIVE_ONLY"
                base = {
                    "hypothesis": "H2_generic" if ablation_variant == "layout_agnostic_stratified_support" else "H2_volume_only",
                    "comparison_id": "H2_Full_vs_layout_agnostic_stratified_support" if ablation_variant == "layout_agnostic_stratified_support" else "H2_Full_vs_cv_only_support",
                    "route_id": route_id,
                    "ablation_variant": ablation_variant,
                    "metric": metric,
                    "primary_metric": metric == H2_PRIMARY_METRIC,
                    "domain": COMMON_DOMAIN_ID,
                    "common_domain_contract_sha256": common_domain_sha,
                    "full_pooled_aggregate": pooled_full,
                    "ablation_pooled_aggregate": pooled_abl,
                    "effect_ablation_minus_full": float(pooled_abl - pooled_full),
                    "paired_sample_distribution": paired_distribution,
                    "per_seed_effects": seed_effects,
                    "per_seed": per_seed_distribution,
                    "bootstrap": bootstrap,
                    "claim_status": claim,
                    "claim_rule": "CI > 0, paired median > 0, and seed0/1/2 effects > 0" if metric == H2_PRIMARY_METRIC else "secondary descriptive only",
                }
                effect_rows.append(base)
                pooled_rows.append({key: base[key] for key in ("hypothesis", "comparison_id", "route_id", "ablation_variant", "metric", "primary_metric", "full_pooled_aggregate", "ablation_pooled_aggregate", "effect_ablation_minus_full", "paired_sample_distribution", "per_seed_effects", "claim_status")})
                bootstrap_rows.append({
                    "hypothesis": base["hypothesis"],
                    "comparison_id": base["comparison_id"],
                    "route_id": route_id,
                    "ablation_variant": ablation_variant,
                    "metric": metric,
                    "primary_metric": base["primary_metric"],
                    "bootstrap": bootstrap,
                    "paired_median": paired_distribution["median"],
                    "per_seed_effects": seed_effects,
                    "claim_status": claim,
                })
                per_seed_rows.extend({**row, "hypothesis": base["hypothesis"], "comparison_id": base["comparison_id"], "route_id": route_id, "ablation_variant": ablation_variant, "metric": metric} for row in per_seed_distribution)
                per_sample_rows.extend(paired)
                if metric == H2_PRIMARY_METRIC:
                    worst_rows.append({
                        "hypothesis": base["hypothesis"],
                        "comparison_id": base["comparison_id"],
                        "route_id": route_id,
                        "ablation_variant": ablation_variant,
                        "metric": metric,
                        "per_seed": per_seed_distribution,
                        "pooled": paired_distribution,
                    })
    primary_effect_rows = [row for row in effect_rows if row["primary_metric"]]
    _write_json(output_root / "h2_per_sample_effects.json", {"schema_version": "heat3d_v7_g1_h2_per_sample_effects_v1", "rows": per_sample_rows})
    _write_json(output_root / "h2_per_seed_effects.json", {"schema_version": "heat3d_v7_g1_h2_per_seed_effects_v1", "rows": per_seed_rows})
    _write_json(output_root / "h2_pooled_summaries.json", {"schema_version": "heat3d_v7_g1_h2_pooled_summaries_v1", "rows": pooled_rows})
    _write_json(output_root / "h2_bootstrap_ci_receipt.json", {
        "schema_version": "heat3d_v7_g1_h2_bootstrap_ci_receipt_v1",
        "status": "COMPLETE",
        "formal_code_sha": FORMAL_CODE_SHA,
        "preregistration_sha256": PREREG_SHA,
        "domain": COMMON_DOMAIN_ID,
        "primary_metric": H2_PRIMARY_METRIC,
        "effect_direction": "ablation_error - Full_error; positive favors Full",
        "routes": route_ids,
        "rows": bootstrap_rows,
    })
    _write_json(output_root / "h2_hypothesis_effect_table.json", {
        "schema_version": "heat3d_v7_g1_h2_hypothesis_effect_table_v1",
        "status": "COMPLETE",
        "primary_route_id": decision["primary_route_id"],
        "robustness_route_id": decision["robustness_route_id"],
        "primary_metric": H2_PRIMARY_METRIC,
        "rows": primary_effect_rows,
    })
    _write_json(output_root / "h2_variant_route_summary.json", {"schema_version": "heat3d_v7_g1_h2_variant_route_summary_v1", "rows": variant_summary})
    _write_json(output_root / "h2_route_comparison.json", {
        "schema_version": "heat3d_v7_g1_h2_route_comparison_v1",
        "definition": "U_v2_direct240825 - U_v2_16384_reconstruction; positive means direct route has higher error",
        "rows": route_comparison,
    })
    _write_json(output_root / "h2_worst_case_diagnostics.json", {"schema_version": "heat3d_v7_g1_h2_worst_case_diagnostics_v1", "rows": worst_rows})
    lines = [
        "# V7 G1 H2 full-field formal effect table",
        "",
        "Primary metric is `source_region_RMSE_K`. Effect is `ablation_error - Full_error`; positive favors Full. CI is the 10,000-replicate two-level percentile bootstrap CI.",
        "",
        "| Route | Comparison | Metric | Full pooled | Ablation pooled | Effect | Paired median | p90 | p95 | Worst-10 mean | Seed effects (0,1,2) | 95% CI | Claim status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in primary_effect_rows:
        dist = row["paired_sample_distribution"]
        ci = row["bootstrap"]
        lines.append(
            f"| `{row['route_id']}` | {row['comparison_id']} | `{row['metric']}` | "
            f"{row['full_pooled_aggregate']:.6g} | {row['ablation_pooled_aggregate']:.6g} | "
            f"{row['effect_ablation_minus_full']:.6g} | {dist['median']:.6g} | {dist['p90']:.6g} | "
            f"{dist['p95']:.6g} | {dist['worst_10_mean']:.6g} | "
            f"{', '.join(f'{value:.6g}' for value in row['per_seed_effects'])} | "
            f"[{ci['ci_low']:.6g}, {ci['ci_high']:.6g}] | {row['claim_status']} |"
        )
    (output_root / "h2_hypothesis_effect_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    route_lines = [
        "# V7 G1 H2 U-route robustness",
        "",
        "Route difference is `U_v2_direct240825 - U_v2_16384_reconstruction`; positive means direct has higher error on the same 240825 physical field.",
        "",
        "| Variant | Metric | Direct seed values | 16384→240825 seed values | Route difference seed values | Difference mean ± SD |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in route_comparison:
        route_lines.append(
            f"| {row['variant']} | `{row['metric']}` | "
            f"{', '.join(f'{v:.6g}' for v in row['direct_seed_values'])} | "
            f"{', '.join(f'{v:.6g}' for v in row['reconstruction_seed_values'])} | "
            f"{', '.join(f'{v:.6g}' for v in row['route_difference_seed_values'])} | "
            f"{row['route_difference_mean']:.6g} ± {row['route_difference_sample_sd']:.6g} |"
        )
    (output_root / "h2_route_comparison.md").write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    _write_json(
        output_root / "h2_analysis_receipt.json",
        {
            "schema_version": "heat3d_v7_g1_h2_analysis_receipt_v1",
            "status": "COMPLETE",
            "analysis_implementation_path": "scripts/closeout_heat3d_v7_g1_h2.py",
            "analysis_implementation_sha256": _sha256(Path(__file__).resolve()),
            "formal_code_sha": FORMAL_CODE_SHA,
            "preregistration_sha256": PREREG_SHA,
            "route_primary_decision_path": str(decision_path),
            "route_primary_decision_sha256": _sha256(decision_path),
            "governance_amendment_path": str(governance_path),
            "governance_amendment_sha256": governance_sha,
            "native_capacity_manifest_path": str(capacity_path),
            "native_capacity_manifest_sha256": capacity_sha,
            "native_anchor_path": str(native_anchor_path),
            "native_anchor_sha256": native_anchor_sha,
            "primary_route_id": decision["primary_route_id"],
            "robustness_route_id": decision["robustness_route_id"],
            "domain": COMMON_DOMAIN_ID,
            "point_count_per_sample": FULL_FIELD_RESOLUTION,
            "runs": 9,
            "routes": route_ids,
            "valid_iid_count_per_seed": VALID_COUNT,
            "checkpoint": "best; pre-registered valid_iid sample_first_relative_rmse_pct selection",
            "primary_metric": H2_PRIMARY_METRIC,
            "training_performed": False,
            "optimizer_called": False,
            "solver_called": False,
            "test_iid_access": False,
            "sealed_access": False,
            "all_source_region_estimable": True,
            "zero_fill_detected": False,
            "row_deletion_detected": False,
            "common_domain_contract_sha256": common_domain_sha,
            "claim_status": {
                "U_v2_16384_reconstruction": {
                    "generic": next(row["claim_status"] for row in primary_effect_rows if row["route_id"] == "U_v2_16384_reconstruction" and row["ablation_variant"] == "layout_agnostic_stratified_support"),
                    "cv_only": next(row["claim_status"] for row in primary_effect_rows if row["route_id"] == "U_v2_16384_reconstruction" and row["ablation_variant"] == "cv_only_support"),
                },
                "U_v2_direct240825": {
                    "generic": next(row["claim_status"] for row in primary_effect_rows if row["route_id"] == "U_v2_direct240825" and row["ablation_variant"] == "layout_agnostic_stratified_support"),
                    "cv_only": next(row["claim_status"] for row in primary_effect_rows if row["route_id"] == "U_v2_direct240825" and row["ablation_variant"] == "cv_only_support"),
                },
            },
        },
    )
    print(f"H2 analysis complete: {len(primary_effect_rows)} primary route/comparison rows", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if args.mode == "execute":
        return _execute(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
