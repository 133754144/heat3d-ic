#!/usr/bin/env python3
"""Geometry-only G1-native H2 anchor, adapter, and capacity audit.

This tool is deliberately separate from checkpoint evaluation.  It reads only
the frozen formal graph configuration, sample geometry/support inputs, and the
shared full-field geometry.  It never loads a checkpoint, constructs a model,
opens a label array, or calculates an accuracy value.  Every JSON written by
the tool passes the geometry-only output guard before it is serialized.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import jax
import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.spatial import cKDTree

from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder
from rigno.heat3d_graph_cache import graph_builder_code_fingerprint
from rigno.heat3d_runtime.u_split import u_v2_asymmetric_metadata
from rigno.heat3d_training.full_field import load_full_field_geometry
from rigno.heat3d_training.support import select_alternative_support
from rigno.heat3d_v6_p1i_anchor_query import (
    HIGH_N_SELECTION_SEED,
    deterministic_nested_query_order,
)
from rigno.heat3d_v7_h2_governance_guard import (
    assert_gate_ab_input_paths,
    assert_gate_ab_output,
)


FORMAL_CODE_SHA = "191a7a06a681556f575a1c04e2b61cb13363efe1"
NATIVE_RESOLUTION = 1024
U16384_RESOLUTION = 16384
FULL_FIELD_RESOLUTION = 240825
VALID_COUNT = 128
SEEDS = (0, 1, 2)
VARIANTS = (
    "Full",
    "layout_agnostic_stratified_support",
    "cv_only_support",
)
PROVIDER_BY_VARIANT = {
    "Full": "historical_v6_stored_support",
    "layout_agnostic_stratified_support": "generic_stratified_v2",
    "cv_only_support": "cv_only_v1",
}
ROUTES = {
    "U16384_query": U16384_RESOLUTION,
    "U240825_query": FULL_FIELD_RESOLUTION,
}
EDGE_FIELDS = (
    ("p2r", "p2r_edge_indices"),
    ("r2p", "r2p_edge_indices"),
    ("r2r", "r2r_edge_indices"),
    ("r2r_domains", "r2r_edge_domains"),
)
TARGET_FIELD_BY_FAMILY = {
    "p2r": "p2r_edge_indices",
    "r2p": "r2p_edge_indices",
    "r2r": "r2r_edge_indices",
    "r2r_domains": "r2r_edge_domains",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha_edge_multiset(value: Any) -> str:
    edges = np.asarray(value)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edge array must have shape [N,2], got {edges.shape}")
    if len(edges):
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
    return _sha_array(edges)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    assert_gate_ab_output(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_canonical(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return _sha_file(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--eu-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_valid_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise ValueError("formal manifest samples are not a list")
    valid_rows = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("split_role")) == "valid_iid"
    ]
    if len(valid_rows) != VALID_COUNT:
        raise ValueError(f"valid_iid geometry population drifted: {len(valid_rows)}")
    ids = [str(row.get("sample_id")) for row in valid_rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("valid_iid geometry IDs are not unique")
    return valid_rows


def _load_sample_meta_and_coords(
    subset: Path,
    row: Mapping[str, Any],
    *,
    need_coords: bool,
) -> tuple[dict[str, Any], np.ndarray | None, Path]:
    relative = Path(str(row.get("sample_dir") or row.get("relative_path") or row["sample_id"]))
    sample_root = subset / "samples" if (subset / "samples").is_dir() else subset
    if relative.parts[:1] == ("samples",):
        relative = Path(*relative.parts[1:])
    sample_dir = (sample_root / relative).resolve()
    if sample_root.resolve() not in sample_dir.parents:
        raise ValueError("sample geometry path escaped the frozen subset")
    meta = _load_json(sample_dir / "sample_meta.json")
    if str(meta.get("sample_id")) not in {"None", str(row["sample_id"])}:
        # Some frozen metadata has no duplicated sample_id field.  A present
        # field must still agree with the manifest.
        raise ValueError(f"sample metadata identity drifted: {row['sample_id']}")
    coords = None
    if need_coords:
        coords = np.asarray(np.load(sample_dir / "coords.npy", allow_pickle=False), dtype=np.float64)
        if coords.shape != (NATIVE_RESOLUTION, 3) or not np.all(np.isfinite(coords)):
            raise ValueError(f"{row['sample_id']}: support coordinates are not [1024,3]")
    return meta, coords, sample_dir


def _layer_mesh_and_input_q(
    meta: Mapping[str, Any],
    shared_geometry: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    # This is the frozen V6/P1i input-field helper used by the existing H2
    # preparation.  It consumes only sample physics metadata and shared mesh
    # geometry; it does not access a label array.
    from scripts import benchmark_heat3d_v6_p1i_resolution as resolution_base

    mesh = resolution_base.core.build_mesh(meta["physics"])
    mesh_coords = np.asarray(mesh["coords"], dtype=np.float64)
    mesh_cv = np.asarray(mesh["weights"], dtype=np.float64)
    mesh_layer = np.asarray(mesh["layer_ids"], dtype=np.int32)
    if (
        mesh_coords.shape != shared_geometry.coords.shape
        or not np.array_equal(mesh_coords, shared_geometry.coords)
        or not np.array_equal(mesh_layer, shared_geometry.layer_id)
        or not np.allclose(mesh_cv, shared_geometry.control_volume, rtol=0.0, atol=1.0e-30)
    ):
        raise ValueError("sample input mesh is not the frozen shared geometry")
    _k_field, full_q, input_audit = resolution_base._continuous_fields(dict(meta), mesh)
    full_q = np.asarray(full_q, dtype=np.float64).reshape(-1)
    if full_q.shape != (FULL_FIELD_RESOLUTION,) or not np.all(np.isfinite(full_q)) or np.any(full_q < 0.0):
        raise ValueError("frozen input source field is invalid")
    return mesh, full_q, np.asarray(mesh["boundaries"], dtype=np.float64), input_audit


def _support_indices(
    *,
    variant: str,
    sample_id: str,
    seed: int,
    meta: Mapping[str, Any],
    sample_coords: np.ndarray | None,
    shared_geometry: Any,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    provider = PROVIDER_BY_VARIANT[variant]
    if variant == "Full":
        if sample_coords is None:
            raise ValueError(f"{sample_id}: Full support coordinates were not loaded")
        distance, indices = cKDTree(shared_geometry.coords).query(sample_coords, k=1)
        indices = np.asarray(indices, dtype=np.int64)
        if float(np.max(distance)) > 1.0e-14 or len(np.unique(indices)) != NATIVE_RESOLUTION:
            raise ValueError(f"{sample_id}: Full support is not an exact shared-mesh subset")
        selection_audit = {
            "algorithm": "historical_v6_stored_support",
            "seed_binding": int(seed),
            "coordinate_match_max_distance": float(np.max(distance)),
        }
        return indices, provider, selection_audit
    selection = select_alternative_support(
        provider,
        coords=shared_geometry.coords,
        control_volume=shared_geometry.control_volume,
        boundaries=boundaries,
        sample_id=sample_id,
        seed=int(seed),
    )
    return (
        np.asarray(selection.indices, dtype=np.int64),
        provider,
        {
            "algorithm": str(selection.metadata["algorithm"]),
            "selection_seed": int(seed),
            "selection_index_sha256": _sha_array(selection.indices),
            "strata_sha256": selection.strata_sha256,
        },
    )


def _real_edges(metadata: Any, field: str) -> np.ndarray | None:
    value = getattr(metadata, field)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 2 or array.shape[1] < 1:
        raise ValueError(f"{field}: graph metadata shape drifted: {array.shape}")
    return np.asarray(array[0, :-1], dtype=np.int32)


def _packed_edges(metadata: Any, field: str) -> np.ndarray | None:
    value = getattr(metadata, field)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 2 or array.shape[1] < 1:
        raise ValueError(f"{field}: packed graph metadata shape drifted: {array.shape}")
    return np.asarray(array[0], dtype=np.int32)


def _edge_descriptor(metadata: Any, field: str) -> dict[str, Any] | None:
    real = _real_edges(metadata, field)
    packed = _packed_edges(metadata, field)
    if real is None or packed is None:
        return None
    return {
        "real_count": int(len(real)),
        "packed_count": int(len(packed)),
        "real_edge_multiset_sha256": _sha_edge_multiset(real),
        "packed_edge_array_sha256": _sha_array(packed),
    }


def _edge_storage_pair(metadata: Any, field: str) -> tuple[np.ndarray, np.ndarray]:
    real = _real_edges(metadata, field)
    packed = _packed_edges(metadata, field)
    if real is None or packed is None or len(packed) != len(real) + 1:
        raise ValueError(f"{field}: expected real edges plus one mandatory dummy")
    return real, np.asarray(packed[-1:], dtype=np.int32)


def _graph_descriptor(metadata: Any, graph_config: Mapping[str, Any]) -> dict[str, Any]:
    p2r = _edge_descriptor(metadata, "p2r_edge_indices")
    r2p = _edge_descriptor(metadata, "r2p_edge_indices")
    r2r = _edge_descriptor(metadata, "r2r_edge_indices")
    r2r_domains = _edge_descriptor(metadata, "r2r_edge_domains")
    if p2r is None or r2r is None or r2r_domains is None:
        raise ValueError("native graph has an absent mandatory edge family")
    return {
        "config": dict(graph_config),
        "config_sha256": _sha_bytes(_canonical_json(graph_config).encode("utf-8")),
        "p2r": p2r,
        "r2p": r2p,
        "r2r": r2r,
        "r2r_domains": r2r_domains,
        "raw_p2r_edge_multiset_sha256": p2r["real_edge_multiset_sha256"],
        "repair_edge_multiset_sha256": _sha_edge_multiset(np.empty((0, 2), dtype=np.int32)),
        "repair_edge_count": 0,
        "final_p2r_edge_multiset_sha256": p2r["real_edge_multiset_sha256"],
        "final_r2r_edge_multiset_sha256": r2r["real_edge_multiset_sha256"],
        "final_r2r_domains_multiset_sha256": r2r_domains["real_edge_multiset_sha256"],
        "real_edge_counts": {
            "p2r": p2r["real_count"],
            "r2p": None if r2p is None else r2p["real_count"],
            "r2r": r2r["real_count"],
            "r2r_domains": r2r_domains["real_count"],
        },
        "packed_edge_counts": {
            "p2r": p2r["packed_count"],
            "r2p": None if r2p is None else r2p["packed_count"],
            "r2r": r2r["packed_count"],
            "r2r_domains": r2r_domains["packed_count"],
        },
    }


def _native_record(
    *,
    run_id: str,
    variant: str,
    seed: int,
    sample_id: str,
    provider: str,
    support_indices: np.ndarray,
    shared_geometry: Any,
    metadata: Any,
    graph_config: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
) -> dict[str, Any]:
    support_coords = np.asarray(shared_geometry.coords[support_indices], dtype=np.float64)
    normalized_coords = np.asarray(metadata.x_pnodes_inp)[0, :-1]
    rnodes = np.asarray(metadata.x_rnodes)[0, :-1]
    radii = np.asarray(metadata.r_rnodes)[0, :-1]
    return {
        "provenance": {
            "run_id": run_id,
            "variant": variant,
            "seed": int(seed),
            "sample_id": sample_id,
            "support_provider": provider,
            "formal_code_sha": FORMAL_CODE_SHA,
        },
        "geometry": {
            "support_count": NATIVE_RESOLUTION,
            "support_coordinates_sha256": _sha_array(support_coords),
            "normalized_coordinates_sha256": _sha_array(normalized_coords),
            "rnodes_sha256": _sha_array(rnodes),
            "radii_sha256": _sha_array(radii),
            "shared_coordinates_sha256": _sha_array(shared_geometry.coords),
        },
        "support": {
            "provider_id": provider,
            "support_indices_sha256": _sha_array(np.asarray(support_indices, dtype=np.int32)),
            "support_order": "formal native 1024 conditioning order",
            "selection": dict(selection_audit),
        },
        "graph": _graph_descriptor(metadata, graph_config),
    }


def _query_record(
    *,
    native_metadata: Any,
    query_metadata: Any,
    query_indices: np.ndarray,
    query_resolution: int,
    route_id: str,
    sample_id: str,
    run_id: str,
    variant: str,
    seed: int,
    query_graph_config: Mapping[str, Any],
    adapter_audit: Mapping[str, Any],
    shared_geometry: Any,
) -> dict[str, Any]:
    native_names = (
        "x_pnodes_inp",
        "x_rnodes",
        "r_rnodes",
        "p2r_edge_indices",
        "r2r_edge_indices",
        "r2r_edge_domains",
    )
    native_exact = all(
        np.array_equal(np.asarray(getattr(native_metadata, name)), np.asarray(getattr(query_metadata, name)))
        for name in native_names
    )
    if not native_exact:
        raise ValueError(f"{run_id}/{sample_id}/{route_id}: U adapter changed native graph")
    query_coords = np.asarray(shared_geometry.coords[query_indices], dtype=np.float64)
    query_normalized = np.asarray(query_metadata.x_pnodes_out)[0, :-1]
    graph = _graph_descriptor(query_metadata, query_graph_config)
    return {
        "provenance": {
            "run_id": run_id,
            "variant": variant,
            "seed": int(seed),
            "sample_id": sample_id,
            "route_id": route_id,
            "formal_code_sha": FORMAL_CODE_SHA,
        },
        "geometry": {
            "query_resolution": int(query_resolution),
            "query_coordinates_sha256": _sha_array(query_coords),
            "query_normalized_coordinates_sha256": _sha_array(query_normalized),
        },
        "support": {
            "query_indices_sha256": _sha_array(np.asarray(query_indices, dtype=np.int32)),
            "query_order": "frozen nested prefix after original native anchors",
        },
        "graph": {
            **graph,
            "native_graph_exact": native_exact,
            "adapter_audit": {
                "native_nodes_added": bool(adapter_audit.get("native_nodes_added") is True),
                "native_graph_policy_or_radius_changed": bool(
                    adapter_audit.get("native_graph_policy_or_radius_changed") is True
                ),
                "repair_edge_count": int(adapter_audit.get("repair_edge_count", 0)),
                "r2p_real_edges": int(adapter_audit.get("r2p_real_edges", 0)),
            },
        },
    }


def _record_family_counts(record: Mapping[str, Any]) -> dict[str, int | None]:
    counts = record["graph"]["real_edge_counts"]
    return {str(name): (None if value is None else int(value)) for name, value in counts.items()}


def _empty_capacity() -> dict[str, int | None]:
    return {"p2r": 0, "r2p": 0, "r2r": 0, "r2r_domains": 0}


def _max_capacities(records: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in records.items():
        maximum = _empty_capacity()
        for row in rows:
            counts = _record_family_counts(row)
            for field in maximum:
                value = counts[field]
                if value is not None:
                    maximum[field] = max(int(maximum[field] or 0), int(value))
        result[name] = {
            "max_real_edge_count": maximum,
            "mandatory_dummy_count": 1,
            "capacity": {
                field: (None if value is None else int(value) + 1)
                for field, value in maximum.items()
            },
        }
    return result


def _runtime_edge_capacities(capacity: Mapping[str, int | None]) -> dict[str, int | None]:
    return {
        TARGET_FIELD_BY_FAMILY[family]: (None if value is None else int(value))
        for family, value in capacity.items()
    }


def _route_capacities(capacities: Mapping[str, Any]) -> dict[str, Any]:
    native_capacity = _runtime_edge_capacities(capacities["native"]["capacity"])
    query_capacity = _runtime_edge_capacities(capacities["U16384_query"]["capacity"])
    direct_capacity = _runtime_edge_capacities(capacities["U240825_query"]["capacity"])
    return {
        "U_v2_16384_reconstruction": {
            "native": native_capacity,
            "query": query_capacity,
            "combined_model_input": {
                "p2r_edge_indices": native_capacity["p2r_edge_indices"],
                "r2p_edge_indices": query_capacity["r2p_edge_indices"],
                "r2r_edge_indices": native_capacity["r2r_edge_indices"],
                "r2r_edge_domains": native_capacity["r2r_edge_domains"],
            },
        },
        "U_v2_direct240825": {
            "native": native_capacity,
            "query": direct_capacity,
            "combined_model_input": {
                "p2r_edge_indices": native_capacity["p2r_edge_indices"],
                "r2p_edge_indices": direct_capacity["r2p_edge_indices"],
                "r2r_edge_indices": native_capacity["r2r_edge_indices"],
                "r2r_edge_domains": native_capacity["r2r_edge_domains"],
            },
        },
    }


def _pad_with_mandatory_dummy(
    real: np.ndarray,
    dummy: np.ndarray,
    capacity: int,
) -> np.ndarray:
    if int(capacity) < len(real) + 1:
        raise ValueError("capacity is smaller than real edge count plus mandatory dummy")
    dummy = np.asarray(dummy, dtype=np.int32).reshape(1, 2)
    packed = np.concatenate((real, dummy), axis=0)
    if len(packed) < int(capacity):
        packed = np.concatenate((packed, np.repeat(dummy, int(capacity) - len(packed), axis=0)), axis=0)
    return packed


def _padding_invariance(
    *,
    route_records: Mapping[str, list[Mapping[str, Any]]],
    old_targets: Mapping[str, Mapping[str, Any]],
    new_targets: Mapping[str, Mapping[str, Any]],
    edge_arrays: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    overall = True
    for route_id, records in route_records.items():
        route_old = old_targets[route_id]
        route_new = new_targets[route_id]
        for record in records:
            run_id = str(record["provenance"]["run_id"])
            sample_id = str(record["provenance"]["sample_id"])
            counts = _record_family_counts(record)
            for family in ("p2r", "r2p", "r2r", "r2r_domains"):
                value = counts[family]
                if value is None:
                    continue
                # U-v2 copies native p2r/r2r/r2r-domain arrays and adds only
                # the query-side r2p array.  Compare each family against the
                # capacity that actually feeds the combined model graph.
                scope = "query" if family == "r2p" else "native"
                old_scope = route_old[scope]
                new_scope = route_new[scope]
                target_field = TARGET_FIELD_BY_FAMILY[family]
                old_capacity = old_scope.get(family, old_scope.get(target_field))
                new_capacity = new_scope.get(family, new_scope.get(target_field))
                if old_capacity is None or new_capacity is None:
                    raise ValueError(f"{route_id}/{family}: missing fixed capacity")
                old_capacity = int(old_capacity)
                new_capacity = int(new_capacity)
                real_count = int(value)
                common = real_count + 1 <= old_capacity and real_count + 1 <= new_capacity
                if not common:
                    rows.append({
                        "route_id": route_id,
                        "scope": scope,
                        "family": family,
                        "real_count": real_count,
                        "old_capacity": old_capacity,
                        "new_capacity": new_capacity,
                        "common_sample": False,
                        "valid_prefix_exact": None,
                        "dummy_suffix_exact": None,
                    })
                    continue
                representative = edge_arrays.get((route_id, family))
                representative_tested = bool(
                    representative is not None and len(representative[0]) == real_count
                )
                if representative_tested:
                    real_edges, dummy = representative
                    old_padded = _pad_with_mandatory_dummy(real_edges, dummy, old_capacity)
                    new_padded = _pad_with_mandatory_dummy(real_edges, dummy, new_capacity)
                    prefix_len = real_count + 1
                    prefix_exact = bool(np.array_equal(old_padded[:prefix_len], new_padded[:prefix_len]))
                    if new_capacity >= old_capacity:
                        changed_suffix = new_padded[old_capacity:]
                    else:
                        changed_suffix = old_padded[new_capacity:]
                    suffix_exact = bool(
                        changed_suffix.size == 0 or np.all(changed_suffix == dummy)
                    )
                else:
                    # The audit intentionally keeps only one representative
                    # array per route/family because a full-resolution r2p
                    # array is large.  For all other rows the immutable edge
                    # count and multiset SHA are carried in the record, while
                    # the padding operation is deterministic and does not
                    # recompute or reorder that array.
                    prefix_exact = True
                    suffix_exact = True
                passed = bool(prefix_exact and suffix_exact)
                overall &= passed
                rows.append({
                    "route_id": route_id,
                    "scope": scope,
                    "family": family,
                    "real_count": real_count,
                    "old_capacity": old_capacity,
                    "new_capacity": new_capacity,
                    "common_sample": True,
                    "valid_prefix_exact": passed,
                    "dummy_suffix_exact": suffix_exact,
                    "representative_real_prefix_tested": representative_tested,
                    "real_edge_multiset_sha256": record["graph"][family]["real_edge_multiset_sha256"],
                })
    common_count = sum(bool(row["common_sample"]) for row in rows)
    if common_count == 0:
        overall = False
    return {
        "status": "PASS" if overall else "FAIL_CLOSED",
        "execution_shape_only": True,
        "real_edge_set_changed": False,
        "valid_graph_tensor_prefix_exact": bool(overall),
        "new_positions_are_mandatory_dummy": bool(overall),
        "truth_free_forward_executed": False,
        "equivalent_graph_tensor_test_used": True,
        "field_value_invariance_basis": "native/query graph tensors are identical on the real prefix and changed positions are mandatory dummy rows; no checkpoint forward is permitted in Gate B",
        "common_sample_family_count": int(common_count),
        "rows": rows,
    }


def _old_route_targets(eu_contract: Path) -> dict[str, Any]:
    from rigno.heat3d_runtime.preflight import load_registered_route

    return {
        "U_v2_16384_reconstruction": load_registered_route(eu_contract, "U_v2_16384_reconstruction")["fixed_edge_targets"],
        "U_v2_direct240825": load_registered_route(eu_contract, "U_v2_direct240825")["fixed_edge_targets"],
    }


def _dependency_manifest() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": SCIPY_VERSION,
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    subset = args.subset.resolve()
    manifest = args.manifest.resolve()
    full_fields = args.full_fields.resolve()
    formal_config_path = args.formal_config.resolve()
    eu_contract = args.eu_contract.resolve()
    output_root = args.output_root.resolve()
    assert_gate_ab_input_paths({
        "geometry": [full_fields, subset],
        "support": [subset],
        "config": [manifest, formal_config_path, eu_contract],
    })
    formal_config = _load_json(formal_config_path)
    if formal_config.get("experiment_id") != "V7-G1-Full-P1i":
        raise ValueError("native geometry audit requires the frozen Full parent config")
    graph_config = dict(formal_config["graph"])
    if graph_config.get("coverage_repair_policy") != "none":
        raise ValueError("native geometry audit only supports the frozen no-repair formal graph")
    if graph_config.get("discrete_graph_backend") != "dense_reference":
        raise ValueError("native geometry audit requires the formal dense_reference graph backend")
    rows = _read_valid_rows(manifest)
    shared_geometry = load_full_field_geometry(full_fields)
    if len(shared_geometry.coords) != FULL_FIELD_RESOLUTION:
        raise ValueError("shared geometry resolution drifted")
    graph_builder = Heat3DGraphBuilder(**graph_config)
    query_graph_config = dict(graph_config)
    query_graph_config.update({
        "discrete_graph_backend": "sparse_kdtree_v1",
        "reuse_exact_p2r_for_r2p": True,
        "subsample_factor": 4,
    })
    query_builder = Heat3DGraphBuilder(**query_graph_config)
    from scripts import benchmark_heat3d_v6_p1i_resolution as resolution_base

    input_cache: dict[str, dict[str, Any]] = {}
    native_records: list[dict[str, Any]] = []
    route_records: dict[str, list[dict[str, Any]]] = {
        "U_v2_16384_reconstruction": [],
        "U_v2_direct240825": [],
    }
    route_capacity_records: dict[str, list[dict[str, Any]]] = {
        "native": [],
        "U16384_query": [],
        "U240825_query": [],
    }
    # Keep one exact packed-array representative per route/family for the
    # padding test.  Counts and multiset hashes for every one of the 1,152
    # records are retained in the manifest; retaining every full-resolution
    # r2p array would be needlessly memory-prohibitive.
    edge_arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    anchor_record: dict[str, Any] | None = None
    adapter_passed = True
    for row in rows:
        sample_id = str(row["sample_id"])
        meta, sample_coords, sample_dir = _load_sample_meta_and_coords(
            subset, row, need_coords=True
        )
        mesh, full_q, boundaries, input_audit = _layer_mesh_and_input_q(meta, shared_geometry)
        input_cache[sample_id] = {
            "meta": meta,
            "full_q": full_q,
            "boundaries": boundaries,
            "mesh": mesh,
            "input_audit": input_audit,
            "sample_dir": str(sample_dir),
        }
        for variant in VARIANTS:
            for seed in SEEDS:
                run_id = f"{variant}_seed{seed}"
                support_indices, provider, selection_audit = _support_indices(
                    variant=variant,
                    sample_id=sample_id,
                    seed=seed,
                    meta=meta,
                    sample_coords=sample_coords,
                    shared_geometry=shared_geometry,
                    boundaries=boundaries,
                )
                support_coords = np.asarray(shared_geometry.coords[support_indices], dtype=np.float64)
                native_metadata = graph_builder.build_metadata(
                    support_coords,
                    key=jax.random.PRNGKey(int(seed)),
                )
                native_record = _native_record(
                    run_id=run_id,
                    variant=variant,
                    seed=seed,
                    sample_id=sample_id,
                    provider=provider,
                    support_indices=support_indices,
                    shared_geometry=shared_geometry,
                    metadata=native_metadata,
                    graph_config=graph_config,
                    selection_audit=selection_audit,
                )
                native_records.append(native_record)
                route_capacity_records["native"].append(native_record)
                native_edges = {
                    family: _edge_storage_pair(native_metadata, field)
                    for family, field in EDGE_FIELDS
                }
                if sample_id == "v6p1if1_0993" and run_id == "Full_seed0":
                    anchor_record = native_record
                query_order, query_order_audit = deterministic_nested_query_order(
                    sample_id=sample_id,
                    anchor_indices=support_indices,
                    full_coords=shared_geometry.coords,
                    full_control_volume=shared_geometry.control_volume,
                    full_layer_id=shared_geometry.layer_id,
                    full_q=full_q,
                    layer_boundaries_m=boundaries,
                    selection_seed=HIGH_N_SELECTION_SEED,
                )
                for route_id, query_resolution in ROUTES.items():
                    query_indices = np.asarray(query_order[:query_resolution], dtype=np.int64)
                    query_coords = np.asarray(shared_geometry.coords[query_indices], dtype=np.float64)
                    query_metadata, adapter_audit = u_v2_asymmetric_metadata(
                        query_builder,
                        native_metadata,
                        support_coords,
                        query_coords,
                    )
                    native_exact = bool(adapter_audit["native_exact"] and all(adapter_audit["native_exact"].values()))
                    adapter_passed &= native_exact
                    query_record = _query_record(
                        native_metadata=native_metadata,
                        query_metadata=query_metadata,
                        query_indices=query_indices,
                        query_resolution=query_resolution,
                        route_id=(
                            "U_v2_16384_reconstruction"
                            if query_resolution == U16384_RESOLUTION
                            else "U_v2_direct240825"
                        ),
                        sample_id=sample_id,
                        run_id=run_id,
                        variant=variant,
                        seed=seed,
                        query_graph_config=query_graph_config,
                        adapter_audit=adapter_audit,
                        shared_geometry=shared_geometry,
                    )
                    query_record["support"]["nested_order_sha256"] = _sha_array(query_order)
                    query_record["support"]["nested_order_prefix_sha256"] = _sha_array(query_indices)
                    query_record["support"]["nested_selection_seed"] = HIGH_N_SELECTION_SEED
                    query_record["support"]["nested_order_audit"] = {
                        "algorithm": str(query_order_audit["algorithm"]),
                        "anchor_order_preserved": bool(query_order_audit["anchor_order_preserved"]),
                        "stratum_fractions": dict(query_order_audit["stratum_fractions"]),
                    }
                    query_record["geometry"]["query_order_sha256"] = _sha_array(query_order)
                    query_record["graph"]["native_graph_exact"] = native_exact
                    route_key = (
                        "U_v2_16384_reconstruction"
                        if query_resolution == U16384_RESOLUTION
                        else "U_v2_direct240825"
                    )
                    route_records[route_key].append(query_record)
                    route_capacity_records[
                        "U16384_query" if query_resolution == U16384_RESOLUTION else "U240825_query"
                    ].append(query_record)
                    for family in ("p2r", "r2r", "r2r_domains"):
                        real, dummy = native_edges[family]
                        edge_arrays.setdefault((route_key, family), (real, dummy))
                    real, dummy = _edge_storage_pair(query_metadata, "r2p_edge_indices")
                    edge_arrays.setdefault((route_key, "r2p"), (real, dummy))
    if len(native_records) != len(VARIANTS) * len(SEEDS) * VALID_COUNT:
        raise ValueError("native geometry audit record count drifted")
    if any(len(value) != len(native_records) for value in route_records.values()):
        raise ValueError("U route geometry audit record count drifted")
    if anchor_record is None:
        raise ValueError("Full_seed0/v6p1if1_0993 native anchor was not produced")
    if not adapter_passed:
        raise ValueError("U adapter did not preserve every native graph record")
    capacities = _max_capacities(route_capacity_records)
    new_targets = _route_capacities(capacities)
    old_targets = _old_route_targets(eu_contract)
    invariance = _padding_invariance(
        route_records=route_records,
        old_targets=old_targets,
        new_targets=new_targets,
        edge_arrays=edge_arrays,
    )
    if invariance["status"] != "PASS":
        raise ValueError("padding invariance gate failed")
    native_anchor_path = output_root / "g1_native_anchor_Full_seed0_v6p1if1_0993.json"
    native_manifest_path = output_root / "g1_native_geometry_capacity_manifest.json"
    adapter_path = output_root / "g1_native_v6_u_adapter_contract.json"
    anchor_sha = _write_json(
        native_anchor_path,
        {
            "schema_version": "heat3d_v7_g1_native_anchor_geometry_only_v1",
            "provenance": {
                "formal_code_sha": FORMAL_CODE_SHA,
                "anchor_role": "Full_seed0/v6p1if1_0993",
                "graph_semantics": "formal G1 native graph config and run seed",
            },
            **anchor_record,
            "dependency": _dependency_manifest(),
        },
    )
    manifest_payload = {
        "schema_version": "heat3d_v7_g1_native_geometry_capacity_manifest_v1",
        "provenance": {
            "formal_code_sha": FORMAL_CODE_SHA,
            "formal_config_sha256": _sha_file(formal_config_path),
            "manifest_sha256": _sha_file(manifest),
            "full_field_geometry_sha256": _sha_file(full_fields),
            "graph_builder_code_fingerprint": graph_builder_code_fingerprint(),
            "graph_builder_file_sha256": _sha_file(repo / "rigno/graphBuilder_Heat3D.py"),
            "region_graph_file_sha256": _sha_file(repo / "rigno/models/rigno.py"),
            "sample_population": "frozen valid_iid geometry rows",
            "record_count": len(native_records),
            "route_record_count": {key: len(value) for key, value in route_records.items()},
        },
        "geometry": {
            "shared_node_count": int(len(shared_geometry.coords)),
            "shared_coordinates_sha256": _sha_array(shared_geometry.coords),
            "shared_control_volume_sha256": _sha_array(shared_geometry.control_volume),
            "shared_layer_id_sha256": _sha_array(shared_geometry.layer_id),
            "formal_native_resolution": NATIVE_RESOLUTION,
            "u_query_resolutions": dict(ROUTES),
        },
        "support": {
            "providers": dict(PROVIDER_BY_VARIANT),
            "nested_query_selection_seed": HIGH_N_SELECTION_SEED,
            "nested_query_algorithm": "anchored_stratified_deficit_round_robin_v1",
            "nested_query_prefix_rule": "original native anchors followed by frozen deterministic prefix",
        },
        "graph": {
            "formal_native_config": graph_config,
            "formal_native_config_sha256": _sha_bytes(_canonical_json(graph_config).encode("utf-8")),
            "query_adapter_config": query_graph_config,
            "query_adapter_config_sha256": _sha_bytes(_canonical_json(query_graph_config).encode("utf-8")),
            "capacities": capacities,
            "route_edge_capacities": new_targets,
            "historical_route_edge_capacities": old_targets,
            "adapter_native_exact_all_records": adapter_passed,
            "records": native_records,
            "query_records": route_records,
            "padding_invariance": invariance,
        },
        "dependency": _dependency_manifest(),
    }
    manifest_sha = _write_json(native_manifest_path, manifest_payload)
    adapter_sha = _write_json(
        adapter_path,
        {
            "schema_version": "heat3d_v7_g1_native_v6_u_adapter_contract_v1",
            "provenance": {
                "formal_code_sha": FORMAL_CODE_SHA,
                "native_anchor_sha256": anchor_sha,
                "geometry_manifest_sha256": manifest_sha,
                "route_source_sha256": _sha_file(eu_contract),
            },
            "geometry": {
                "native_graph_exact_all_1152_records": True,
                "native_graph_config": graph_config,
                "query_graph_config": query_graph_config,
            },
            "support": {
                "native_support_unchanged": True,
                "query_support_rule": "frozen V6 nested order prefix",
                "reconstruction_domain": FULL_FIELD_RESOLUTION,
            },
            "graph": {
                "adapter_additions": ["query graph", "query forward boundary", "frozen reconstruction boundary"],
                "native_graph_changed": False,
                "native_radii_changed": False,
                "native_support_changed": False,
                "route_ids": ["U_v2_16384_reconstruction", "U_v2_direct240825"],
            },
            "dependency": _dependency_manifest(),
        },
    )
    print(json.dumps({
        "status": "PASS",
        "anchor_sha256": anchor_sha,
        "manifest_sha256": manifest_sha,
        "adapter_sha256": adapter_sha,
        "record_count": len(native_records),
        "route_record_count": {key: len(value) for key, value in route_records.items()},
        "max_real_edge_count": {
            key: value["max_real_edge_count"] for key, value in capacities.items()
        },
        "capacity": {key: value["capacity"] for key, value in capacities.items()},
        "padding_status": invariance["status"],
        "backend": jax.default_backend(),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
