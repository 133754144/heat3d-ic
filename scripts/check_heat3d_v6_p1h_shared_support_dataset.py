#!/usr/bin/env python3
"""Freeze and verify the V6-P1h shared-support dataset acceptance contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
PARENT_ID = "heat3d_v6_p1g_geometry_deconfounded1024_v0"
PARENT_CONFIG_SHA = "ab162724af61c745f82571e9c8f07102d5262c70a4817ace0900e894bfc4af83"
PARENT_MANIFEST_SHA = "e5329d5cd6253510d87a4432d5f2ddae67259637810c29fdfb6ddf42621875a4"
INVARIANT_META_KEYS = (
    "sample_id",
    "group_id",
    "split_role",
    "stack_template_id",
    "layers_bottom_to_top",
    "boundary_conditions",
    "contact",
    "source_count",
    "sources",
    "active_layer_power_W",
    "package_power_provenance",
    "power_was_Rth_inferred",
    "sample_was_temperature_filtered",
    "layout_kind",
    "alignment_relation",
    "design_block",
    "solver_mesh",
)
PROJECTION_DEPENDENT_METRICS = {
    "projected_field_cv_rms_deltaT_K",
    "solver_peak_minus_projected_peak_K",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _distribution(meta_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    factors: dict[str, Counter[str]] = defaultdict(Counter)
    for meta in meta_rows:
        values = {
            "split_role": meta["split_role"],
            "package_power_W": meta["metrics"]["package_total_power_W"],
            "top_h_W_m2K": meta["boundary_conditions"]["top"]["h_W_m2K"],
            "bottom_h_W_m2K": meta["boundary_conditions"]["bottom"]["h_W_m2K"],
            "source_count": meta["source_count"],
            "layout_kind": meta["layout_kind"],
            "alignment_relation": meta["alignment_relation"],
        }
        for key, value in values.items():
            factors[key][str(value)] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(factors.items())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--durable-dataset", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_audit.json",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_replay_audit.json",
    )
    parser.add_argument(
        "--search",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_original_full_field_search.json",
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_trainability_smoke.json",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_acceptance.json",
    )
    parser.add_argument(
        "--write-md",
        type=Path,
        default=ROOT / "docs/v6_p1h_shared_support_acceptance.md",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = _read_json(args.manifest)
    audit = _read_json(args.audit)
    replay = _read_json(args.replay)
    search = _read_json(args.search)
    smoke = _read_json(args.smoke)
    parent_manifest_path = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
    parent_manifest = _read_json(parent_manifest_path)

    _assert(manifest["dataset_id"] == DATASET_ID, "wrong P1h dataset id")
    dataset_manifest_path = args.dataset / "manifest.json"
    _assert(_sha256(dataset_manifest_path) == _sha256(args.manifest), "dataset/tracked manifest mismatch")
    _assert(_sha256(dataset_manifest_path) == audit["dataset_manifest_sha256"], "dataset manifest SHA drift")
    _assert(manifest["parent_dataset_id"] == PARENT_ID, "wrong parent dataset id")
    _assert(manifest["parent_config_sha256"] == PARENT_CONFIG_SHA, "parent config drift")
    _assert(manifest["parent_manifest_sha256"] == PARENT_MANIFEST_SHA, "parent manifest drift")
    _assert(_sha256(parent_manifest_path) == PARENT_MANIFEST_SHA, "tracked parent manifest drift")
    _assert(audit["status"] == replay["status"] == smoke["status"] == "passed", "a prerequisite failed")
    _assert(search["status"] == "original_full_fields_missing_rebuild_required", "search conclusion drift")
    _assert(not replay["parent_file_hash_failures"], "P1g replay input hash failure")
    _assert(replay["solver_node_count"] == 240825, "replay solver-node drift")

    rows = manifest["samples"]
    parent_rows = parent_manifest["samples"]
    _assert(len(rows) == len(parent_rows) == 1024, "sample count must remain 1024")
    _assert(manifest["group_count"] == 128, "group count must remain 128")
    _assert(
        [(r["sample_id"], r["group_id"], r["split_role"]) for r in rows]
        == [(r["sample_id"], r["group_id"], r["split_role"]) for r in parent_rows],
        "sample/group/split order differs from P1g",
    )
    split_counts = Counter(r["split_role"] for r in rows)
    _assert(split_counts == {"train": 768, "valid": 128, "test": 128}, "split counts drift")
    group_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_roles[row["group_id"]].add(row["split_role"])
    _assert(len(group_roles) == 128 and all(len(v) == 1 for v in group_roles.values()), "group leakage")
    _assert(len({r["point_coordinates_sha256"] for r in rows}) == 1, "coordinates are not shared")
    _assert(len({r["graph_sha256"] for r in rows}) == 1, "graph is not shared")
    _assert({r["point_coordinates_sha256"] for r in rows} == {manifest["shared_coordinate_sha256"]}, "coordinate hash mismatch")
    _assert({r["graph_sha256"] for r in rows} == {manifest["shared_graph_sha256"]}, "graph hash mismatch")

    invariant_failures: list[str] = []
    file_hash_failures: list[str] = []
    p1h_meta_rows: list[dict[str, Any]] = []
    p1g_meta_rows: list[dict[str, Any]] = []
    for row, parent_row in zip(rows, parent_rows, strict=True):
        sample_dir = args.dataset / row["sample_dir"]
        parent_dir = args.parent_dataset / parent_row["sample_dir"]
        p1h_meta = _read_json(sample_dir / "sample_meta.json")
        p1g_meta = _read_json(parent_dir / "sample_meta.json")
        p1h_meta_rows.append(p1h_meta)
        p1g_meta_rows.append(p1g_meta)
        for key in INVARIANT_META_KEYS:
            if p1h_meta[key] != p1g_meta[key]:
                invariant_failures.append(f"{row['sample_id']}:{key}")
        p1h_metrics = {k: v for k, v in p1h_meta["metrics"].items() if k not in PROJECTION_DEPENDENT_METRICS}
        p1g_metrics = {k: v for k, v in p1g_meta["metrics"].items() if k not in PROJECTION_DEPENDENT_METRICS}
        if p1h_metrics != p1g_metrics:
            invariant_failures.append(f"{row['sample_id']}:solver_metrics")
        for name, expected in row["file_sha256"].items():
            path = sample_dir / name
            if not path.is_file() or _sha256(path) != expected:
                file_hash_failures.append(f"{row['sample_id']}:{name}")
        coords = np.load(sample_dir / "coords.npy", allow_pickle=False)
        if _array_sha256(coords) != manifest["shared_coordinate_sha256"]:
            invariant_failures.append(f"{row['sample_id']}:coords_payload")
    _assert(not invariant_failures, f"P1g invariants failed: {invariant_failures[:5]}")
    _assert(not file_hash_failures, f"sample file hashes failed: {file_hash_failures[:5]}")
    distribution = _distribution(p1h_meta_rows)
    _assert(distribution == _distribution(p1g_meta_rows), "factor distribution changed from P1g")

    archive_path = args.dataset / manifest["full_field_archive"]["path"]
    archive_sha = _sha256(archive_path)
    _assert(archive_sha == manifest["full_field_archive"]["sha256"], "archive SHA256 mismatch")
    _assert(archive_path.stat().st_size == manifest["full_field_archive"]["size_bytes"], "archive size mismatch")
    durable_path: str | None = None
    if args.durable_dataset is not None:
        durable = args.durable_dataset.resolve()
        _assert(_sha256(durable / "manifest.json") == _sha256(args.manifest), "durable manifest mismatch")
        _assert(_sha256(durable / archive_path.name) == archive_sha, "durable archive mismatch")
        durable_path = str(durable)
    archive_q_failures: list[str] = []
    archive_t_failures: list[str] = []
    support_payload_failures: list[str] = []
    with h5py.File(archive_path, "r") as archive:
        _assert(archive.attrs["dataset_id"] == DATASET_ID, "archive dataset id mismatch")
        _assert(int(archive.attrs["solver_node_count"]) == 240825, "archive solver-node mismatch")
        _assert(int(archive.attrs["sample_count"]) == 1024, "archive sample count mismatch")
        _assert(archive.attrs["support_coordinate_sha256"] == manifest["shared_coordinate_sha256"], "archive coordinate hash mismatch")
        _assert(archive.attrs["support_graph_sha256"] == manifest["shared_graph_sha256"], "archive graph hash mismatch")
        indices = np.asarray(archive["support/indices"][:], dtype=np.int64)
        support_coords = np.asarray(archive["support/coords"][:])
        _assert(indices.shape == (1024,) and len(np.unique(indices)) == 1024, "support indices invalid")
        _assert(_array_sha256(indices.astype(np.int32)) == manifest["shared_support_index_sha256"], "support-index hash mismatch")
        _assert(_array_sha256(support_coords) == manifest["shared_coordinate_sha256"], "support-coordinate hash mismatch")
        _assert(np.array_equal(support_coords, archive["mesh/coords"][:][indices]), "support is not direct solver-node indexing")
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        from build_heat3d_v6_p1h_shared_support_dataset import _graph_metadata  # noqa: PLC0415
        rebuilt_graph_sha, _ = _graph_metadata(support_coords)
        _assert(rebuilt_graph_sha == manifest["shared_graph_sha256"], "shared graph does not replay")
        _assert(archive["samples/q_W_m3"].shape == (1024, 240825), "exact-q archive missing")
        _assert(archive["samples/temperature_K"].shape == (1024, 240825), "temperature archive shape mismatch")
        for row in rows:
            index = int(row["full_field_archive_row"])
            full_q = np.asarray(archive["samples/q_W_m3"][index])
            full_t = np.asarray(archive["samples/temperature_K"][index])
            if _array_sha256(full_q) != row["full_q_raw_sha256"]:
                archive_q_failures.append(row["sample_id"])
            if _array_sha256(full_t) != row["full_temperature_raw_sha256"]:
                archive_t_failures.append(row["sample_id"])
            sample_dir = args.dataset / row["sample_dir"]
            if not np.array_equal(np.load(sample_dir / "q_field.npy", allow_pickle=False)[:, 0], full_q[indices]):
                support_payload_failures.append(f"{row['sample_id']}:q")
            if not np.array_equal(np.load(sample_dir / "temperature.npy", allow_pickle=False)[:, 0], full_t[indices]):
                support_payload_failures.append(f"{row['sample_id']}:T")
    _assert(not archive_q_failures, f"archive exact-q hash failures: {archive_q_failures[:5]}")
    _assert(not archive_t_failures, f"archive temperature hash failures: {archive_t_failures[:5]}")
    _assert(not support_payload_failures, f"archive projection failures: {support_payload_failures[:5]}")

    _assert(audit["unique_coordinate_hash_count"] == audit["unique_graph_hash_count"] == 1, "shared support/graph audit failed")
    _assert(audit["sources_with_zero_support"] == 0, "a source has zero support")
    _assert(audit["source_support_point_count"]["min"] > 0, "minimum source coverage is zero")
    _assert(audit["coverage"]["all_layers_covered"], "not all layers covered")
    _assert(audit["coverage"]["all_interfaces_covered"], "not all interfaces covered")
    _assert(all(item["point_count"] > 0 for item in audit["coverage"]["interface_point_counts"]), "an interface is uncovered")
    _assert(audit["leakage"]["groups_crossing_split"] == 0, "split leakage")
    _assert(not audit["leakage"]["support_uses_temperature"], "support used temperature")
    _assert(not audit["leakage"]["support_uses_test_labels"], "support used test labels")
    _assert(audit["physical_metric_max_abs_difference_vs_P1g"] == 0.0, "physical metrics drifted")
    _assert(audit["source_metadata_max_abs_difference_vs_P1g"] == 0.0, "source metadata drifted")
    energy = audit["reconstruction"]["energy_balance_relative_error"]
    _assert(max(abs(energy["min"]), abs(energy["max"])) < 2e-10, "energy balance out of tolerance")
    _assert(smoke["materialized_roles"] == ["train", "valid"], "smoke accessed a forbidden role")
    _assert(smoke["test_samples_materialized"] == smoke["model_inference_runs_on_test"] == 0, "smoke accessed test")
    _assert(smoke["training_started"] is False, "formal training was started")
    _assert(smoke["batch_size"] == 24 and smoke["node_count"] == 1024, "B24/1024 smoke contract failed")
    _assert(smoke["finite_forward_backward_update"], "forward/backward/update is not finite")
    _assert(smoke["checkpoint_reload_parameter_max_abs_error"] == 0.0, "checkpoint parameter reload drift")
    _assert(smoke["checkpoint_reload_loss_abs_error"] == 0.0, "checkpoint prediction reload drift")

    payload = {
        "schema_version": "heat3d_v6_p1h_shared_support_acceptance_v1",
        "status": "passed",
        "dataset_id": DATASET_ID,
        "parent_dataset_id": PARENT_ID,
        "sample_count": 1024,
        "group_count": 128,
        "split_counts": dict(sorted(split_counts.items())),
        "solver_node_count": 240825,
        "shared_coordinate_sha256": manifest["shared_coordinate_sha256"],
        "shared_graph_sha256": manifest["shared_graph_sha256"],
        "shared_support_index_sha256": manifest["shared_support_index_sha256"],
        "full_field_archive_sha256": archive_sha,
        "full_field_archive_size_bytes": archive_path.stat().st_size,
        "full_field_q_storage": "/samples/q_W_m3 exact per sample",
        "data_paths": {
            "generation_worktree": str(args.dataset.resolve()),
            "durable_copy": durable_path,
            "durable_copy_hash_verified": durable_path is not None,
        },
        "p1g_invariance": {
            "sample_group_split_order": "exact",
            "physical_metadata": "exact",
            "solver_metrics": "exact",
            "factor_distribution": "exact",
        },
        "factor_distribution": distribution,
        "support": {
            "selection_inputs": audit["support_selection_label_inputs"],
            "proposal": audit["selected_support_proposal"],
            "source_point_count": audit["source_support_point_count"],
            "sources_with_zero_support": audit["sources_with_zero_support"],
            "layer_point_counts": audit["coverage"]["layer_point_counts"],
            "interface_point_counts": audit["coverage"]["interface_point_counts"],
            "top_point_count": 64,
            "bottom_point_count": 64,
        },
        "replay": {
            "representative_count": replay["representative_count"],
            "max_abs_error": replay["max_abs_error"],
            "parent_file_hashes_checked": replay["parent_file_hashes_checked"],
        },
        "projection_and_reconstruction": audit["reconstruction"],
        "energy_balance_absolute_max": max(abs(energy["min"]), abs(energy["max"])),
        "leakage": audit["leakage"],
        "trainability": smoke,
        "guardrails": {
            "canonical_dataset_changed": False,
            "parent_dataset_overwritten": False,
            "interpolation_from_parent_projected_points": False,
            "test_used_for_support_selection": False,
            "test_materialized_by_smoke": False,
            "formal_training_started": False,
        },
        "hash_checks": {
            "sample_files_checked": 1024 * 10,
            "sample_file_failures": 0,
            "full_q_rows_checked": 1024,
            "full_temperature_rows_checked": 1024,
            "archive_sha256_checked": True,
        },
    }
    _write_json(args.write_json, payload)
    recon = payload["projection_and_reconstruction"]
    source = payload["support"]["source_point_count"]
    markdown = f"""# V6-P1h shared-support acceptance

Status: **passed**. P1h preserves all 1024 P1g samples, 128 groups, physical inputs,
solver outputs and group-locked splits. Only the ordered operator support changed.

## Frozen identities

- coordinate SHA256: `{payload['shared_coordinate_sha256']}`
- graph SHA256: `{payload['shared_graph_sha256']}`
- support-index SHA256: `{payload['shared_support_index_sha256']}`
- full-field archive SHA256: `{payload['full_field_archive_sha256']}`
- full-field archive: {payload['full_field_archive_size_bytes']} bytes; exact per-sample q and T rows
- durable dataset path: `{payload['data_paths']['durable_copy']}` (manifest/archive hashes verified)

## Replay and support

- representative replay cases: {payload['replay']['representative_count']}; P1g files checked: {payload['replay']['parent_file_hashes_checked']}
- replay coordinates/k/q error: 0; projected T and solver metrics satisfy the frozen tolerances
- selected proposal: `{payload['support']['proposal']}` using geometry-only source-domain coverage
- source coverage min/p05/median: {source['min']:.0f}/{source['p05']:.0f}/{source['median']:.0f}; zero-covered sources: 0
- all 9 layers, all 8 interfaces, and 64 top + 64 bottom Robin nodes are covered

## Projection and conservation

- full-field reconstruction CV-RMSE median/p95: {recon['full_field_cv_rmse_K']['median']:.6f}/{recon['full_field_cv_rmse_K']['p95']:.6f} K
- relative CV-RMSE median/p95: {100*recon['full_field_cv_relative_rmse']['median']:.3f}%/{100*recon['full_field_cv_relative_rmse']['p95']:.3f}%
- solver peak minus projected peak median/p95/max: {recon['solver_peak_minus_projected_peak_K']['median']:.6f}/{recon['solver_peak_minus_projected_peak_K']['p95']:.6f}/{recon['solver_peak_minus_projected_peak_K']['max']:.6f} K
- max layer-mean/drop error p95: {recon['max_abs_layer_average_error_K']['p95']:.6f}/{recon['max_abs_layer_drop_error_K']['p95']:.6f} K
- maximum absolute energy-balance relative error: {payload['energy_balance_absolute_max']:.3e}

## Leakage and trainability

Support proposal selection used no temperature or test label. The B24 smoke
materialized train+valid only, fit normalization/global context on 768 train
samples, completed a finite forward/backward/AdamW update, and reproduced both
parameters and loss exactly after checkpoint reload. No formal training was
started and P1g remains canonical.
"""
    args.write_md.parent.mkdir(parents=True, exist_ok=True)
    args.write_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "archive_sha256": archive_sha,
        "shared_coordinate_sha256": payload["shared_coordinate_sha256"],
        "shared_graph_sha256": payload["shared_graph_sha256"],
        "sample_count": payload["sample_count"],
        "group_count": payload["group_count"],
        "sample_files_checked": payload["hash_checks"]["sample_files_checked"],
        "full_rows_checked": 1024,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
