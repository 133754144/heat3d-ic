#!/usr/bin/env python3
"""Check the controlled P1i cross-resolution protocol and result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_heat3d_v6_p1i_controlled_cross_resolution.py"
EXECUTION_MANIFEST = ROOT / "configs/heat3d_v6_p1i/v6_p1i_controlled_cross_resolution_execution_manifest.json"
R0_PROTOCOL = ROOT / "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_protocol.json"
R0_EXECUTION_MANIFEST = ROOT / "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_execution_manifest.json"
DIAGNOSTIC_NAME = "measure_conservative_full_graph_rediscretization_diagnostic"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("controlled_cross_resolution", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load controlled cross-resolution module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {path}")


def check_protocol(protocol: dict[str, Any]) -> None:
    if protocol["status"] != "frozen_before_evaluation":
        raise RuntimeError("protocol is not frozen before evaluation")
    contract = protocol["role_contract"]
    if contract != {
        "allowed": ["valid_iid", "train_only_normalization_metadata"],
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "tuning_executed": False,
    }:
        raise RuntimeError("role contract drifted")
    if protocol["main_ladder"]["resolutions"] != [512, 1024, 2048, 4096, 8192, 16384]:
        raise RuntimeError("main resolution ladder drifted")
    if protocol["main_ladder"]["discretization_seeds"] != [0, 1, 2, 3]:
        raise RuntimeError("four-seed contract drifted")
    if protocol["factor_diagnostic"]["resolutions"] != [1024, 4096, 16384, 65536]:
        raise RuntimeError("factor resolution ladder drifted")
    if protocol["checkpoint"]["sha256"] != "51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e":
        raise RuntimeError("checkpoint SHA drifted")
    interpretation = protocol["factor_diagnostic"]["interpretation"]
    if "measure-conservative full-graph re-discretization diagnostic" not in interpretation:
        raise RuntimeError("Stage A diagnostic terminology drifted")
    if "not checkpoint-IID" not in interpretation or "not formal same-distribution invariance" not in interpretation:
        raise RuntimeError("Stage A applicability boundary missing")


def check_execution_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "completed_valid_only" or payload["formal_evaluator_commit"] != "9458900528bc84ab571e95682f7ae02b047ed9a2":
        raise RuntimeError("formal execution identity drifted")
    if payload["role_contract"] != {
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "tuning_executed": False,
    }:
        raise RuntimeError("execution role contract drifted")
    if len(payload["engineering_preflights"]) != 2 or not all(
        item["formal_result_used"] is False for item in payload["engineering_preflights"]
    ):
        raise RuntimeError("engineering failure provenance drifted")
    for artifact in payload["artifacts"]:
        artifact_path = ROOT / artifact["path"]
        if not artifact_path.is_file():
            raise RuntimeError(f"missing bound artifact: {artifact_path}")
        actual = file_sha256(artifact_path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"artifact SHA mismatch: {artifact_path}: {actual}")


def check_result(payload: dict[str, Any], module: Any, replay_data: bool) -> None:
    if payload["status"] != "passed" or len(payload["main"]) != 24 or len(payload["factors"]) != 16:
        raise RuntimeError("result cell count/status failed")
    contract = payload["contract"]
    if contract["test_accessed"] or contract["sealed_accessed"] or contract["training_executed"] or contract["tuning_executed"]:
        raise RuntimeError("forbidden role/action recorded")
    if contract.get("direct_N_interpretation") != DIAGNOSTIC_NAME:
        raise RuntimeError("result diagnostic terminology drifted")
    if contract.get("checkpoint_iid") is not False or contract.get("same_distribution_invariance_claimed") is not False:
        raise RuntimeError("result checkpoint-IID boundary drifted")
    main_keys = {
        (int(row["resolution"]), int(row["discretization_seed"])) for row in payload["main"]
    }
    if main_keys != {(n, seed) for n in module.MAIN_RESOLUTIONS for seed in module.DISCRETIZATION_SEEDS}:
        raise RuntimeError("main cell identity drift")
    factor_keys = {(str(row["factor_cell"]), int(row["resolution"])) for row in payload["factors"]}
    if factor_keys != {(cell, n) for cell in module.FACTOR_CELLS for n in module.FACTOR_RESOLUTIONS}:
        raise RuntimeError("factor cell identity drift")
    for row in [*payload["main"], *payload["factors"]]:
        if row["sample_count"] != 32 or row["sample_ids"] != payload["main"][0]["sample_ids"]:
            raise RuntimeError("fixed valid subset drift")
        if not row["contract"]["x_in_equals_x_out"]:
            raise RuntimeError("x_in=x_out contract failed")
        if row["contract"].get("direct_N_interpretation") != DIAGNOSTIC_NAME:
            raise RuntimeError("worker diagnostic terminology drifted")
        actual = row["regional_correction"]["actual_regional_counts"]
        if row["regional_mode"] == "fixed_training_nr" and not all(abs(int(value) - 256) <= 1 for value in actual):
            raise RuntimeError(f"fixed Nr drift: {actual}")
        for sample in row["samples"]:
            for family in ("p2r", "r2r", "r2p"):
                graph = sample["graph"][family]
                if graph["edge_count"] <= 0:
                    raise RuntimeError(f"zero edge count: {family}")
            # The production coverage contract is physical-node coverage:
            # every input physical node reaches p2r and every output physical
            # node receives r2p.  Refined regional centroids can be inactive in
            # those bipartite graphs and are reported rather than hidden.
            if sample["graph"]["p2r"]["out_degree"]["zero_count"]:
                raise RuntimeError("p2r misses physical input nodes")
            if sample["graph"]["r2p"]["in_degree"]["zero_count"]:
                raise RuntimeError("r2p misses physical output nodes")
            if (
                sample["graph"]["r2r"]["isolated_node_count"]
                or sample["graph"]["r2r"]["weakly_connected_components"] != 1
            ):
                raise RuntimeError("regional processor graph is disconnected")
            if row["support_mode"] == "source_aware":
                conservation = sample["conservation"]
                if conservation["relative_volume_error"] > 1e-12 or conservation["relative_source_power_error"] > 1e-12:
                    raise RuntimeError("physical conservation failed")
                if max(conservation["relative_cv_k_moment_error_xyz"]) > 1e-12:
                    raise RuntimeError("conductivity CV-moment conservation failed")
    finite_tree(payload)
    if replay_data and payload["nested_replay_binding"]["entry_count"] != 32 * 4 * 6:
        raise RuntimeError("nested replay binding count failed")


def check_r0_closeout(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "passed" or payload["diagnostic_name"] != "measure-conservative full-graph re-discretization diagnostic":
        raise RuntimeError("R0 closeout identity drifted")
    if payload["checkpoint_iid"] is not False or payload["formal_same_distribution_invariance"] is not False:
        raise RuntimeError("R0 applicability boundary drifted")
    contract = payload["contract"]
    if contract["test_accessed"] or contract["sealed_accessed"] or contract["training_executed"] or contract["tuning_executed"]:
        raise RuntimeError("R0 forbidden role/action recorded")
    if len(payload["resolution_rows"]) != 25:
        raise RuntimeError("R0 plus 24 re-discretization rows required")
    r0_rows = [row for row in payload["resolution_rows"] if row["reference_label"] == "R0"]
    if len(r0_rows) != 1 or int(r0_rows[0]["resolution"]) != 1024:
        raise RuntimeError("exact R0 reference missing")
    if any(abs(float(value)) > float(payload["r0"]["formal_tolerance"]) for domain in payload["r0"]["formal128_replay_differences"].values() for value in domain.values()):
        raise RuntimeError("R0 formal metric replay tolerance failed")
    if payload["r0_to_r1"]["graph_hash_equal_fraction"] != 0.0:
        raise RuntimeError("R0/R1 graph discontinuity unexpectedly absent")
    if len(payload["per_sample_drift"]) != 24:
        raise RuntimeError("all 24 diagnostic cells must carry R0-relative drift")
    if {
        (int(row["resolution"]), int(row["discretization_seed"]))
        for row in payload["per_sample_drift"]
    } != {
        (resolution, seed)
        for resolution in (512, 1024, 2048, 4096, 8192, 16384)
        for seed in (0, 1, 2, 3)
    }:
        raise RuntimeError("R0-relative drift cell identity failed")
    finite_tree(payload)


def check_r0_execution_archive(path: Path, closeout_path: Path, decoder_audit_path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_commit = "3eede1a192628e3bfc97f5f0be4b588ce4e0d7ed"
    if payload["status"] != "completed_valid_only":
        raise RuntimeError("R0 execution archive status drifted")
    if payload["preparation_commit"] != expected_commit or payload["execution_commit"] != expected_commit:
        raise RuntimeError("R0 preparation/execution commit drifted")
    execution = payload["execution"]
    if (
        execution["final_status"] != "passed"
        or execution["fixed32_gate"] != "passed"
        or execution["formal128_gate"] != "passed"
        or float(execution["formal_replay_absolute_tolerance"]) != 0.005
        or int(execution["resolution_row_count"]) != 25
        or int(execution["main_cell_count"]) != 24
    ):
        raise RuntimeError("R0 execution gate metadata drifted")
    if payload["role_contract"] != {
        "valid_only": True,
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "tuning_executed": False,
        "checkpoint_modified": False,
    }:
        raise RuntimeError("R0 execution role contract drifted")
    repaired = payload["repaired_protocol_failures"]
    if len(repaired) != 2 or any(
        row["status"] != "failed_closed_repaired" or row["scientific_result_used"] is not False
        for row in repaired
    ):
        raise RuntimeError("R0 repaired failure provenance drifted")
    if payload["post_gate_orchestration_fix"]["fix_commit"] != expected_commit:
        raise RuntimeError("R0 orchestration fix provenance drifted")
    decoder = payload["decoder_only_condition"]
    if (
        decoder["status"] != "failed_closed_decoder_only_absent"
        or decoder["randomblock_not_executed"] is not True
        or decoder["randomblock_result_used"] is not False
    ):
        raise RuntimeError("R0 random-block/decoder fail-closed status drifted")
    if ROOT / decoder["audit_path"] != decoder_audit_path.resolve():
        raise RuntimeError("R0 decoder audit binding drifted")

    artifact_paths = {row["path"] for row in payload["artifacts"]}
    required = {
        "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_closeout.json",
        "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_closeout.csv",
        "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_execution.log",
        "docs/v6_p1i_cross_resolution_r0_closeout.md",
        "docs/figures/v6_p1i_r0_resolution_error.png",
        "docs/figures/v6_p1i_r0_resolution_time.png",
    }
    if artifact_paths != required:
        raise RuntimeError("R0 archived artifact set drifted")
    for artifact in payload["artifacts"]:
        artifact_path = ROOT / artifact["path"]
        if not artifact_path.is_file():
            raise RuntimeError(f"missing R0 archived artifact: {artifact_path}")
        if artifact_path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"R0 artifact size mismatch: {artifact_path}")
        if file_sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"R0 artifact SHA mismatch: {artifact_path}")
    if closeout_path.resolve() != (ROOT / "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_closeout.json").resolve():
        raise RuntimeError("R0 closeout path is not the archived artifact")

    csv_path = ROOT / "configs/heat3d_v6_p1i/v6_p1i_cross_resolution_r0_closeout.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 25 or sum(row["reference_label"] == "R0" for row in csv_rows) != 1:
        raise RuntimeError("R0 CSV row contract failed")
    for row in csv_rows:
        for key in ("resolution", "support_point_global_pct", "full_point_global_pct", "worker_wall_seconds"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"R0 CSV non-finite field: {key}")

    report = (ROOT / "docs/v6_p1i_cross_resolution_r0_closeout.md").read_text(encoding="utf-8")
    if "R0 exact checkpoint replay" not in report or "R0 to R1 discontinuity" not in report:
        raise RuntimeError("R0 Markdown report sections missing")
    for png in (
        ROOT / "docs/figures/v6_p1i_r0_resolution_error.png",
        ROOT / "docs/figures/v6_p1i_r0_resolution_time.png",
    ):
        if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"invalid PNG signature: {png}")


def check_decoder_audit(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "failed_closed_decoder_only_absent" or payload["decoder_only_verified"] is not False:
        raise RuntimeError("decoder-only audit must fail closed")
    randomblock = payload["randomblock_condition"]
    if randomblock["execution_status"] != "not_executed_fail_closed" or int(randomblock["dedicated_checkpoint_count"]) != 0:
        raise RuntimeError("random-block conditional gate drifted")
    contract = payload["role_contract"]
    if any(contract.values()):
        raise RuntimeError("decoder-only audit performed a forbidden action")
    if len(payload["fail_closed_blockers"]) < 5:
        raise RuntimeError("decoder-only interface audit is incomplete")


def check_r0_protocol(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "frozen_before_evaluation":
        raise RuntimeError("R0 protocol was not frozen before evaluation")
    if payload["stage_a_name"] != "measure-conservative full-graph re-discretization diagnostic":
        raise RuntimeError("Stage A name drifted")
    if payload["stage_a_boundaries"] != {
        "checkpoint_iid": False,
        "formal_same_distribution_invariance": False,
        "model_selection_allowed": False,
        "production_speedup_claim_allowed": False,
    }:
        raise RuntimeError("Stage A boundaries drifted")
    if payload["decoder_only_condition"]["randomblock_execution_allowed"] is not False:
        raise RuntimeError("random-block execution must remain fail-closed")


def replay_supports(payload: dict[str, Any], module: Any, args: argparse.Namespace) -> None:
    data = module.base.FamilyData(
        family="p1i", dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=None,
    )
    rows = data.selected_rows(32)
    by_cell = {
        (int(item["discretization_seed"]), int(item["resolution"])): item
        for item in payload["main"]
    }
    shared = data.full_shared()
    coords = np.asarray(shared["coords"], dtype=np.float64)
    cv = np.asarray(shared["cv"], dtype=np.float64)
    layer = np.asarray(shared["layer"], dtype=np.int32)
    replay_count = 0
    for seed in module.DISCRETIZATION_SEEDS:
        for sample_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            meta = json.loads((data.sample_dir(row) / "sample_meta.json").read_text(encoding="utf-8"))
            sequences, _ = module.support_sequences(
                sample_id=sample_id, discretization_seed=seed, meta=meta,
                coords=coords, cv=cv, layer=layer,
            )
            prior_set = None
            for n in module.MAIN_RESOLUTIONS:
                indices, _, shortage = module.support_indices(sequences, n)
                if any(shortage.values()):
                    raise RuntimeError(f"formal N={n} quota shortage")
                current = set(map(int, indices))
                if prior_set is not None and not prior_set < current:
                    raise RuntimeError(f"{sample_id}/seed{seed}: N={n} not strictly nested")
                expected = by_cell[(seed, n)]["samples"][sample_index]["support_hash"]
                if module.array_sha256(indices) != expected:
                    raise RuntimeError(f"{sample_id}/seed{seed}/N={n}: support hash replay mismatch")
                prior_set = current
                replay_count += 1
    if replay_count != 32 * 4 * 6:
        raise RuntimeError("full deterministic replay count failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_controlled_cross_resolution_protocol.json",
    )
    parser.add_argument("--result", type=Path)
    parser.add_argument("--replay-data", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--full-fields", type=Path)
    parser.add_argument("--execution-manifest", type=Path, default=EXECUTION_MANIFEST)
    parser.add_argument("--r0-closeout", type=Path)
    parser.add_argument("--decoder-audit", type=Path)
    parser.add_argument("--r0-execution-manifest", type=Path, default=R0_EXECUTION_MANIFEST)
    parser.add_argument("--r0-protocol", type=Path, default=R0_PROTOCOL)
    args = parser.parse_args()
    module = load_module()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    check_protocol(protocol)
    check_r0_protocol(args.r0_protocol)
    # The legacy Stage-A execution manifest binds the checker version that
    # produced that historical bundle.  R0 closeout has its own immutable
    # execution manifest and must not rewrite the legacy provenance merely
    # because this checker gained R0 archive validation.
    if args.execution_manifest.exists() and args.r0_closeout is None:
        check_execution_manifest(args.execution_manifest)
    if args.result is not None:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
        check_result(payload, module, args.replay_data)
        if args.replay_data:
            if not all((args.dataset_root, args.manifest, args.full_fields)):
                parser.error("--replay-data requires dataset/full-field paths")
            replay_supports(payload, module, args)
    if args.r0_closeout is not None:
        check_r0_closeout(args.r0_closeout)
    if args.decoder_audit is not None:
        check_decoder_audit(args.decoder_audit)
    if args.r0_closeout is not None and args.decoder_audit is not None:
        check_r0_execution_archive(args.r0_execution_manifest, args.r0_closeout, args.decoder_audit)
    print(json.dumps({
        "status": "passed",
        "protocol": str(args.protocol),
        "result": None if args.result is None else str(args.result),
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
