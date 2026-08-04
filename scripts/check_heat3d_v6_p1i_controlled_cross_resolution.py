#!/usr/bin/env python3
"""Check the controlled P1i cross-resolution protocol and result bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_heat3d_v6_p1i_controlled_cross_resolution.py"


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


def check_result(payload: dict[str, Any], module: Any, replay_data: bool) -> None:
    if payload["status"] != "passed" or len(payload["main"]) != 24 or len(payload["factors"]) != 16:
        raise RuntimeError("result cell count/status failed")
    contract = payload["contract"]
    if contract["test_accessed"] or contract["sealed_accessed"] or contract["training_executed"] or contract["tuning_executed"]:
        raise RuntimeError("forbidden role/action recorded")
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
        actual = row["regional_correction"]["actual_regional_counts"]
        if row["regional_mode"] == "fixed_training_nr" and not all(abs(int(value) - 256) <= 1 for value in actual):
            raise RuntimeError(f"fixed Nr drift: {actual}")
        for sample in row["samples"]:
            for family in ("p2r", "r2r", "r2p"):
                graph = sample["graph"][family]
                if graph["edge_count"] <= 0 or graph["in_degree"]["zero_count"] or graph["out_degree"]["zero_count"]:
                    raise RuntimeError(f"zero edge/degree: {family}")
            if row["support_mode"] == "source_aware":
                conservation = sample["conservation"]
                if conservation["relative_volume_error"] > 1e-12 or conservation["relative_source_power_error"] > 1e-12:
                    raise RuntimeError("physical conservation failed")
                if max(conservation["relative_cv_k_moment_error_xyz"]) > 1e-12:
                    raise RuntimeError("conductivity CV-moment conservation failed")
    finite_tree(payload)
    if replay_data and payload["nested_replay_binding"]["entry_count"] != 32 * 4 * 6:
        raise RuntimeError("nested replay binding count failed")


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
    args = parser.parse_args()
    module = load_module()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    check_protocol(protocol)
    if args.result is not None:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
        check_result(payload, module, args.replay_data)
        if args.replay_data:
            if not all((args.dataset_root, args.manifest, args.full_fields)):
                parser.error("--replay-data requires dataset/full-field paths")
            replay_supports(payload, module, args)
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
