#!/usr/bin/env python3
"""Inference-only fixed test_iid visualization export for the frozen G1 U-v2 route.

This is deliberately separate from all training/evaluation entry points.  The
caller supplies the already-frozen sample id and role from the publication
figure selection receipt.  It writes one exact 240825-node artifact and a
receipt; it never selects a checkpoint or computes a model-selection metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GRID = (65, 65, 57)
NODE_COUNT = int(np.prod(GRID))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_test_truth(geometry: Any, sample_id: str, role: str) -> np.ndarray:
    positions = {sid: i for i, sid in enumerate(geometry.sample_ids)}
    row = positions.get(sample_id)
    if row is None or geometry.split_roles[row] != role:
        raise ValueError(f"sample/role mismatch: {sample_id} / {role}")
    with h5py.File(geometry.path, "r") as archive:
        value = np.asarray(archive["samples/deltaT_K"][row], dtype=np.float64).reshape(-1)
    if value.shape != (NODE_COUNT,) or not np.isfinite(value).all():
        raise ValueError(f"invalid full-field truth for {sample_id}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-receipt", type=Path, required=True)
    parser.add_argument("--source-run-config", type=Path, required=True)
    parser.add_argument("--route-contract", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--role", choices=("valid_iid", "test_iid"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.role != "test_iid":
        raise SystemExit("this entry point is reserved for the authorized test visualization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(args.repo))

    import jax
    from rigno.heat3d_runtime.high_n import FullFieldGeometry
    from rigno.heat3d_runtime.u_split import UHighNRuntime
    from rigno.heat3d_v6_p1i_anchor_query import HIGH_N_SELECTION_SEED
    from rigno.heat3d_v6_dataset import Heat3DV6DualRobinDataset

    helper = load_module(args.helper, "v7_g1_h2_closeout_helper")
    raw_receipt = json.loads(args.formal_receipt.read_text(encoding="utf-8"))
    source_config = json.loads(args.source_run_config.read_text(encoding="utf-8"))
    geometry = FullFieldGeometry.load(args.full_fields)
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root,
        args.manifest,
        include_roles={args.role},
    )
    matches = [row for row in dataset.samples if str(row.sample_id) == args.sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {args.role} sample, found {len(matches)}")
    example = matches[0]
    if str(example.meta.get("v6_adapter", {}).get("manifest_split_role")) != args.role:
        raise ValueError("dataset role metadata drifted")

    train_examples, _valid_examples, context_rows, _provider = helper._load_examples(
        repo=args.repo,
        subset=args.subset,
        manifest=args.manifest,
        full_fields=args.full_fields,
        variant="Full",
        seed=args.seed,
    )
    contract = load_module(args.repo / "rigno/heat3d_runtime/preflight.py", "v7_preflight")
    route = contract.load_registered_route(args.route_contract, "U_v2_direct240825")
    route = contract.bind_registered_route(
        contract_path=args.route_contract,
        route_id="U_v2_direct240825",
        requested_strategy=str(route["strategy_name"]),
        anchor_context_resolution=int(route["anchor_context_resolution"]),
        encoder_input_resolution=int(route["encoder_input_resolution"]),
        output_query_resolution=int(route["output_query_resolution"]),
        reconstruction_resolution=int(route["reconstruction_resolution"]),
        fixed_edge_targets=route["fixed_edge_targets"],
    )
    capacity = json.loads(args.capacity_manifest.read_text(encoding="utf-8"))
    route = helper._amend_route_capacity(route, capacity, sha256_file(args.capacity_manifest))
    session, _runtime_config, _stats_raw, _model_config = helper._build_session(
        repo=args.repo,
        run={"run_id": f"Full_seed{args.seed}", "variant": "Full", "seed": args.seed},
        raw_receipt=raw_receipt,
        raw_checkpoint_path=args.checkpoint,
        source_run_config=source_config,
        train_examples=train_examples,
        context_rows_by_id=context_rows,
        route=route,
    )
    runtime = UHighNRuntime.from_session(
        session,
        geometry,
        graph_builder_fingerprint=sha256_file(args.repo / "rigno/graphBuilder_Heat3D.py"),
    )
    fixture = helper._prepare_fixture(
        example=example,
        geometry=geometry,
        resolution=240825,
        selection_seed=int(HIGH_N_SELECTION_SEED),
    )
    case = runtime.build_case(
        example,
        NODE_COUNT,
        support=fixture["support"],
        native_edge_targets=route["fixed_edge_targets"]["native"],
        query_edge_targets=route["fixed_edge_targets"]["query"],
    )
    output = runtime.apply(case)
    jax.block_until_ready(output["raw_temperature"])
    prediction = np.asarray(output["raw_temperature"], dtype=np.float64).reshape(-1) - 300.0
    if prediction.shape != (NODE_COUNT,) or not np.isfinite(prediction).all():
        raise ValueError("non-finite or wrong-shape U-v2 output")
    truth = read_test_truth(geometry, args.sample_id, args.role)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray(args.sample_id),
        role=np.asarray(args.role),
        coords=np.asarray(geometry.coords, dtype=np.float64),
        layer_id=np.asarray(geometry.layer_id, dtype=np.int32),
        truth=np.asarray(truth, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.float32),
    )
    receipt = {
        "schema_version": "heat3d_v7_g2_publication_test_visualization_inference_v1",
        "status": "PASS_VISUALIZATION_ONLY_TEST_INFERENCE",
        "visualization_only": True,
        "used_for_model_selection": False,
        "used_for_claim_or_color_policy": False,
        "sealed_access": False,
        "role": args.role,
        "sample_id": args.sample_id,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "formal_receipt_sha256": sha256_file(args.formal_receipt),
        "source_run_config_sha256": sha256_file(args.source_run_config),
        "route_contract_sha256": sha256_file(args.route_contract),
        "capacity_manifest_sha256": sha256_file(args.capacity_manifest),
        "full_field_archive_sha256": sha256_file(args.full_fields),
        "manifest_sha256": sha256_file(args.manifest),
        "grid": list(GRID),
        "layout": "canonical x_y_z flattened in shared full-field order",
        "temperature_space": "deltaT_K",
        "route_id": "U_v2_direct240825",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "prediction_sha256": sha256_array(prediction),
        "truth_sha256": sha256_array(truth),
        "coords_sha256": sha256_array(geometry.coords),
        "layer_id_sha256": sha256_array(geometry.layer_id),
        "test_iid_read": True,
        "training_started": False,
        "optimizer_called": False,
        "runner_repo_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
        ).strip(),
        "runner_script_sha256": sha256_file(Path(__file__)),
        "command": sys.argv,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "u_v2_audit": case.audit,
    }
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
