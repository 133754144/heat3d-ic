#!/usr/bin/env python3
"""V7 stable-runtime U-v2 high-resolution direct-query inference.

This is a reference entrypoint, not a benchmark.  It accepts only existing
support artifacts and the valid_iid split; it does not write predictions,
generate data, invoke a solver, or read target labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from rigno.heat3d_runtime import (
    FullFieldGeometry,
    RuntimeSession,
    SupportArtifact,
    UHighNRuntime,
    bind_registered_route,
)
from rigno.heat3d_v6_dataset import (
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
from rigno.heat3d_v6_p1i_anchor_query import array_sha256


def _edge_targets(path: Path, section: str) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "graph_cache" in payload:
        values = payload["graph_cache"]["edge_targets"]
    else:
        values = payload["padding"]["actual_padding_envelope"][section]
    if not isinstance(values, dict) or not values:
        raise ValueError(f"padding envelope is not a non-empty mapping: {path}:{section}")
    return {str(key): None if value is None else int(value) for key, value in values.items()}


def _combined_edge_targets(
    native_targets: dict[str, int | None],
    query_targets: dict[str, int | None],
) -> dict[str, int | None]:
    return {
        "p2r_edge_indices": native_targets["p2r_edge_indices"],
        "r2p_edge_indices": query_targets["r2p_edge_indices"],
        "r2r_edge_domains": query_targets["r2r_edge_domains"],
        "r2r_edge_indices": query_targets["r2r_edge_indices"],
    }


def _support_path(root: Path, resolution: int, sample_id: str) -> Path:
    candidates = (
        root / str(resolution) / f"{sample_id}.npz",
        root / "support" / str(resolution) / f"{sample_id}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no pre-existing support artifact for {sample_id} at {resolution}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--eu-contract", type=Path, required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--anchor-context-resolution", type=int, required=True)
    parser.add_argument("--encoder-input-resolution", type=int, required=True)
    parser.add_argument("--output-query-resolution", type=int, required=True)
    parser.add_argument("--reconstruction-resolution", type=int, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--native-padding", type=Path, required=True)
    parser.add_argument("--query-padding", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(16384, 32768, 240825), required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--split", choices=("valid_iid",), default="valid_iid")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root, args.manifest, include_roles={"valid_iid"}
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise ValueError("V7 U-v2 reference requires frozen V6/P1i continuous dataset")
    examples = list(dataset.samples)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        examples = examples[: int(args.max_samples)]
    native_targets = _edge_targets(args.native_padding, "native")
    query_targets = _edge_targets(args.query_padding, "query")
    if int(args.output_query_resolution) != int(args.resolution):
        raise ValueError("--resolution must equal --output-query-resolution")
    requested_targets = {
        "native": native_targets,
        "query": query_targets,
        "combined_model_input": _combined_edge_targets(native_targets, query_targets),
    }
    route_contract = bind_registered_route(
        contract_path=args.eu_contract,
        route_id=str(args.route_id),
        requested_strategy=str(args.strategy),
        anchor_context_resolution=int(args.anchor_context_resolution),
        encoder_input_resolution=int(args.encoder_input_resolution),
        output_query_resolution=int(args.output_query_resolution),
        reconstruction_resolution=int(args.reconstruction_resolution),
        fixed_edge_targets=requested_targets,
    )
    session = RuntimeSession.from_paths(
        args.checkpoint,
        args.run_config,
        execution_role="production_inference",
        route_contract=route_contract,
    )
    geometry = FullFieldGeometry.load(args.full_fields)
    runtime = UHighNRuntime.from_session(session, geometry)
    cases = []
    for anchor in examples:
        support = SupportArtifact.load(
            _support_path(args.support_root, args.resolution, str(anchor.sample_id)),
            expected_resolution=args.resolution,
        )
        case = runtime.build_case(
            anchor,
            args.resolution,
            support=support,
            native_edge_targets=native_targets,
            query_edge_targets=query_targets,
        )
        output = runtime.apply(case)
        jax.block_until_ready(output["raw_temperature"])
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
        scale = float(np.asarray(output["s_hat"], dtype=np.float64).reshape(-1)[0])
        cases.append(
            {
                "sample_id": str(anchor.sample_id),
                "route_id": route_contract["route_id"],
                "strategy": route_contract["strategy_name"],
                "anchor_context_resolution": 1024,
                "encoder_input_resolution": 1024,
                "output_query_resolution": int(args.resolution),
                "reconstruction_resolution": 240825,
                "direct_query": True,
                "support_indices_sha256": support.descriptor()["selected_indices_sha256"],
                "query_coordinates_sha256": array_sha256(np.asarray(case.query.condition.coords)),
                "raw_prediction_sha256": array_sha256(raw),
                "prediction_representation": "absolute_temperature_K",
                "reference_temperature_K": 300.0,
                "prediction_stage": "model_raw",
                "predicted_scale": scale,
                "native_edge_targets": native_targets,
                "query_edge_targets": query_targets,
                "runtime_audit": case.audit,
            }
        )
    return {
        "status": "inference_complete",
        "execution_role": "production_inference",
        "experiment_role": "v7_u_v2_reference_inference",
        "dataset_role": "valid_iid",
        "dataset_id": dataset.manifest["dataset_id"],
        "sample_count": len(cases),
        "temperature_representation_contract": {
            "model_output": "absolute_temperature_K",
            "formal_evaluation_adapter_required": "deltaT_K = temperature_K - reference_temperature_K",
            "formal_evaluation_input": "deltaT_K",
            "reference_temperature_K": 300.0,
        },
        "route_id": route_contract["route_id"],
        "strategy": route_contract["strategy_name"],
        "anchor_context_resolution": 1024,
        "encoder_input_resolution": 1024,
        "output_query_resolution": int(args.resolution),
        "direct_query": True,
        "reconstruction_resolution": 240825,
        "device": str(jax.devices()[0]),
        "backend": str(jax.default_backend()),
        "checkpoint": session.checkpoint.descriptor(),
        "full_field_geometry": geometry.descriptor(),
        "runtime": runtime.session.descriptor(),
        "cases": cases,
        "test_iid_or_sealed_accessed": False,
        "training_or_solver_invoked": False,
        "support_generated": False,
        "prediction_artifact_written": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
