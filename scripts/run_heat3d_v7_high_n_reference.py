#!/usr/bin/env python3
"""V7 stable-runtime anchor-derived high-N reference inference.

Only pre-existing support artifacts are accepted for resolutions above the
native 1024 anchor.  This entrypoint never imports a script module, generates
support/data, invokes a solver, writes a prediction artifact, or reads a
non-valid_iid label.  It prints a JSON provenance summary to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

from rigno.heat3d_runtime import (
    FullFieldGeometry,
    HighNRuntime,
    RuntimeSession,
    bind_registered_route,
)
from rigno.heat3d_v6_dataset import (
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
    Heat3DV6DualRobinDataset,
)
from rigno.heat3d_v6_p1i_anchor_query import array_sha256


ALLOWED_RESOLUTIONS = (1024, 16384, 32768)


def _edge_targets(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "graph_cache" in payload:
        values = payload["graph_cache"]["edge_targets"]
    else:
        values = payload["padding"]["actual_padding_envelope"]["query"]
    if not isinstance(values, dict) or not values:
        raise ValueError(f"padding envelope is not a non-empty mapping: {path}")
    return {str(key): None if value is None else int(value) for key, value in values.items()}


def _route_contract(args: argparse.Namespace) -> dict[str, object] | None:
    resolution = int(args.resolution)
    if resolution == 1024:
        if any(
            value is not None
            for value in (
                args.route_id,
                args.strategy,
                args.anchor_context_resolution,
                args.encoder_input_resolution,
                args.output_query_resolution,
                args.reconstruction_resolution,
                args.padding_envelope,
            )
        ):
            raise ValueError("native 1024 has no high-resolution route binding arguments")
        return None
    required = {
        "eu-contract": args.eu_contract,
        "route-id": args.route_id,
        "strategy": args.strategy,
        "anchor-context-resolution": args.anchor_context_resolution,
        "encoder-input-resolution": args.encoder_input_resolution,
        "output-query-resolution": args.output_query_resolution,
        "reconstruction-resolution": args.reconstruction_resolution,
        "padding-envelope": args.padding_envelope,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"production route binding requires explicit arguments: {missing}")
    if int(args.output_query_resolution) != resolution:
        raise ValueError("--resolution must equal --output-query-resolution")
    return bind_registered_route(
        contract_path=args.eu_contract,
        route_id=str(args.route_id),
        requested_strategy=str(args.strategy),
        anchor_context_resolution=int(args.anchor_context_resolution),
        encoder_input_resolution=int(args.encoder_input_resolution),
        output_query_resolution=int(args.output_query_resolution),
        reconstruction_resolution=int(args.reconstruction_resolution),
        fixed_edge_targets=_edge_targets(args.padding_envelope),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--eu-contract", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--anchor-predictions", type=Path)
    parser.add_argument("--resolution", type=int, choices=ALLOWED_RESOLUTIONS, default=1024)
    parser.add_argument("--route-id")
    parser.add_argument("--strategy")
    parser.add_argument("--anchor-context-resolution", type=int)
    parser.add_argument("--encoder-input-resolution", type=int)
    parser.add_argument("--output-query-resolution", type=int)
    parser.add_argument("--reconstruction-resolution", type=int)
    parser.add_argument("--padding-envelope", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--split", choices=("valid_iid",), default="valid_iid")
    parser.add_argument("--graph-cache-dir", type=Path)
    parser.add_argument("--write-graph-cache", action="store_true")
    return parser


def _support_path(root: Path | None, resolution: int, sample_id: str) -> Path:
    if root is None:
        raise FileNotFoundError(
            f"resolution={resolution} requires --support-root containing pre-existing support artifacts"
        )
    return root / str(resolution) / f"{sample_id}.npz"


def _anchor_scales(path: Path | None, sample_ids: list[str]) -> dict[str, float]:
    if path is None:
        raise FileNotFoundError(
            "resolutions above 1024 require the pre-existing frozen 1024 anchor prediction artifact"
        )
    with np.load(path, allow_pickle=False) as payload:
        observed_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
        values = np.asarray(payload["predicted_scales"], dtype=np.float64).reshape(-1)
    if observed_ids != sample_ids or len(values) != len(sample_ids):
        raise ValueError("frozen 1024 anchor prediction sample order drifted")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("frozen 1024 anchor scales are not finite and positive")
    return dict(zip(observed_ids, map(float, values), strict=True))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split != "valid_iid":
        raise ValueError("V7 high-N reference inference is restricted to valid_iid")
    if args.write_graph_cache and args.graph_cache_dir is None:
        raise ValueError("--write-graph-cache requires --graph-cache-dir")
    dataset = Heat3DV6DualRobinDataset(
        args.dataset_root,
        args.manifest,
        include_roles={"valid_iid"},
    )
    if dataset.manifest["dataset_id"] != CONTINUOUS_PHYSICS_V6_DATASET_ID:
        raise ValueError("high-N reference requires frozen V6/P1i continuous_physics1024_v1")
    examples = list(dataset.samples)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        examples = examples[: int(args.max_samples)]
    if not examples:
        raise ValueError("valid_iid fixture contains no examples")

    route_contract = _route_contract(args)
    session = RuntimeSession.from_paths(
        args.checkpoint,
        args.run_config,
        execution_role="production_inference",
        route_contract=route_contract,
    )
    geometry = FullFieldGeometry.load(args.full_fields)
    runtime = HighNRuntime.from_session(session, geometry)
    anchor_scales = (
        {}
        if args.resolution == 1024
        else _anchor_scales(args.anchor_predictions, [str(row.sample_id) for row in examples])
    )
    case_summaries = []
    for anchor in examples:
        support_path = (
            None
            if args.resolution == 1024
            else _support_path(args.support_root, args.resolution, str(anchor.sample_id))
        )
        fixed_targets = None if route_contract is None else route_contract["fixed_edge_targets"]
        if fixed_targets is not None and not isinstance(fixed_targets, dict):
            raise ValueError("production E route requires explicit fixed_edge_targets mapping")
        case = runtime.build_case(
            anchor,
            args.resolution,
            support_path=support_path,
            edge_targets=fixed_targets,
            cache_dir=args.graph_cache_dir,
            write_cache=bool(args.write_graph_cache),
        )
        output = runtime.session.apply(case.group)
        jax.block_until_ready(output["raw_temperature"])
        raw = np.asarray(output["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
        query_scale = float(np.asarray(output["s_hat"], dtype=np.float64).reshape(-1)[0])
        scale = query_scale if args.resolution == 1024 else anchor_scales[str(anchor.sample_id)]
        prediction = (
            raw
            if args.resolution == 1024
            else runtime.apply_anchor_scale(
                raw, scale, np.asarray(case.example.operator_point_weights)
            )
        )
        case_summaries.append(
            {
                "sample_id": str(anchor.sample_id),
                "route_id": None if route_contract is None else route_contract["route_id"],
                "strategy": None if route_contract is None else route_contract["strategy_name"],
                "resolution": int(args.resolution),
                "support_indices_sha256": case.group["support_indices_sha256"],
                "graph_tensors_sha256": case.group["graph_tensors_sha256"],
                "raw_temperature_shape": list(raw.shape),
                "predicted_scale": float(scale),
                "query_scale": query_scale,
                "prediction_shape": list(prediction.shape),
                "prediction_sha256": array_sha256(prediction),
                "graph": case.graph.audit,
            }
        )
    return {
        "status": "inference_complete",
        "experiment_role": "high_n_reference_inference",
        "dataset_role": "valid_iid",
        "dataset_id": dataset.manifest["dataset_id"],
        "resolution": int(args.resolution),
        "route_id": None if route_contract is None else route_contract["route_id"],
        "strategy": None if route_contract is None else route_contract["strategy_name"],
        "sample_count": len(case_summaries),
        "device": str(jax.devices()[0]),
        "backend": str(jax.default_backend()),
        "checkpoint": session.checkpoint.descriptor(),
        "full_field_geometry": geometry.descriptor(),
        "runtime": runtime.session.descriptor(),
        "cases": case_summaries,
        "test_iid_or_sealed_accessed": False,
        "training_or_solver_invoked": False,
        "support_generated": False,
        "prediction_artifact_written": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
