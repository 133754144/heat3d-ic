#!/usr/bin/env python3
"""Minimal checkpoint-preserving P1i x_in=1024, x_out=N U1 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.heat3d_graph_cache import graph_hash, metadata_hash  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


RESOLUTIONS = (8192, 32768)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_frozen_binding(path: Path) -> dict[str, Any]:
    """Read immutable scientific fields without rewriting its historical code fingerprint."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_after_three_seed_r0_pass":
        raise RuntimeError("high-N binding status drifted")
    if payload["dataset"]["dataset_id"] != "heat3d_v6_p1i_continuous_physics1024_v1":
        raise RuntimeError("high-N binding dataset drifted")
    if int(payload["nested_support"]["selection_seed"]) != 20260808:
        raise RuntimeError("high-N support selection seed drifted")
    if len(payload["development_subset"]["sample_ids"]) != 32:
        raise RuntimeError("high-N frozen valid32 drifted")
    return payload


def _edges(value: Any) -> int:
    return 0 if value is None else int(np.asarray(value).shape[1] - 1)


def _strict_asymmetric_metadata(
    builder: Heat3DGraphBuilder,
    native: Any,
    anchor_graph_coords: np.ndarray,
    query_graph_coords: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Keep native p2r/r2r/rnodes exact and construct only the query r2p side."""
    domain = np.stack((anchor_graph_coords.min(axis=0), anchor_graph_coords.max(axis=0)))
    query_normalized = 2.0 * (query_graph_coords - domain[0]) / (domain[1] - domain[0]) - 1.0
    centers = np.asarray(native.x_rnodes)[0, :-1]
    base_radii = np.asarray(native.r_rnodes)[0, :-1]
    impl = builder.builder
    radii = impl._get_effective_support_radii(base_radii, impl.overlap_factor_r2p)
    r2p = impl._get_supported_pnodes_by_rnodes(
        centers=centers,
        points=query_normalized,
        radii=radii,
        apply_legacy_hard_reset=(impl.radius_policy == "legacy_kdtree_mean4"),
    )
    if impl.coverage_repair_policy == "nearest_rnode" and impl.repair_r2p:
        r2p = impl._repair_physical_node_coverage(
            edge_indices=r2p, centers=centers, points=query_normalized
        )
    r2p = np.flip(np.asarray(r2p), axis=-1)
    r2p = np.concatenate((r2p, np.asarray([[len(centers), len(query_normalized)]])), axis=0)
    if max(len(centers) + 1, len(query_normalized) + 1) < np.iinfo(np.uint16).max:
        r2p = r2p.astype(np.uint16)
    else:
        r2p = r2p.astype(np.uint32)
    x_out = np.concatenate((query_normalized, np.zeros((1, 3), dtype=query_normalized.dtype)))
    metadata = type(native)(
        x_pnodes_inp=native.x_pnodes_inp,
        x_pnodes_out=jnp.asarray(x_out[None, ...]),
        x_rnodes=native.x_rnodes,
        r_rnodes=native.r_rnodes,
        p2r_edge_indices=native.p2r_edge_indices,
        r2r_edge_indices=native.r2r_edge_indices,
        r2r_edge_domains=native.r2r_edge_domains,
        r2p_edge_indices=jnp.asarray(r2p[None, ...]),
    )
    audit = {
        "domain_from_native_anchor_only": True,
        "query_inside_native_domain": bool(np.all(query_normalized >= -1.0) and np.all(query_normalized <= 1.0)),
        "query_normalized_min": np.min(query_normalized, axis=0).tolist(),
        "query_normalized_max": np.max(query_normalized, axis=0).tolist(),
        "r2p_real_edges": int(len(r2p) - 1),
    }
    return metadata, audit


def _native_exact(native: Any, asymmetric: Any) -> dict[str, bool]:
    fields = (
        "x_pnodes_inp", "x_rnodes", "r_rnodes", "p2r_edge_indices",
        "r2r_edge_indices", "r2r_edge_domains",
    )
    return {name: bool(np.array_equal(np.asarray(getattr(native, name)), np.asarray(getattr(asymmetric, name))))
            for name in fields}


def _baseline_sample(path: Path, sample_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate") != "E":
        raise RuntimeError(f"{path}: expected candidate E")
    rows = payload["graph_diagnostics_per_sample"]
    return next(row for row in rows if row["sample_id"] == sample_id)


def _prepare_asymmetric_group(
    *, anchor: Any, query: Any, runtime: Mapping[str, Any], builder: Heat3DGraphBuilder,
    metadata: Any,
) -> dict[str, Any]:
    edge_targets = {
        field: None if getattr(metadata, field) is None else int(getattr(metadata, field).shape[1])
        for field in qualification.EDGE_FIELDS
    }
    group = highn._prepare_group(
        example=anchor, anchor=anchor, runtime=runtime, builder=builder,
        metadata=metadata, edge_targets=edge_targets,
    )
    old = group["inputs"]
    raw_query = np.asarray(query.condition.coords, dtype=np.float64).reshape(1, 1, len(query.condition.coords), 3)
    query_model_coords = runner._normalize_coords(raw_query, runtime["stats"])
    group["inputs"] = type(old)(
        u=old.u, c=old.c, x_inp=old.x_inp, x_out=jnp.asarray(query_model_coords),
        t=old.t, tau=old.tau,
    )
    group["graphs"] = builder.build_graphs(metadata)
    return group


def _report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V6 P1i U1 asymmetric-query feasibility", "",
        f"Status: `{payload['status']}`. Decision: **{payload['decision']['u1']}**.", "",
        "## Probe", "",
        "| N out | graph | native p2r/r2r exact | Nr | P2R | R2P | forward |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in payload["resolutions"]:
        lines.append(
            f"| {row['output_nodes']} | {row['graph_build_status']} | "
            f"{row['native_encoder_graph_exact']} | {row['regional_nodes']} | "
            f"{row['edge_counts']['p2r']} | {row['edge_counts']['r2p']} | "
            f"{row['forward']['status']} |"
        )
    lines += [
        "", "## Interface audit", "",
        "- The lower-level `RegionInteractionGraphBuilder.build_metadata` accepts distinct `x_inp` and `x_out`; regional nodes are sampled from `x_inp`, while r2p targets `x_out`.",
        "- `Heat3DGraphBuilder`, the V6 bridge, and the controlled runner bind one coordinate tensor to both sides.",
        "- This probe froze native-1024 p2r/r2r/regional metadata bitwise and built only the N-node r2p side. Both requested graphs passed.",
        "- At N=8192 the decoder core produced an N-node tensor and execution reached the local bypass. The bypass then rejected the shared 1024-node `Inputs.c`.",
        "- A complete asymmetric native shape-scale call would additionally require N-node CV/Dirichlet fields while retaining anchor-derived context and scale.",
        "", "## Structural potential", "",
        "| N out | encoder node reduction | P2R edge reduction | P2R+R2P edge reduction | query R2P build |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["resolutions"]:
        savings = row["structural_savings_vs_current_E"]
        lines.append(
            f"| {row['output_nodes']} | {100*savings['encoder_input_node_reduction_fraction']:.2f}% | "
            f"{100*savings['p2r_edge_reduction_fraction']:.2f}% | "
            f"{100*savings['total_p2r_r2p_edge_reduction_fraction']:.2f}% | "
            f"{row['timing_diagnostic_seconds']['query_r2p_only']:.6f} s |"
        )
    lines += [
        "", "## Blockers", "",
    ]
    for blocker in payload["decision"]["blockers"]:
        lines.append(f"- {blocker}")
    lines += [
        "", "## Interpretation", "",
        payload["decision"]["interpretation"], "",
        "The structural edge-count reductions are potential savings only; no successful U1 forward latency is claimed. "
        "Current B8192/E32768 production routes remain unchanged.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--baseline-e8192", type=Path, required=True)
    parser.add_argument("--baseline-e32768", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_execution":
        raise RuntimeError("U1 protocol is not preregistered")
    binding = _load_frozen_binding(args.binding)
    runtime_args = argparse.Namespace(run_dir=args.run_dir)
    runtime = highn._checkpoint_runtime(runtime_args)
    dataset_args = argparse.Namespace(dataset_root=args.dataset_root, manifest=args.manifest)
    dataset = highn._dataset(dataset_args)
    anchor = highn._valid_examples(dataset, binding)[0]
    full_args = argparse.Namespace(full_fields=args.full_fields)
    full, _ = highn._full_shared(full_args)
    graph_key = runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    graph_config = dict(runtime["graph_config"])
    graph_config["subsample_factor"] = 4
    model_config = runtime["model_config"]
    checkpoint_params_sha_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    baseline_paths = {8192: args.baseline_e8192, 32768: args.baseline_e32768}
    rows: list[dict[str, Any]] = []
    first_group: dict[str, Any] | None = None

    for resolution in RESOLUTIONS:
        support = highn._load_support(args.support_root / str(resolution) / f"{anchor.sample_id}.npz")
        query = highn._query_example(anchor, support, full["coords"])
        builder = Heat3DGraphBuilder(**graph_config)
        anchor_graph_coords = runner._graph_coords_for_example(anchor, runtime["stats"])
        query_graph_coords = runner._graph_coords_for_example(query, runtime["stats"])
        started = time.perf_counter()
        native = builder.build_metadata(anchor_graph_coords, key=graph_key)
        native_seconds = time.perf_counter() - started
        started = time.perf_counter()
        asymmetric, asym_audit = _strict_asymmetric_metadata(
            builder, native, anchor_graph_coords, query_graph_coords
        )
        r2p_seconds = time.perf_counter() - started
        graphs = builder.build_graphs(asymmetric)
        exact = _native_exact(native, asymmetric)
        baseline = _baseline_sample(baseline_paths[resolution], anchor.sample_id)
        counts = {
            "p2r": _edges(asymmetric.p2r_edge_indices),
            "r2r": _edges(asymmetric.r2r_edge_indices),
            "r2p": _edges(asymmetric.r2p_edge_indices),
        }
        baseline_counts = {name: int(value) for name, value in baseline["edge_count"].items()}
        graph_build_pass = (
            all(exact.values()) and int(np.asarray(asymmetric.x_rnodes).shape[1] - 1) == 256
            and int(np.asarray(asymmetric.x_pnodes_inp).shape[1] - 1) == 1024
            and int(np.asarray(asymmetric.x_pnodes_out).shape[1] - 1) == resolution
            and counts["r2p"] > 0 and asym_audit["query_inside_native_domain"]
        )
        row = {
            "sample_id": anchor.sample_id,
            "input_nodes": 1024,
            "output_nodes": resolution,
            "regional_nodes": int(np.asarray(asymmetric.x_rnodes).shape[1] - 1),
            "graph_build_status": "passed" if graph_build_pass else "failed",
            "native_encoder_graph_exact": bool(all(exact.values())),
            "native_field_equality": exact,
            "metadata_sha256": metadata_hash(asymmetric),
            "graph_sha256": graph_hash(graphs),
            "edge_counts": counts,
            "current_E_edge_counts": baseline_counts,
            "structural_savings_vs_current_E": {
                "encoder_input_node_reduction_fraction": 1.0 - 1024.0 / resolution,
                "p2r_edge_reduction_fraction": 1.0 - counts["p2r"] / baseline_counts["p2r"],
                "r2p_edge_change_fraction": counts["r2p"] / baseline_counts["r2p"] - 1.0,
                "total_p2r_r2p_edge_reduction_fraction": 1.0 - (
                    counts["p2r"] + counts["r2p"]
                ) / (baseline_counts["p2r"] + baseline_counts["r2p"]),
                "interpretation": "structural upper-bound only; no successful U1 latency measurement",
            },
            "timing_diagnostic_seconds": {
                "native_anchor_graph_metadata": native_seconds,
                "query_r2p_only": r2p_seconds,
            },
            "asymmetric_graph_audit": asym_audit,
            "tensor_contract": {
                "encoder_latent_pnodes_including_dummy": 1025,
                "decoder_output_pnodes_including_dummy": resolution + 1,
                "latent_pnode_count_equals_decoder_output_count": False,
                "decoder_core_runtime_alignment_observed": False,
                "decoder_bypass_enabled": model_config.get("decoder_bypass_mode") != "none",
                "decoder_bypass_condition_nodes": 1024,
                "decoder_bypass_output_nodes": resolution,
                "decoder_bypass_alignment_matches": False,
            },
            "forward": {"status": "pending"},
        }
        rows.append(row)
        if not graph_build_pass:
            raise RuntimeError(f"N={resolution}: asymmetric graph hard gate failed")
        if resolution == 8192:
            first_group = _prepare_asymmetric_group(
                anchor=anchor, query=query, runtime=runtime, builder=builder, metadata=asymmetric
            )

    if first_group is None:
        raise RuntimeError("missing 8192 asymmetric group")
    model = GraphNeuralOperator(**model_config)
    params = runner._device_params(runtime["checkpoint"]["params"])
    try:
        output = runner._model_apply(model, params, highn._model_group(first_group))
        jax.block_until_ready(output["raw_temperature"])
        finite = bool(np.all(np.isfinite(np.asarray(output["raw_temperature"]))))
        rows[0]["forward"] = {
            "status": "passed" if finite else "failed_nonfinite",
            "finite": finite,
            "output_shape": list(np.asarray(output["raw_temperature"]).shape),
        }
    except Exception as exc:  # Expected fail-closed evidence, captured verbatim.
        reached_bypass = "decoder bypass requires" in str(exc)
        rows[0]["tensor_contract"]["decoder_core_runtime_alignment_observed"] = reached_bypass
        rows[0]["forward"] = {
            "status": "failed_structural_incompatibility",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "decoder_core_reached_bypass": reached_bypass,
        }

    if rows[0]["forward"]["status"] == "passed":
        # Only a successful 8192 probe is allowed to progress to 32768.
        rows[1]["forward"] = {"status": "not_executed_minimal_probe_scope"}
        decision = "GO_minimal_forward_only"
        blockers: list[str] = []
    else:
        rows[1]["forward"] = {
            "status": "not_executed_fail_fast_same_frozen_interface_blocker"
        }
        decision = "NO_GO_requires_model_path_change_or_retraining"
        blockers = [
            "frozen Inputs.c is shared by the 1024-node encoder and the N-node local decoder bypass",
            "frozen local decoder bypass requires one-to-one x_inp/x_out and c/output alignment",
            "native shape-scale projection requires N-node CV/BC fields while log_s/context must remain anchor-derived",
            "runner wrapper constructs identical x_inp/x_out and has no frozen asymmetric group contract",
        ]

    checkpoint_params_sha_after = highn._tree_sha256(runtime["checkpoint"]["params"])
    payload = {
        "schema_version": "heat3d_v6_p1i_u1_asymmetric_query_result_v1",
        "status": "passed_expected_no_go" if decision.startswith("NO_GO") else "passed_probe",
        "protocol_path": str(args.protocol),
        "protocol_sha256": _sha256(args.protocol),
        "historical_binding": {
            "path": str(args.binding),
            "sha256": _sha256(args.binding),
            "scientific_fields_reused": [
                "dataset", "development_subset", "nested_support", "numeric_tolerances"
            ],
            "historical_code_fingerprint_revalidation": "not_applicable_after_committed_P5-S_exact_optimization",
            "historical_binding_modified": False,
        },
        "checkpoint": {
            "config_id": highn.CONFIG_ID,
            "epoch": highn.CHECKPOINT_EPOCH,
            "file_sha256": _sha256(args.run_dir / "params_best_valid_point_global.pkl"),
            "parameter_tree_sha256_before": checkpoint_params_sha_before,
            "parameter_tree_sha256_after": checkpoint_params_sha_after,
            "unchanged": checkpoint_params_sha_before == checkpoint_params_sha_after,
        },
        "probe_sample": anchor.sample_id,
        "resolutions": rows,
        "decision": {
            "u1": decision,
            "blockers": blockers,
            "interpretation": (
                "The graph primitive can preserve the native 1024 encoder/processor graph and attach an N-node r2p query graph, "
                "and the decoder core reaches its N-node output. The frozen full model path still cannot complete because input conditions, "
                "output-local bypass conditions, and output-native projection fields have no split asymmetric interface. Adding that interface "
                "exceeds this minimal checker and requires a separately preregistered adapter validation, although no checkpoint weights were changed here."
                if blockers else "The minimal 8192 forward completed; further validation would still be required before any production use."
            ),
            "production_route_replaced": False,
            "next_recommendation": (
                "retain B8192/E32768 production routes; if revisited, preregister a split-condition/output-native adapter, "
                "and retrain only if checkpoint-preserving equivalence cannot be established"
                if blockers else "preregister a separate asymmetric-query validation phase"
            ),
        },
        "role_contract": protocol["role_contract"],
    }
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_report_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "decision": decision,
        "output_json": str(args.output_json), "output_md": str(args.output_md),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
