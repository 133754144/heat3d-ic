#!/usr/bin/env python3
"""Two-case V8-P2B canonical-contract engineering smoke on Mac CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rigno.heat3d_v8 import (  # noqa: E402
    MassHBMReadOnlyAdapter,
    SupportProvenance,
    build_canonical_bridge,
    canonical_oracle_condition,
    geometry_fingerprint,
    oracle_global_context,
    physical_scale_features,
    select_v8_support,
)
from rigno.heat3d_v8.bridge import (  # noqa: E402
    asymmetric_query_method,
    output_local_p2r_graph,
)
from rigno.heat3d_v8.query import (  # noqa: E402
    GeometryQueryCache,
    concatenate_query_chunks,
    iter_query_chunks,
)
from rigno.models.operator import Inputs  # noqa: E402
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder  # noqa: E402


CASES = (
    "cases/Exp0_SolverAcceleration/6765682094df176c/repeat_01",
    "cases/Exp0_SolverAcceleration/2f7b8b6c5aee5687/repeat_01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--query-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "heat3d_v8" / "v8_p2b_canonical.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "v8_mass_hbm_p2b_smoke_receipt.json",
    )
    return parser.parse_args()


def _tree_l2(tree: Any) -> float:
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree))))


def _parameter_count(tree: Any) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)))


def _support_masks(view) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(view.case.coords_m)
    source = np.asarray(view.q_W_m3 != 0.0)
    interface = np.zeros(n, dtype=bool)
    if len(view.case.interface.lower_cell_index):
        interface[np.concatenate(
            [view.case.interface.lower_cell_index, view.case.interface.upper_cell_index]
        )] = True
    boundary = np.zeros(n, dtype=bool)
    boundary[view.case.boundary.cell_index] = True
    return source, interface, boundary


def _model_config(raw: dict[str, Any], condition_names: tuple[str, ...], context_names: tuple[str, ...]) -> dict[str, Any]:
    config = dict(raw)
    architecture = config.pop("architecture")
    policy = config.pop("decoder_bypass_feature_policy")
    schema = config.pop("global_context_schema")
    if architecture != "RIGNO" or policy != "all_v8_condition_features" or schema != "V8_GLOBAL_CONTEXT_FEATURES":
        raise ValueError("V8 canonical model policy drift")
    config.update(
        decoder_bypass_feature_indices=tuple(range(len(condition_names))),
        decoder_bypass_feature_names=condition_names,
        decoder_bypass_local_feature_names=condition_names,
        decoder_bypass_num_features=len(condition_names),
        global_context_feature_dim=len(context_names),
        global_context_feature_names=context_names,
    )
    return config


def _query_inputs(condition: np.ndarray, normalized_coords: np.ndarray) -> Inputs:
    c = jnp.asarray(condition[None, None, :, :], dtype=jnp.float32)
    coords = jnp.asarray(normalized_coords[None, None, :, :], dtype=jnp.float32)
    return Inputs(
        u=jnp.zeros((1, 1, len(condition), 1), dtype=jnp.float32),
        c=c,
        x_inp=coords,
        x_out=coords,
        t=None,
        tau=None,
    )


def _case_record(adapter, case_directory: str, graph_config: dict[str, Any]):
    view = adapter.load_oracle_view(case_directory)
    fingerprint = geometry_fingerprint(adapter.dataset_root / case_directory)
    condition, condition_names, condition_provenance = canonical_oracle_condition(view)
    bridge_full = build_canonical_bridge(view)
    context, context_values, context_provenance = oracle_global_context(view)
    source, interface, boundary = _support_masks(view)
    support = select_v8_support(
        source_mask=source,
        interface_mask=interface,
        boundary_sink_mask=boundary,
        control_volume_m3=view.case.control_volume_m3,
        sample_id=view.case.sample_id,
        support_provenance=SupportProvenance.ORACLE_SUPPORT,
        count=1024,
        seed=0,
    )
    bridge = build_canonical_bridge(view, support.indices)
    support_coords = np.asarray(view.case.coords_m[support.indices], dtype=np.float32)
    builder = RegionInteractionGraphBuilder(**graph_config)
    metadata = builder.build_metadata(
        x_inp=support_coords,
        x_out=support_coords,
        domain=bridge.coordinate_domain_m,
        key=jax.random.PRNGKey(0),
    )
    graphs = builder.build_graphs(metadata)
    reference = float(view.metadata["reference_temperature_K"])
    target = jnp.asarray(
        ((view.target_temperature_K[support.indices] - reference) / 100.0)[None, None, :, None],
        dtype=jnp.float32,
    )
    scale = physical_scale_features(view)
    integrated_power = float(np.sum(view.q_W_m3 * view.case.control_volume_m3))
    power_error = float(view.metadata["power_relative_error"])
    adapter_pass = bool(
        np.all(np.isfinite(condition))
        and np.all(view.case.control_volume_m3 > 0.0)
        and power_error <= 1.0e-12
        and np.all(np.isfinite(view.target_temperature_K))
    )
    return {
        "view": view,
        "fingerprint": fingerprint,
        "condition": condition,
        "condition_names": condition_names,
        "condition_provenance": condition_provenance,
        "bridge_full": bridge_full,
        "bridge": bridge,
        "context": jnp.asarray(context[None, :]),
        "context_values": context_values,
        "context_provenance": context_provenance,
        "support": support,
        "support_coords": support_coords,
        "builder": builder,
        "metadata": metadata,
        "graphs": graphs,
        "target": target,
        "scale": scale,
        "adapter_pass": adapter_pass,
        "integrated_power_W": integrated_power,
        "power_relative_error": power_error,
    }


def _full_query(model, params, record, chunk_size: int) -> dict[str, Any]:
    view = record["view"]
    support_coords = record["support_coords"]
    support_inputs = record["bridge"].inputs
    full_condition = record["condition"]
    full_normalized = record["bridge_full"].normalized_coords
    builder = record["builder"]
    domain = record["bridge"].coordinate_domain_m
    cache = GeometryQueryCache()
    builder_id = "v8-p2b-sparse-kdtree-query-v1"
    chunks = iter_query_chunks(
        np.asarray(view.case.coords_m, dtype=np.float32),
        chunk_size=chunk_size,
        support_coords=support_coords,
        builder_id=builder_id,
    )
    predictions = []
    build_seconds = 0.0
    forward_seconds = 0.0
    first_miss = False
    repeat_hit = False
    for number, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        metadata, hit, _ = cache.get_or_build(
            support_coords=support_coords,
            query_coords=chunk.coords,
            builder_id=builder_id,
            build=lambda chunk=chunk: builder.build_metadata(
                x_inp=support_coords,
                x_out=chunk.coords,
                domain=domain,
                key=jax.random.PRNGKey(0),
            ),
        )
        build_seconds += time.perf_counter() - t0
        if number == 0:
            first_miss = not hit
            repeated, repeated_hit, _ = cache.get_or_build(
                support_coords=support_coords,
                query_coords=chunk.coords,
                builder_id=builder_id,
                build=lambda: (_ for _ in ()).throw(AssertionError("cache rebuild")),
            )
            repeat_hit = bool(repeated_hit and repeated is metadata)
        graphs = builder.build_graphs(metadata)
        query_inputs = _query_inputs(
            full_condition[chunk.start:chunk.stop],
            full_normalized[chunk.start:chunk.stop],
        )
        local = output_local_p2r_graph(builder, metadata)
        t0 = time.perf_counter()
        prediction = model.apply(
            {"params": params},
            inputs_in=support_inputs,
            inputs_out=query_inputs,
            graphs=graphs,
            output_local_p2r=local,
            reuse_input_local_latents=False,
            global_context=record["context"],
            method=asymmetric_query_method,
        )
        jax.block_until_ready(prediction)
        forward_seconds += time.perf_counter() - t0
        predictions.append(np.asarray(prediction[0, 0, :, 0]))
    full = concatenate_query_chunks(chunks, predictions, expected_count=len(view.case.coords_m))
    ordering = concatenate_query_chunks(
        chunks,
        [np.arange(chunk.start, chunk.stop) for chunk in chunks],
        expected_count=len(view.case.coords_m),
    )
    status = bool(
        np.all(np.isfinite(full))
        and np.array_equal(ordering, np.arange(len(full)))
        and first_miss
        and repeat_hit
    )
    return {
        "status": "PASS" if status else "FAIL",
        "point_count": int(len(full)),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "backend": "sparse_kdtree_v1",
        "dense_pairwise_used": False,
        "output_finite": bool(np.all(np.isfinite(full))),
        "ordering_exact": bool(np.array_equal(ordering, np.arange(len(full)))),
        "cache_first_miss": first_miss,
        "cache_repeat_hit": repeat_hit,
        "metadata_build_seconds": build_seconds,
        "network_forward_seconds": forward_seconds,
    }


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if jax.default_backend() != "cpu":
        raise RuntimeError("V8-P2B smoke is Mac CPU only")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "engineering_smoke_only_not_formal_training":
        raise ValueError("canonical config status drift")
    adapter = MassHBMReadOnlyAdapter(args.dataset_root)
    records = [_case_record(adapter, case, dict(config["graph"])) for case in CASES]
    if records[0]["condition_names"] != records[1]["condition_names"]:
        raise ValueError("two-case condition schema mismatch")
    if records[0]["fingerprint"].sha256 == records[1]["fingerprint"].sha256:
        raise ValueError("two smoke cases must have distinct geometry fingerprints")
    scales_distinct = records[0]["scale"] != records[1]["scale"]
    if not scales_distinct:
        raise ValueError("physical scale features failed to distinguish cases")
    from rigno.heat3d_v8.context import V8_GLOBAL_CONTEXT_FEATURES

    model_config = _model_config(
        config["model"], records[0]["condition_names"], V8_GLOBAL_CONTEXT_FEATURES
    )
    model = RIGNO(**model_config)
    params = model.init(
        jax.random.PRNGKey(7),
        inputs=records[0]["bridge"].inputs,
        graphs=records[0]["graphs"],
        global_context=records[0]["context"],
    )["params"]

    def predictions(current_params):
        return [
            model.apply(
                {"params": current_params},
                inputs=record["bridge"].inputs,
                graphs=record["graphs"],
                global_context=record["context"],
            )
            for record in records
        ]

    def joint_loss(current_params):
        values = predictions(current_params)
        sample_losses = [jnp.mean(jnp.square(prediction - record["target"])) for prediction, record in zip(values, records)]
        return jnp.mean(jnp.stack(sample_losses)), jnp.stack(sample_losses)

    initial_joint, initial_samples = joint_loss(params)
    jax.block_until_ready((initial_joint, initial_samples))
    optimizer = optax.adam(args.learning_rate)
    state = optimizer.init(params)
    gradient_norms = []
    train_started = time.perf_counter()
    for _ in range(args.steps):
        (loss_value, _), gradients = jax.value_and_grad(joint_loss, has_aux=True)(params)
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
        jax.block_until_ready(params)
        gradient_norms.append(_tree_l2(gradients))
    final_joint, final_samples = joint_loss(params)
    jax.block_until_ready((final_joint, final_samples))
    training_seconds = time.perf_counter() - train_started

    equivalence_rows = []
    for record in records:
        standard = model.apply(
            {"params": params},
            inputs=record["bridge"].inputs,
            graphs=record["graphs"],
            global_context=record["context"],
        )
        asymmetric = model.apply(
            {"params": params},
            inputs_in=record["bridge"].inputs,
            inputs_out=record["bridge"].inputs,
            graphs=record["graphs"],
            output_local_p2r=record["graphs"].p2r,
            reuse_input_local_latents=True,
            global_context=record["context"],
            method=asymmetric_query_method,
        )
        jax.block_until_ready((standard, asymmetric))
        difference = float(np.max(np.abs(np.asarray(standard) - np.asarray(asymmetric))))
        equivalence_rows.append({
            "sample_id": record["view"].case.sample_id,
            "max_abs_difference": difference,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "pass": bool(np.allclose(np.asarray(standard), np.asarray(asymmetric), atol=1.0e-6, rtol=1.0e-6)),
        })

    query_rows = [_full_query(model, params, record, args.query_chunk_size) for record in records]
    case_rows = []
    for index, record in enumerate(records):
        view = record["view"]
        explicit_columns = [
            i for i, name in enumerate(record["condition_names"])
            if name.startswith("interface_G_")
        ]
        explicit_nonzero = bool(np.any(record["condition"][:, explicit_columns] != 0.0))
        case_rows.append({
            "case_directory": CASES[index],
            "sample_id": view.case.sample_id,
            "shape_zyx": list(view.case.shape_zyx),
            "cell_count": int(len(view.case.coords_m)),
            "geometry_fingerprint": record["fingerprint"].sha256,
            "geometry_fingerprint_claim": record["fingerprint"].independence_claim,
            "physical_extents_m": {
                "Lx": float(view.case.geometry_metadata["x_extent_mm"]) * 1.0e-3,
                "Ly": float(view.case.geometry_metadata["y_extent_mm"]) * 1.0e-3,
                "Lz": float(view.case.geometry_metadata["z_extent_mm"]) * 1.0e-3,
            },
            "scale_features": record["scale"],
            "adapter_status": "PASS" if record["adapter_pass"] else "FAIL",
            "power_relative_error": record["power_relative_error"],
            "interface_mode": view.case.interface.mode,
            "interface_double_count_guard": view.case.interface.double_count_guard,
            "explicit_interface_face_count": int(len(view.case.interface.tbr_m2K_W)),
            "explicit_interface_condition_nonzero": explicit_nonzero,
            "support_provenance": record["support"].provenance.value,
            "support_class_counts": record["support"].class_counts,
            "initial_sample_loss": float(initial_samples[index]),
            "final_sample_loss": float(final_samples[index]),
            "loss_ratio": float(final_samples[index] / initial_samples[index]),
            "full_field_query": query_rows[index],
        })

    explicit_rows = [row for row in case_rows if row["interface_mode"] == "explicit_series_z"]
    explicit_path_pass = bool(
        len(explicit_rows) == 1
        and explicit_rows[0]["explicit_interface_face_count"] > 0
        and explicit_rows[0]["explicit_interface_condition_nonzero"]
        and "PASS_EXPLICIT" in explicit_rows[0]["interface_double_count_guard"]
    )
    bridge_pass = bool(
        all(np.all(np.asarray(record["bridge"].inputs.u) == 0.0) for record in records)
        and all(record["bridge"].inputs.c.shape[-1] == len(record["condition_names"]) for record in records)
    )
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    receipt = {
        "schema_version": "heat3d_v8_p2b_two_sample_smoke_v1",
        "classification": "ENGINEERING_SMOKE_NOT_ACCURACY_NOT_FORMAL_TRAINING",
        "branch_baseline": "8c3e06b5ad647ecc5572cedc0d013fc61ae2dc8c",
        "execution": {
            "backend": jax.default_backend(),
            "devbox_used": False,
            "gpu_used": False,
            "formal_training": False,
            "checkpoint_written": False,
        },
        "contract": {
            "config": str(args.config.relative_to(ROOT)),
            "u": "zero_delta_field",
            "c": "all_27_v8_physics_features",
            "condition_feature_names": list(records[0]["condition_names"]),
            "physical_scale_features": list(records[0]["scale"]),
            "scale_features_distinct": scales_distinct,
            "physical_scale_ambiguity_removed": scales_distinct,
            "canonical_bridge_status": "PASS" if bridge_pass else "FAIL",
            "query_contract": "1024 sparse global conditioning support plus query-local physical coefficients",
            "support_is_not_total_physics_read_count": True,
        },
        "model": {
            "node_latent_size": model_config["node_latent_size"],
            "edge_latent_size": model_config["edge_latent_size"],
            "processor_steps": model_config["processor_steps"],
            "mlp_hidden_layers": model_config["mlp_hidden_layers"],
            "conditioned_normalization": model_config["conditioned_normalization"],
            "decoder_bypass": "all V8 condition features",
            "global_context_feature_names": list(V8_GLOBAL_CONTEXT_FEATURES),
            "parameter_count": _parameter_count(params),
            "v7_full_capacity_migration": "PASS" if (
                model_config["node_latent_size"] == 96
                and model_config["edge_latent_size"] == 96
                and model_config["processor_steps"] == 6
                and model_config["mlp_hidden_layers"] == 2
                and not model_config["conditioned_normalization"]
            ) else "FAIL",
        },
        "query_equivalence": {
            "status": "PASS" if all(row["pass"] for row in equivalence_rows) else "FAIL",
            "support_equals_query": True,
            "rows": equivalence_rows,
        },
        "support_provenance": {
            "status": "PASS" if all(record["support"].provenance == SupportProvenance.ORACLE_SUPPORT for record in records) else "FAIL",
            "used": "ORACLE_SUPPORT",
            "reason": "heat-source candidate topology is final-q nonzero mask",
            "deployable_claim": False,
            "target_independent_claim": False,
            "future_track_b_required": "PRE_SOLVE_SUPPORT",
        },
        "explicit_tbr_path": {
            "status": "PASS" if explicit_path_pass else "FAIL",
            "double_count_guard_required": True,
        },
        "optimization": {
            "status": "PASS" if (
                np.isfinite(float(final_joint))
                and float(final_joint) < float(initial_joint)
                and np.all(np.isfinite(gradient_norms))
            ) else "FAIL",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "initial_joint_loss": float(initial_joint),
            "final_joint_loss": float(final_joint),
            "joint_loss_ratio": float(final_joint / initial_joint),
            "first_gradient_norm": gradient_norms[0],
            "last_gradient_norm": gradient_norms[-1],
            "wall_seconds": training_seconds,
            "accuracy_claim": False,
        },
        "cases": case_rows,
        "geometry_fingerprint": {
            "formal_split_frozen": False,
            "claim": "LABEL_ARRAY_INDEPENDENT_ONLY",
            "strict_target_independence": False,
            "reason": "solver_geometry mesh provenance is unconfirmed",
        },
        "process": {
            "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "timing_classification": "ENGINEERING_DIAGNOSTIC_NOT_PUBLICATION_LATENCY",
        },
    }
    hard_pass = bool(
        all(row["adapter_status"] == "PASS" for row in case_rows)
        and receipt["contract"]["canonical_bridge_status"] == "PASS"
        and receipt["model"]["v7_full_capacity_migration"] == "PASS"
        and receipt["query_equivalence"]["status"] == "PASS"
        and receipt["explicit_tbr_path"]["status"] == "PASS"
        and receipt["optimization"]["status"] == "PASS"
        and receipt["support_provenance"]["status"] == "PASS"
        and all(row["full_field_query"]["status"] == "PASS" for row in case_rows)
    )
    receipt["status"] = "PASS" if hard_pass else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": receipt["status"],
        "loss_ratio": receipt["optimization"]["joint_loss_ratio"],
        "equivalence": receipt["query_equivalence"]["status"],
        "explicit_tbr": receipt["explicit_tbr_path"]["status"],
    }))
    return 0 if hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
