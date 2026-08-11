#!/usr/bin/env python3
"""Checkpoint-preserving P1i asymmetric-query split-adapter probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import h5py
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as qualification  # noqa: E402
import probe_heat3d_v6_p1i_u1_asymmetric_query as prior_u1  # noqa: E402
import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402
import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
import run_heat3d_v6_p1i_graph_scale_candidate as candidate  # noqa: E402
from rigno.graphBuilder_Heat3D import Heat3DGraphBuilder  # noqa: E402
from rigno.models.rigno import RIGNO as GraphNeuralOperator  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)), "p95_seconds": float(np.quantile(array, 0.95)),
    }


def _tree_diff(left: Any, right: Any) -> dict[str, float | bool]:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return {"array_equal": False, "max_abs": math.inf, "rmse": math.inf}
    equal = True; maximum = 0.0; sum_sq = 0.0; count = 0
    for a, b in zip(left_leaves, right_leaves, strict=True):
        x = np.asarray(a); y = np.asarray(b)
        if x.shape != y.shape:
            return {"array_equal": False, "max_abs": math.inf, "rmse": math.inf}
        equal = equal and np.array_equal(x, y)
        delta = x.astype(np.float64) - y.astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(delta))) if delta.size else 0.0)
        sum_sq += float(np.sum(np.square(delta))); count += int(delta.size)
    return {"array_equal": bool(equal), "max_abs": maximum, "rmse": math.sqrt(sum_sq / max(count, 1))}


def _features(module: Any, inputs: Any) -> tuple[Any, Any]:
    batch_size = inputs.u.shape[0]
    n = inputs.x_inp.shape[2]
    if module.concatenate_t:
        t = jnp.asarray(inputs.t, dtype=jnp.float32)
        if t.ndim == 4:
            t = t[:, :, 0, 0]
        if t.size == 1:
            t = jnp.tile(t.reshape(1, 1), reps=(batch_size, 1))
    if module.concatenate_tau:
        tau = jnp.asarray(inputs.tau, dtype=jnp.float32)
        if tau.ndim == 4:
            tau = tau[:, :, 0, 0]
        if tau.size == 1:
            tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    else:
        tau = None
    u = inputs.u if inputs.c is None else jnp.concatenate([inputs.u, inputs.c], axis=-1)
    result = jnp.moveaxis(u, source=(0, 1, 2, 3), destination=(0, 3, 1, 2)).squeeze(axis=3)
    forced = []
    if module.concatenate_t:
        forced.append(jnp.tile(t[:, None, :], reps=(1, n, 1)))
    if module.concatenate_tau:
        forced.append(jnp.tile(tau[:, None, :], reps=(1, n, 1)))
    return jnp.concatenate([result, *forced], axis=-1), tau


def _native_finish(
    module: Any, *, psi: Any, processed: Any, pre_film: Any,
    control_volumes: Any, log_s_phys: Any, reference_temperature: Any,
    dirichlet_mask: Any, prescribed_temperature: Any, global_context: Any,
    qk_region_features: Any, scale_context: Any,
    scale_region_source_weights: Any, scale_region_volume_weights: Any,
) -> dict[str, Any]:
    volumes = module._prediction_field(control_volumes, psi, "control_volumes")
    volume_sum = jnp.sum(volumes, axis=2, keepdims=True)
    dirichlet = module._prediction_field(dirichlet_mask, psi, "dirichlet_mask") > 0.5
    psi_free = jnp.where(dirichlet, jnp.zeros_like(psi), psi)
    psi_rms = jnp.sqrt(
        jnp.sum(jnp.square(psi_free) * volumes, axis=2, keepdims=True)
        / jnp.maximum(volume_sum, module.shape_scale_epsilon)
    )
    phi_hat = psi_free / jnp.maximum(psi_rms, module.shape_scale_epsilon)
    context = module._global_context_array(global_context, batch_size=psi.shape[0], dtype=psi.dtype)
    if module.scale_head_mode == "physics_plus_pooled_latent":
        pooled = module._pooled_scale_features(
            processed, pre_film, qk_region_features=qk_region_features,
            global_context=global_context,
            scale_region_source_weights=scale_region_source_weights,
            scale_region_volume_weights=scale_region_volume_weights,
        )
        scale_context_array = module._scale_context_array(
            scale_context, batch_size=psi.shape[0], dtype=psi.dtype,
        )
        scale_features = jnp.concatenate([context, scale_context_array, pooled], axis=-1)
    else:
        pooled = jnp.zeros((psi.shape[0], 0), dtype=psi.dtype)
        scale_context_array = module._scale_context_array(
            scale_context, batch_size=psi.shape[0], dtype=psi.dtype,
        )
        scale_features = jnp.concatenate([context, scale_context_array], axis=-1)
    hidden = jax.nn.gelu(module.global_scale_hidden(scale_features))
    for layer in module.global_scale_extra_hidden:
        hidden = jax.nn.gelu(layer(hidden))
    residual = module.global_scale_output(hidden)[:, :, None, None]
    log_s_hat = module._sample_scalar(log_s_phys, psi, "log_s_phys") + residual
    s_hat = jnp.exp(log_s_hat)
    delta = s_hat * phi_hat
    reference = module._prediction_field(reference_temperature, psi, "reference_temperature")
    prescribed = module._prediction_field(prescribed_temperature, psi, "prescribed_temperature")
    raw_unprojected = reference + delta
    raw = jnp.where(dirichlet, prescribed, raw_unprojected)
    return {
        "psi": psi, "phi_hat": phi_hat, "s_hat": s_hat,
        "processed_rnodes": processed, "processed_rnodes_pre_film": pre_film,
        "raw_temperature": raw, "deltaT_hat": raw - reference,
        "pooled_rnodes": pooled,
    }


def _trace_method(
    module: Any, inputs_in: Any, inputs_out: Any, graphs: Any, output_local_p2r: Any,
    *, split: bool, control_volumes: Any, log_s_phys: Any,
    reference_temperature: Any, dirichlet_mask: Any, prescribed_temperature: Any,
    global_context: Any, qk_region_features: Any, scale_context: Any,
    scale_region_source_weights: Any, scale_region_volume_weights: Any,
) -> dict[str, Any]:
    features_in, tau = _features(module, inputs_in)
    features_in = jnp.concatenate([
        features_in,
        jnp.zeros((features_in.shape[0], 1, features_in.shape[-1]), dtype=features_in.dtype),
    ], axis=1)
    latent_r, latent_in = module.encoder(graphs.p2r, features_in, tau, key=None)
    updated = module.processor(graphs.r2r, latent_r, tau, key=None)
    pre_film = updated[:, :-1]
    updated = module._apply_global_film(updated, global_context)
    processed = updated[:, :-1]
    decoder_r = module._apply_shape_attention(
        updated, qk_region_features=qk_region_features, global_context=global_context,
    )
    if split:
        features_out, tau_out = _features(module, inputs_out)
        features_out = jnp.concatenate([
            features_out,
            jnp.zeros((features_out.shape[0], 1, features_out.shape[-1]), dtype=features_out.dtype),
        ], axis=1)
        _, latent_out = module.encoder(output_local_p2r, features_out, tau_out, key=None)
    else:
        latent_out = latent_in
    decoded = module.decoder(graphs.r2p, decoder_r, latent_out, tau, key=None)
    decoded = decoded[:, :-1, :]
    pre_bypass = module._prepare_features(decoded)
    post_bypass = module._apply_decoder_bypass(pre_bypass, inputs_out if split else inputs_in)
    result = _native_finish(
        module, psi=post_bypass, processed=processed, pre_film=pre_film,
        control_volumes=control_volumes, log_s_phys=log_s_phys,
        reference_temperature=reference_temperature, dirichlet_mask=dirichlet_mask,
        prescribed_temperature=prescribed_temperature, global_context=global_context,
        qk_region_features=qk_region_features, scale_context=scale_context,
        scale_region_source_weights=scale_region_source_weights,
        scale_region_volume_weights=scale_region_volume_weights,
    )
    result.update({
        "encoder_input_pnodes": latent_in[:, :-1],
        "encoder_output_local_pnodes": latent_out[:, :-1],
        "decoder_pre_bypass": pre_bypass,
        "decoder_post_bypass": post_bypass,
    })
    return result


def _pre_bypass_method(
    module: Any, inputs: Any, graphs: Any, *, global_context: Any,
    qk_region_features: Any,
) -> dict[str, Any]:
    features, tau = _features(module, inputs)
    output, processed, pre_film = module._encode_process_decode(
        graphs=graphs, pnode_features=features, tau=tau,
        global_context=global_context, qk_region_features=qk_region_features, key=None,
    )
    return {
        "decoder_pre_bypass": module._prepare_features(output),
        "processed_rnodes": processed, "processed_rnodes_pre_film": pre_film,
    }


def _dummy_local_p2r(builder: Heat3DGraphBuilder, metadata: Any) -> Any:
    n_p = int(np.asarray(metadata.x_pnodes_out).shape[1] - 1)
    n_r = int(np.asarray(metadata.x_rnodes).shape[1] - 1)
    dtype = np.uint16 if max(n_p + 1, n_r + 1) < np.iinfo(np.uint16).max else np.uint32
    dummy = jnp.asarray(np.asarray([[[n_p, n_r]]], dtype=dtype))
    local = type(metadata)(
        x_pnodes_inp=metadata.x_pnodes_out,
        x_pnodes_out=metadata.x_pnodes_out,
        x_rnodes=metadata.x_rnodes,
        r_rnodes=metadata.r_rnodes,
        p2r_edge_indices=dummy,
        r2r_edge_indices=metadata.r2r_edge_indices,
        r2r_edge_domains=metadata.r2r_edge_domains,
        r2p_edge_indices=None,
    )
    return builder.build_graphs(local).p2r


def _edge_targets(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text())
    if "graph_cache" in payload:
        return {key: (None if value is None else int(value)) for key, value in payload["graph_cache"]["edge_targets"].items()}
    targets: dict[str, int | None] = {}
    for field in qualification.EDGE_FIELDS:
        values = []
        for row in payload["graph_metadata_artifacts"]:
            with np.load(row["path"], allow_pickle=False) as archive:
                none_fields = {str(value) for value in np.asarray(archive["__none_fields_utf8"]).tolist()}
                if field not in none_fields:
                    values.append(int(np.asarray(archive[field]).shape[1]))
        targets[field] = max(values) if values else None
    return targets


def _combined_targets(native: Mapping[str, int | None], query: Mapping[str, int | None]) -> dict[str, int | None]:
    return {
        "p2r_edge_indices": native["p2r_edge_indices"],
        "r2r_edge_indices": native["r2r_edge_indices"],
        "r2r_edge_domains": native["r2r_edge_domains"],
        "r2p_edge_indices": query["r2p_edge_indices"],
    }


def _model_kwargs(anchor_group: Mapping[str, Any], output_group: Mapping[str, Any]) -> dict[str, Any]:
    anchor_physics = anchor_group["native_physics"]
    output_physics = output_group["native_physics"]
    return {
        "control_volumes": output_physics["control_volumes"],
        "log_s_phys": anchor_physics["log_s_phys"],
        "reference_temperature": output_physics["reference_temperature"],
        "dirichlet_mask": output_physics["dirichlet_mask"],
        "prescribed_temperature": output_physics["prescribed_temperature"],
        "global_context": anchor_group.get("global_context"),
        "qk_region_features": anchor_group.get("qk_region_features"),
        "scale_context": anchor_group.get("scale_context"),
        "scale_region_source_weights": anchor_group.get("scale_region_source_weights"),
        "scale_region_volume_weights": anchor_group.get("scale_region_volume_weights"),
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "binding", "artifact_root", "dataset_root", "manifest",
        "full_fields", "run_dir", "native_padding_result", "query_padding_result",
        "baseline_result", "identity_result", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path)
    parser.add_argument("--resolution", type=int, choices=[1024, 8192, 32768], required=True)
    parser.add_argument("--sample-count", type=int, choices=[1, 32], default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    backend = jax.devices()[0].platform
    if args.resolution == 1024 and backend != "cpu":
        raise RuntimeError("U1 bitwise identity gate requires the deterministic CPU backend")
    if args.resolution != 1024 and backend != "gpu":
        raise RuntimeError("U1 high-N split adapter requires the frozen GPU backend")
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_before_execution":
        raise RuntimeError("U1 split protocol is not preregistered")
    if args.resolution != 1024:
        identity = json.loads(args.identity_result.read_text())
        if identity["status"] != "passed" or not identity["identity_hard_gate_passed"]:
            raise RuntimeError("U1 high-N is blocked by identity gate")
    binding = prior_u1._load_frozen_binding(args.binding)
    runtime = highn._checkpoint_runtime(argparse.Namespace(run_dir=args.run_dir))
    dataset = highn._dataset(argparse.Namespace(dataset_root=args.dataset_root, manifest=args.manifest))
    anchors = highn._valid_examples(dataset, binding)[: args.sample_count]
    full, archive_lookup = highn._full_shared(argparse.Namespace(full_fields=args.full_fields))
    preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    support_rows = {
        row["sample_id"]: row for row in preflight.get("supports", {}).get(str(args.resolution), [])
    }
    physics_rows = {row["sample_id"]: row for row in preflight["samples"]}
    native_targets = _edge_targets(args.native_padding_result)
    query_targets = native_targets if args.resolution == 1024 else _edge_targets(args.query_padding_result)
    asymmetric_targets = _combined_targets(native_targets, query_targets)
    graph_key = runner._metadata_key(int(runtime["run_config"]["graph_seed"]))
    graph_config = dict(runtime["graph_config"]); graph_config["subsample_factor"] = 4
    model = GraphNeuralOperator(**runtime["model_config"])
    params = runner._device_params(runtime["checkpoint"]["params"])
    params_before = highn._tree_sha256(runtime["checkpoint"]["params"])
    rows = []; metric_rows = []; full_metric_rows = []; forward_times = []
    identity_fields = protocol["identity_hard_gate"]["fields"]
    compiled = None

    with h5py.File(args.full_fields, "r") as temperature_archive:
        for number, anchor in enumerate(anchors, start=1):
            builder = Heat3DGraphBuilder(**graph_config)
            anchor_coords = runner._graph_coords_for_example(anchor, runtime["stats"])
            native_metadata = builder.build_metadata(anchor_coords, key=graph_key)
            if args.resolution == 1024:
                query = anchor; asymmetric = native_metadata
            else:
                support = highn._load_support(Path(support_rows[anchor.sample_id]["support_file"]))
                query = highn._query_example(anchor, support, full["coords"])
                asymmetric, graph_audit = prior_u1._strict_asymmetric_metadata(
                    builder, native_metadata, anchor_coords,
                    runner._graph_coords_for_example(query, runtime["stats"]),
                )
                if not graph_audit["query_inside_native_domain"]:
                    raise RuntimeError("U1 query outside native domain")
            targets = native_targets if args.resolution == 1024 else asymmetric_targets
            anchor_group = highn._prepare_group(
                example=anchor, anchor=anchor, runtime=runtime, builder=builder,
                metadata=native_metadata, edge_targets=native_targets,
            )
            output_group = anchor_group if args.resolution == 1024 else highn._prepare_group(
                example=query, anchor=anchor, runtime=runtime, builder=builder,
                metadata=asymmetric, edge_targets=targets,
            )
            graphs = output_group["graphs"] if args.resolution != 1024 else anchor_group["graphs"]
            local_p2r = graphs.p2r if args.resolution == 1024 else _dummy_local_p2r(builder, asymmetric)
            kwargs = _model_kwargs(anchor_group, output_group)

            if args.resolution == 1024:
                public = runner._model_apply(model, params, highn._model_group(anchor_group))
                standard = model.apply(
                    {"params": params}, inputs_in=anchor_group["inputs"],
                    inputs_out=anchor_group["inputs"], graphs=graphs,
                    output_local_p2r=graphs.p2r, split=False,
                    method=_trace_method, **kwargs,
                )
                split = model.apply(
                    {"params": params}, inputs_in=anchor_group["inputs"],
                    inputs_out=anchor_group["inputs"], graphs=graphs,
                    output_local_p2r=graphs.p2r, split=True,
                    method=_trace_method, **kwargs,
                )
                jax.block_until_ready(split["raw_temperature"])
                comparisons = {
                    field: _tree_diff(standard[field], split[field]) for field in identity_fields
                }
                comparisons["public_raw_temperature"] = _tree_diff(
                    public["raw_temperature"], standard["raw_temperature"]
                )
                comparisons["public_s_hat"] = _tree_diff(public["s_hat"], standard["s_hat"])
                locality = _tree_diff(
                    split["encoder_input_pnodes"], split["encoder_output_local_pnodes"]
                )
                passed = all(row["array_equal"] for row in comparisons.values()) and locality["array_equal"]
                rows.append({
                    "sample_id": anchor.sample_id, "comparisons": comparisons,
                    "encoder_pnode_local_transform_exact": locality, "passed": passed,
                })
                if not passed:
                    print(json.dumps(rows[-1], indent=2, sort_keys=True), flush=True)
                    raise RuntimeError(f"U1 identity fail-fast: {anchor.sample_id}")
            else:
                if number == 1:
                    pre = model.apply(
                        {"params": params}, inputs=anchor_group["inputs"], graphs=graphs,
                        global_context=anchor_group.get("global_context"),
                        qk_region_features=anchor_group.get("qk_region_features"),
                        method=_pre_bypass_method,
                    )
                    pre_shape = list(np.asarray(pre["decoder_pre_bypass"]).shape)
                    if pre_shape[2] != args.resolution:
                        raise RuntimeError(f"U1 pre-bypass output is not N nodes: {pre_shape}")
                started = time.perf_counter()
                split = model.apply(
                    {"params": params}, inputs_in=anchor_group["inputs"],
                    inputs_out=output_group["inputs"], graphs=graphs,
                    output_local_p2r=local_p2r, split=True,
                    method=_trace_method, **kwargs,
                )
                jax.block_until_ready(split["raw_temperature"])
                elapsed = time.perf_counter() - started
                if compiled is None:
                    compiled = jax.jit(lambda p, i, o, g, l, kw: model.apply(
                        {"params": p}, inputs_in=i, inputs_out=o, graphs=g,
                        output_local_p2r=l, split=True, method=_trace_method, **kw,
                    ))
                    warm = compiled(params, anchor_group["inputs"], output_group["inputs"], graphs, local_p2r, kwargs)
                    jax.block_until_ready(warm["raw_temperature"])
                started = time.perf_counter()
                split = compiled(params, anchor_group["inputs"], output_group["inputs"], graphs, local_p2r, kwargs)
                jax.block_until_ready(split["raw_temperature"])
                forward_times.append(time.perf_counter() - started)
                raw = np.asarray(split["raw_temperature"], dtype=np.float64)[0, 0, :, 0]
                if not np.all(np.isfinite(raw)):
                    raise RuntimeError(f"U1 nonfinite: {anchor.sample_id}")
                support = highn._load_support(Path(support_rows[anchor.sample_id]["support_file"]))
                indices = np.asarray(support["selected_indices"], dtype=np.int64)
                truth = np.asarray(
                    temperature_archive["samples/deltaT_K"][archive_lookup[anchor.sample_id]], dtype=np.float64,
                )
                with np.load(physics_rows[anchor.sample_id]["physics_cache_file"], allow_pickle=False) as physics:
                    full_q = np.asarray(physics["q_W_m3"], dtype=np.float64)
                support_delta = raw - highn.REFERENCE_K
                metric_rows.append(highn._metric_row(
                    support_delta, truth[indices], np.asarray(support["operator_control_volume"]),
                    full["coords"][indices], np.asarray(support["layer_id"]), np.asarray(support["q_W_m3"]),
                ))
                baseline = json.loads(args.baseline_result.read_text())
                map_row = next(row for row in baseline["reconstruction_cache"]["samples"] if row["sample_id"] == anchor.sample_id)
                mapping = candidate.publication._load_mapping_no_audit(Path(map_row["cache_file"]))
                full_delta = np.sum(
                    support_delta[np.asarray(mapping.neighbor_local_indices)]
                    * np.asarray(mapping.neighbor_weights), axis=1,
                )
                full_metric_rows.append(highn._metric_row(
                    full_delta, truth, full["cv"], full["coords"], full["layer"], full_q,
                ))
                rows.append({
                    "sample_id": anchor.sample_id, "pre_bypass_output_shape": pre_shape if number == 1 else None,
                    "finite": True, "first_eager_seconds": elapsed,
                    "steady_forward_seconds": forward_times[-1],
                    "encoder_locality": _tree_diff(
                        split["encoder_input_pnodes"], split["encoder_output_local_pnodes"]
                    ) if args.resolution == 1024 else {
                        "array_equal": "not_same_nodes_high_n",
                        "input_shape": list(np.asarray(split["encoder_input_pnodes"]).shape),
                        "output_shape": list(np.asarray(split["encoder_output_local_pnodes"]).shape),
                    },
                })
            print(f"[U1-split] N={args.resolution} {number}/{args.sample_count}", flush=True)

    if args.resolution == 1024:
        result = {
            "schema_version": "heat3d_v6_p1i_u1_split_identity_v1",
            "status": "passed", "resolution": 1024,
            "sample_count": args.sample_count, "identity_hard_gate_passed": True,
            "samples": rows,
        }
    else:
        result = {
            "schema_version": "heat3d_v6_p1i_u1_split_high_n_v1",
            "status": "passed" if args.sample_count == 32 else "passed_smoke",
            "resolution": args.resolution, "sample_count": args.sample_count,
            "pre_bypass_output_shape": rows[0]["pre_bypass_output_shape"],
            "support_accuracy": qualification.metric_accumulate(metric_rows, full=False),
            "full_field_accuracy": qualification.metric_accumulate(full_metric_rows, full=True),
            "steady_forward": _stats(forward_times),
            "memory": candidate.publication._device_memory(),
            "samples": rows,
        }
    params_after = highn._tree_sha256(runtime["checkpoint"]["params"])
    result.update({
        "backend": backend,
        "protocol_sha256": _sha256(args.protocol),
        "checkpoint_parameter_tree_before": params_before,
        "checkpoint_parameter_tree_after": params_after,
        "checkpoint_parameters_unchanged": params_before == params_after,
        "role_contract": protocol["role_contract"],
    })
    if not result["checkpoint_parameters_unchanged"]:
        raise RuntimeError("U1 checkpoint parameter tree changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "resolution": args.resolution}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
