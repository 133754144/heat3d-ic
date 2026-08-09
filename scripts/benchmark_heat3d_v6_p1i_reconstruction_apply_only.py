#!/usr/bin/env python3
"""Timing-only GPU reconstruction apply on frozen P1i graph/map artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_heat3d_v6_p1i_publication_gpu_pipeline as publication
import run_heat3d_v6_p1i_anchor_high_n_development as highn
from rigno.models.rigno import RIGNO as GraphNeuralOperator


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)), "std_seconds": float(np.std(array)),
        "p95_seconds": float(np.percentile(array, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--gpu-only-amendment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("reconstruction apply timing requires GPU")
    binding = highn._binding(args)
    highn._protocol_amendment(args)
    args.preflight = json.loads((args.artifact_root / "actual_data_preflight.json").read_text())
    args.sample_ids = list(binding["development_subset"]["sample_ids"])
    runtime = highn._checkpoint_runtime(args)
    dataset = highn._dataset(args)
    anchors = highn._valid_examples(dataset, binding)
    full, _ = highn._full_shared(args)
    with np.load(args.baseline_root / "resolution_1024_predictions.npz", allow_pickle=False) as payload:
        sample_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
        scales = np.asarray(payload["predicted_scales"], dtype=np.float64)
    if sample_ids != args.sample_ids:
        raise RuntimeError("anchor-scale order drift")
    publication.args_anchor_scales = dict(zip(sample_ids, map(float, scales), strict=True))
    model = GraphNeuralOperator(**runtime["model_config"])
    params = highn.runner._device_params(runtime["checkpoint"]["params"])

    @jax.jit
    def model_core(model_params, group, weights, anchor_scale):
        output = highn.runner._model_apply(model, model_params, group)
        raw = output["raw_temperature"][0, 0, :, 0]
        delta = raw - highn.REFERENCE_K
        normalized = weights / jnp.sum(weights)
        query_scale = jnp.sqrt(jnp.sum(normalized * delta * delta))
        return delta / query_scale * anchor_scale

    @jax.jit
    def apply_only(values, indices, weights):
        return jnp.sum(values[indices] * weights.astype(values.dtype), axis=1)

    rows = []
    first_id = args.sample_ids[0]
    for resolution in args.resolutions:
        payload = publication._load_resolution(args, resolution)
        group, _, device_map, weights, anchor_scale = publication._prepare_case(
            sample_id=first_id, resolution_payload=payload,
            anchors_by_id={anchor.sample_id: anchor for anchor in anchors},
            full_coords=full["coords"], runtime=runtime,
        )
        values = model_core(params, group, jnp.asarray(weights), jnp.asarray(anchor_scale))
        jax.block_until_ready(values)
        compiled = apply_only(values, device_map.neighbor_local_indices, device_map.neighbor_weights)
        jax.block_until_ready(compiled)
        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = apply_only(values, device_map.neighbor_local_indices, device_map.neighbor_weights)
            jax.block_until_ready(result)
            timings.append(time.perf_counter() - started)
        rows.append({"resolution": resolution, **distribution(timings)})
    result = {
        "schema_version": "heat3d_v6_p1i_reconstruction_apply_only_timing_v1",
        "status": "passed", "results": rows,
        "timing_contract": {
            "single_sample_B1": True, "gpu_synchronized": True,
            "graph_and_map_cached": True, "map_build": False,
            "model_forward_timed": False, "labels_or_metrics_read": False,
            "jit_compile_excluded": True,
        },
        "role_contract": {"training": False, "test": False, "sealed": False, "valid32_seed0_only": True},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "passed", "resolutions": args.resolutions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
