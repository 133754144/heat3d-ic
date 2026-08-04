#!/usr/bin/env python3
"""Valid-only direct-N structured-support OOD compatibility diagnostic.

This is deliberately separate from the frozen production route.  It evaluates
the unchanged checkpoint directly on a legal structured support of N nodes and
never feeds labels into graph construction, context construction, or inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np
import yaml


ROOT = Path(os.environ.get("HEAT3D_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import benchmark_heat3d_v6_inference_qualification as base  # noqa: E402
import benchmark_heat3d_v6_p1i_resolution as prior  # noqa: E402
import benchmark_heat3d_v6_unified_resolution as unified  # noqa: E402
from scripts import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


RESOLUTIONS = unified.RESOLUTIONS
STATES = ("process_cold", "known_topology_new_physics", "fully_cached")


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 20:
        raise RuntimeError(f"formal timing distribution requires >=20 values, got {array.size}")
    return {
        "count": int(array.size), "median": float(np.median(array)),
        "mean": float(np.mean(array)), "std": float(np.std(array, ddof=1)),
        "p95": float(np.quantile(array, 0.95)), "min": float(np.min(array)),
        "max": float(np.max(array)), "values": array.tolist(),
    }


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def direct_example(
    data: base.FamilyData,
    row: Mapping[str, Any],
    resolution: int,
    random_core: Any | None,
    random_config: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    original, public = data.load_example(row)
    sample_id = str(row["sample_id"])
    if data.family == "p1i":
        meta = json.loads((data.sample_dir(row) / "sample_meta.json").read_text(encoding="utf-8"))
        mesh = prior.core.build_mesh(unified.target_physics(meta["physics"], resolution))
        k, q, power = prior._continuous_fields(meta, mesh)
        if power["relative_power_error"] > 1e-12:
            raise RuntimeError("P1i direct-support source power drift")
    else:
        cases = {str(item["sample_id"]): item for item in random_config["cases"]}
        groups = {str(item["group_id"]): item for item in random_config["layout_groups"]}
        case = cases[sample_id]; group = groups[str(case["group_id"])]
        mesh = random_core.build_mesh(unified.target_physics(random_config["physics"], resolution))
        layout = random_core.validate_layout(group, mesh)
        k, q, _ = random_core.build_case_fields(case, group, mesh, layout)
        meta = dict(original.meta)
        meta["top_h_W_m2K"] = float(case["top_h_W_m2K"])
        meta["bottom_h_W_m2K"] = float(case["bottom_h_W_m2K"])
    example = prior._example(original, meta, mesh, k, q)
    truth_full = data.truth(row, include_full_kq=False)["full_delta"]
    truth = unified.regular_truth(data.full_shared(), truth_full, np.asarray(mesh["coords"], dtype=np.float64))
    weights = np.asarray(mesh.get("weights") if "weights" in mesh else mesh["info"]["weights"], dtype=np.float64)
    return example, {
        "truth": truth, "weights": weights, "coords": np.asarray(mesh["coords"]),
        "layer": np.asarray(mesh["layer_ids"], dtype=np.int32), "q": np.asarray(q),
        "reference_K": float(public["reference_K"]),
    }


def direct_runtime(args: argparse.Namespace) -> base.ModelRuntime:
    runtime = base.ModelRuntime(
        args.run_dir, args.checkpoint_sha256, args.checkpoint_epoch, None,
        verify_checkpoint_sha=not args.checkpoint_sha_preverified,
    )
    graph_config = dict(runtime.run_config["graph_config"])
    graph_config["discrete_graph_backend"] = "sparse_kdtree_v1"
    graph_config["discrete_graph_chunk_size"] = 2048
    runtime.builder = runner.RunSharedSupportGraphBuilder(runner.Heat3DGraphBuilder(**graph_config))
    return runtime


def one(
    data: base.FamilyData,
    runtime: base.ModelRuntime,
    row: Mapping[str, Any],
    resolution: int,
    random_core: Any | None,
    random_config: Mapping[str, Any] | None,
    groups: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    e2e = time.perf_counter()
    started = time.perf_counter(); example, public = direct_example(data, row, resolution, random_core, random_config); data_s = time.perf_counter() - started
    sample_id = str(row["sample_id"])
    started = time.perf_counter(); group = groups.get(sample_id) or runtime.graph(example); graph_s = time.perf_counter() - started
    started = time.perf_counter(); prediction = runtime.forward(group); forward_s = time.perf_counter() - started
    started = time.perf_counter(); output_bytes = unified.serialize(prediction); output_s = time.perf_counter() - started
    cutoff = time.perf_counter(); wall = cutoff - e2e
    metric_row = {
        "prediction": prediction - public["reference_K"], "truth": public["truth"],
        "weights": public["weights"], "coords": public["coords"],
        "layer": public["layer"], "q": public["q"],
    }
    return {
        "data_seconds": data_s, "graph_seconds": graph_s,
        "jit_or_forward_seconds": forward_s, "output_seconds": output_s,
        "continuous_wall_seconds": wall, "output_bytes": output_bytes,
        "prediction_serialization_completed_monotonic_s": cutoff,
    }, {"metric_row": metric_row}


def worker(args: argparse.Namespace) -> int:
    data = base.FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    selected = data.selected_rows(args.sample_count)
    if args.sample_id:
        selected = [data.row_by_id[args.sample_id]]
    random_core = unified.import_random_core(args.randomblock_core) if args.family == "randomblock" else None
    random_config = yaml.safe_load(args.randomblock_config.read_text(encoding="utf-8")) if args.family == "randomblock" else None
    runtime = direct_runtime(args); groups: dict[str, Any] = {}; cache_prep = 0.0
    if args.state == "known_topology_new_physics":
        warm = data.warmup_rows(data.selected_rows(args.sample_count))[0]
        warm_example, _ = direct_example(data, warm, args.resolution, random_core, random_config)
        runtime.forward(runtime.graph(warm_example))
    elif args.state == "fully_cached":
        started = time.perf_counter()
        for row in selected:
            example, _ = direct_example(data, row, args.resolution, random_core, random_config)
            group = runtime.graph(example); runtime.forward(group)
            groups[str(row["sample_id"])] = group
        cache_prep = time.perf_counter() - started
    measurements = []; metric_rows = []
    for row in selected:
        measurement, extra = one(data, runtime, row, args.resolution, random_core, random_config, groups)
        measurements.append(measurement); metric_rows.append(extra["metric_row"])
    metrics = base.metric_accumulate(metric_rows, full=False)
    metrics["domain"] = f"direct_structured_support_{args.resolution}"
    payload = {
        "schema_version": "heat3d_v6_direct_resolution_worker_v1", "status": "passed",
        "family": args.family, "resolution": args.resolution, "state": args.state,
        "sample_ids": [str(row["sample_id"]) for row in selected],
        "measurements": measurements, "stage_timing": unified.summarize(measurements),
        "metrics": metrics,
        "cache_preparation_seconds_outside_timing": cache_prep,
        "process_peak_ram_bytes": rss_bytes(), "device_memory": base.device_memory(),
        "contract": {
            "diagnostic_only": True, "structured_support_OOD": True,
            "scientific_graph_parameters_unchanged": True,
            "engineering_graph_backend": "sparse_kdtree_v1", "batch_size": 1,
            "labels_excluded_from_inputs_and_timing": True,
        },
        "accessed_roles": ["valid_iid", "train_frozen_normalization_metadata"],
        "test_accessed": False, "sealed_accessed": False, "training_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def command(args: argparse.Namespace, state: str, output: Path, sample_id: str | None = None) -> list[str]:
    values = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--family", args.family,
        "--state", state, "--resolution", str(args.resolution), "--sample-count", str(args.sample_count),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--run-dir", str(args.run_dir),
        "--checkpoint-sha256", args.checkpoint_sha256, "--checkpoint-epoch", str(args.checkpoint_epoch),
        "--output", str(output), "--checkpoint-sha-preverified",
    ]
    if args.randomblock_config: values += ["--randomblock-config", str(args.randomblock_config)]
    if args.randomblock_core: values += ["--randomblock-core", str(args.randomblock_core)]
    if sample_id: values += ["--sample-id", sample_id]
    return values


def run_child(values: list[str]) -> tuple[dict[str, Any], float]:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MEM_FRACTION": "0.85", "HEAT3D_REPO_ROOT": str(ROOT)})
    started = time.perf_counter()
    done = subprocess.run(values, env=env, text=True, capture_output=True, timeout=300)
    if done.returncode:
        raise RuntimeError(f"child failed: {' '.join(values)}\n{done.stdout}\n{done.stderr}")
    payload = json.loads(Path(values[values.index("--output") + 1]).read_text(encoding="utf-8"))
    wall = time.perf_counter() - started
    if payload["state"] == "process_cold":
        wall = float(payload["measurements"][-1]["prediction_serialization_completed_monotonic_s"]) - started
    return payload, wall


def orchestrate(args: argparse.Namespace) -> int:
    checkpoint = args.run_dir / "params_best_valid_point_global.pkl"
    if base.sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA preflight failed")
    data = base.FamilyData(
        family=args.family, dataset_root=args.dataset_root, manifest_path=args.manifest,
        full_fields_path=args.full_fields, randomblock_config=args.randomblock_config,
    )
    selected = data.selected_rows(args.sample_count); args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.family == "randomblock":
        preflight = unified.random_resolution_preflight(args, selected)
        if preflight["status"] != "passed":
            output = {
                "schema_version": "heat3d_v6_direct_resolution_family_v1", "status": "passed",
                "family": args.family, "resolution": args.resolution, "sample_count": len(selected),
                "sample_ids": [str(row["sample_id"]) for row in selected],
                "states": {state: {"status": "not_run_resolution_infeasible", "preflight": preflight} for state in STATES},
                "new_topology": {"status": "not_applicable_fixed_structured_support"},
                "checkpoint": {"sha256": args.checkpoint_sha256, "epoch": args.checkpoint_epoch},
                "dataset": {"manifest_sha256": base.sha256(args.manifest), "full_fields_sha256": base.sha256(args.full_fields)},
                "commands": [], "contract": {"valid_only": True, "test_accessed": False, "sealed_accessed": False, "sample_count": 32, "diagnostic_only": True},
            }
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            return 0
    results = {}; commands = []
    for state in STATES:
        try:
            if state == "process_cold":
                payloads = []; walls = []
                for index, row in enumerate(selected):
                    out = args.work_dir / f"{args.family}_{args.resolution}_direct_cold_{index:02d}.json"
                    values = command(args, state, out, str(row["sample_id"])); commands.append(" ".join(values))
                    payload, wall = run_child(values); payloads.append(payload); walls.append(wall)
                    print(f"[direct] {args.family} N={args.resolution} cold {index+1}/32 {wall:.3f}s", flush=True)
                results[state] = {
                    "status": "passed", "external_continuous_wall_seconds": distribution(walls),
                    "stage_timing": {key: distribution([float(item["measurements"][0][key]) for item in payloads]) for key in payloads[0]["stage_timing"]},
                    "peak_ram_bytes": max(int(item["process_peak_ram_bytes"]) for item in payloads),
                    "peak_device_bytes": max(int(item["device_memory"]["peak_bytes_in_use"]) for item in payloads),
                    "sample_ids": [item["sample_ids"][0] for item in payloads],
                    "worker_metrics": [item["metrics"] for item in payloads],
                }
            else:
                out = args.work_dir / f"{args.family}_{args.resolution}_direct_{state}.json"
                values = command(args, state, out); commands.append(" ".join(values))
                payload, wall = run_child(values); payload["external_process_wall_seconds"] = wall
                results[state] = payload
        except Exception as error:
            results[state] = {"status": "failed_incompatible_or_oom", "exception_type": type(error).__name__, "exception": str(error)}
    for state in STATES:
        results.setdefault(state, {"status": "not_run_after_prior_direct_failure"})
    output = {
        "schema_version": "heat3d_v6_direct_resolution_family_v1", "status": "passed_with_failures_recorded" if any(item["status"].startswith("failed") for item in results.values()) else "passed",
        "family": args.family, "resolution": args.resolution, "sample_count": len(selected),
        "sample_ids": [str(row["sample_id"]) for row in selected], "states": results,
        "new_topology": {"status": "not_applicable_fixed_structured_support"},
        "commands": commands,
        "checkpoint": {"sha256": args.checkpoint_sha256, "epoch": args.checkpoint_epoch},
        "dataset": {"manifest_sha256": base.sha256(args.manifest), "full_fields_sha256": base.sha256(args.full_fields)},
        "contract": {"valid_only": True, "test_accessed": False, "sealed_accessed": False, "sample_count": 32, "diagnostic_only": True, "route": "direct_N_structured_support_model_OOD"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", choices=("p1i", "randomblock"), required=True)
    parser.add_argument("--state", choices=STATES)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    parser.add_argument("--sample-count", type=int, default=32); parser.add_argument("--sample-id")
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True); parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint-sha-preverified", action="store_true")
    parser.add_argument("--randomblock-config", type=Path); parser.add_argument("--randomblock-core", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/v6_direct_resolution")); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker and args.state is None: parser.error("worker requires state")
    if args.family == "randomblock" and (args.randomblock_config is None or args.randomblock_core is None): parser.error("randomblock requires config/core")
    return args


if __name__ == "__main__":
    parsed = parse_args(); raise SystemExit(worker(parsed) if parsed.worker else orchestrate(parsed))
