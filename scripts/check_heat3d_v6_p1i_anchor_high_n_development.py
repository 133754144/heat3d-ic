#!/usr/bin/env python3
"""Check the frozen P1i Anchor-derived valid32 High-N development bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANDATORY = (1024, 4096, 8192, 16384)
HIGH_N = (4096, 8192, 16384)
GPU_ONLY_MANDATORY = (4096, 8192, 16384)
GPU_ONLY_OPTIONAL = (32768, 65536)
GPU_ONLY_LADDER = GPU_ONLY_MANDATORY + GPU_ONLY_OPTIONAL


class CheckError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def check_static(binding_path: Path) -> dict[str, Any]:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(binding["status"] == "frozen_after_three_seed_r0_pass", "binding status drift")
    require(tuple(binding["resolutions"]["mandatory"]) == MANDATORY, "mandatory ladder drift")
    require(binding["resolutions"]["optional_valid_only"] == 32768, "optional ladder drift")
    require(binding["development_subset"]["count"] == 32, "valid32 count drift")
    require(len(binding["development_subset"]["sample_ids"]) == 32, "valid32 IDs drift")
    for row in binding["code_fingerprints"].values():
        require(sha256(ROOT / row["path"]) == row["sha256"], f"code fingerprint drift: {row['path']}")
    return binding


def check_results(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    preflight_path = root / "actual_data_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "passed", "preflight failed")
    require(all(value is True for value in preflight["checks"].values()), "preflight check failed")
    ids = binding["development_subset"]["sample_ids"]
    require(preflight["sample_ids"] == ids, "preflight valid32 order drift")
    require(preflight["role_contract"]["test_accessed"] is False, "test accessed")
    require(preflight["role_contract"]["sealed_accessed"] is False, "sealed accessed")
    require(preflight["role_contract"]["training_executed"] is False, "training executed")

    previous: dict[str, np.ndarray] = {}
    for resolution in HIGH_N:
        rows = preflight["supports"][str(resolution)]
        require([row["sample_id"] for row in rows] == ids, f"N={resolution} support IDs drift")
        for row in rows:
            path = Path(row["support_file"])
            require(path.is_file() and sha256(path) == row["support_file_sha256"], "support file hash drift")
            with np.load(path, allow_pickle=False) as payload:
                indices = np.asarray(payload["selected_indices"], dtype=np.int32)
                cv = np.asarray(payload["operator_control_volume"], dtype=np.float64)
                q = np.asarray(payload["q_W_m3"], dtype=np.float64)
            require(indices.shape == (resolution,), "support shape drift")
            require(len(np.unique(indices)) == resolution, "support duplicates")
            require(np.all(np.isfinite(cv)) and np.all(cv > 0.0), "invalid operator CV")
            require(np.all(np.isfinite(q)) and np.all(q >= 0.0), "invalid q")
            if row["sample_id"] in previous:
                require(np.array_equal(indices[:len(previous[row["sample_id"]])], previous[row["sample_id"]]),
                        "nested support prefix drift")
            previous[row["sample_id"]] = indices

    result_rows = []
    for resolution in MANDATORY:
        path = root / f"resolution_{resolution}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["status"] == "passed", f"N={resolution} failed")
        require(payload["sample_ids"] == ids, f"N={resolution} sample order drift")
        gates = payload["implementation_hard_gates"]
        negative = {"test_accessed", "sealed_accessed", "training_executed"}
        require(all(value is True for key, value in gates.items() if key not in negative),
                f"N={resolution} positive hard gate failed")
        require(all(gates[key] is False for key in negative), f"N={resolution} forbidden role accessed")
        require(payload["accuracy_is_not_an_implementation_gate"] is True, "accuracy gate drift")
        require(finite_tree(payload["support_metrics"]), "non-finite support metric")
        require(finite_tree(payload["full_field_model_plus_reconstruction"]), "non-finite full metric")
        require(finite_tree(payload["oracle_sampling_reconstruction_floor"]), "non-finite floor metric")
        replay = payload["fixed_input_gpu_replay"]["prediction"]
        require(math.isfinite(replay["max_abs_error_K"]) and math.isfinite(replay["rmse_K"]),
                "non-finite GPU replay diagnostic")
        cache_equivalence = payload["cached_uncached_prediction_equivalence"]
        require(cache_equivalence["deterministic_cpu_hard_gate"]["status"] == "passed",
                "deterministic CPU cache gate failed")
        require(cache_equivalence["deterministic_cpu_hard_gate"]["cached_uncached_prediction"]["max_abs_error_K"] <= 1.0e-6,
                "deterministic CPU cache prediction drift")
        require(cache_equivalence["gpu_reduction_nondeterminism_is_not_graph_cache_failure"] is True,
                "GPU diagnostic classification drift")
        cross_backend = payload["cross_backend_graph_diagnostic"]
        require(cross_backend["real_edge_topology_exact"] is True,
                "cross-backend real-edge topology drift")
        require(cross_backend["known_float_normalization_drift_not_edge_topology_drift"] is True,
                "cross-backend float-drift classification missing")
        require(cross_backend["cache_directories_are_backend_isolated_key_payload_is_unchanged"] is True,
                "backend cache isolation contract drift")
        graph_rows = payload["graph_cache"]["samples"]
        require(len(graph_rows) == 32, "graph cache sample count drift")
        require(all(row["cached_uncached_hash_exact"] for row in graph_rows), "graph cache equivalence failed")
        for row in graph_rows:
            path_graph = Path(row["cache_file"])
            require(path_graph.is_file() and sha256(path_graph) == row["cache_file_sha256"], "graph cache hash drift")
        reconstruction = payload["reconstruction_cache"]["samples"]
        require(len(reconstruction) == 32, "reconstruction cache sample count drift")
        for row in reconstruction:
            path_map = Path(row["cache_file"])
            require(path_map.is_file() and sha256(path_map) == row["cache_file_sha256"], "map cache hash drift")
        prediction = payload["prediction_artifact"]
        prediction_path = Path(prediction["path"])
        require(prediction_path.is_file() and sha256(prediction_path) == prediction["sha256"],
                "prediction artifact hash drift")
        result_rows.append(payload)

    state = json.loads((root / "execution_state.json").read_text(encoding="utf-8"))
    require(state["status"] == "passed", "execution state failed")
    require([row["resolution"] for row in state["execution"]] == list(MANDATORY), "execution order drift")
    require(all(row["returncode"] == 0 for row in state["execution"]), "worker return code failed")
    require(all(row["cpu_cache_audit_returncode"] == 0 for row in state["execution"]),
            "CPU cache audit return code failed")
    require(state["higher_resolutions_started_after_4096_failure"] is False, "fail-fast contract drift")
    require(not (root / "resolution_32768.json").exists(), "forbidden 32768 result exists")

    closeout_path = root / "anchor_high_n_valid32_closeout.json"
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    require(closeout["status"] == "passed", "closeout failed")
    require([row["resolution"] for row in closeout["rows"]] == list(MANDATORY), "closeout ladder drift")
    require(closeout["accuracy_used_as_gate"] is False, "accuracy used as gate")
    require(closeout["role_contract"]["resolution_32768_executed"] is False, "32768 executed")
    csv_path = root / "anchor_high_n_valid32_summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 4, "summary CSV row count drift")
    require((root / "anchor_high_n_valid32_closeout.md").is_file(), "closeout Markdown missing")
    return {"preflight": preflight, "results": result_rows, "closeout": closeout}


def check_failure_results(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    preflight = json.loads((root / "actual_data_preflight.json").read_text())
    require(preflight["status"] == "passed", "failure closeout preflight failed")
    require(preflight["sample_ids"] == binding["development_subset"]["sample_ids"],
            "failure closeout valid32 IDs drift")
    r1024 = json.loads((root / "resolution_1024.json").read_text())
    r4096 = json.loads((root / "resolution_4096.json").read_text())
    require(r1024["status"] == "passed", "1024 did not pass")
    require(r4096["status"] == "failed", "4096 did not fail closed")
    positive_false = [
        key for key, value in r4096["implementation_hard_gates"].items()
        if value is False and key not in {"test_accessed", "sealed_accessed", "training_executed"}
    ]
    require(positive_false == ["cross_backend_real_edge_topology_exact"],
            f"unexpected 4096 failures: {positive_false}")
    require(r4096["cached_uncached_prediction_equivalence"]["deterministic_cpu_hard_gate"]["status"] == "passed",
            "4096 deterministic CPU cache gate failed")
    require(all(row["cached_uncached_hash_exact"] for row in r4096["graph_cache"]["samples"]),
            "4096 same-GPU cache hash failed")
    require(r4096["cross_backend_graph_diagnostic"]["real_edge_topology_exact"] is False,
            "4096 cross-backend topology unexpectedly exact")
    for payload in (r1024, r4096):
        require(payload["role_contract"]["test_accessed"] is False, "test accessed")
        require(payload["role_contract"]["sealed_accessed"] is False, "sealed accessed")
        require(payload["role_contract"]["training_executed"] is False, "training executed")
        require(finite_tree(payload["support_metrics"]), "non-finite support metrics")
        require(finite_tree(payload["full_field_model_plus_reconstruction"]), "non-finite full metrics")
        require(finite_tree(payload["oracle_sampling_reconstruction_floor"]), "non-finite floor metrics")
    require(not any((root / f"resolution_{n}.json").exists() for n in (8192, 16384, 32768)),
            "higher resolution exists after 4096 fail-close")
    state = json.loads((root / "execution_state.json").read_text())
    require(state["status"] == "failed", "execution state is not failed")
    require([row["resolution"] for row in state["execution"]] == [1024, 4096],
            "fail-close execution order drift")
    require(state["higher_resolutions_started_after_4096_failure"] is False,
            "higher resolution started after 4096 failure")
    closeout = json.loads((root / "anchor_high_n_valid32_failure_closeout.json").read_text())
    require(closeout["status"] == "incomplete_fail_closed_at_4096_graph_gate",
            "failure closeout status drift")
    require(closeout["failed_hard_gates"] == ["cross_backend_real_edge_topology_exact"],
            "failure closeout reason drift")
    require(closeout["role_contract"]["resolution_8192_executed"] is False, "8192 executed")
    require(closeout["role_contract"]["resolution_16384_executed"] is False, "16384 executed")
    require(closeout["role_contract"]["resolution_32768_executed"] is False, "32768 executed")
    with (root / "anchor_high_n_valid32_failure_closeout.csv").open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == 2, "failure CSV row count drift")
    require((root / "anchor_high_n_valid32_failure_closeout.md").is_file(), "failure Markdown missing")
    return {"preflight": preflight, "resolution_1024": r1024,
            "resolution_4096": r4096, "closeout": closeout}


def check_gpu_only_results(
    root: Path,
    binding: dict[str, Any],
    amendment_path: Path,
    baseline_root: Path,
) -> dict[str, Any]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    require(amendment["status"] == "frozen_before_gpu_only_high_n_execution",
            "GPU-only amendment not frozen")
    require(amendment["frozen_binding"]["sha256"] == sha256(
        ROOT / amendment["frozen_binding"]["path"]
    ), "GPU-only amendment binding hash drift")
    require(amendment["backend_contract"]["formal_backend"] == "gpu",
            "formal backend is not GPU")
    require(amendment["backend_contract"]["cross_backend_topology_equality"] == "report_only",
            "cross-backend topology was not downgraded to report-only")
    baseline = baseline_root / "resolution_1024.json"
    require(sha256(baseline) == amendment["baseline_1024_artifacts"]["result_sha256"],
            "1024 baseline result drift")
    require(sha256(baseline_root / "resolution_1024_predictions.npz")
            == amendment["baseline_1024_artifacts"]["predictions_sha256"],
            "1024 anchor predictions drift")

    preflight = json.loads((root / "actual_data_preflight.json").read_text(encoding="utf-8"))
    require(preflight["status"] == "passed", "GPU-only preflight failed")
    require(preflight["sample_ids"] == binding["development_subset"]["sample_ids"],
            "GPU-only valid32 order drift")
    require(tuple(sorted(map(int, preflight["supports"]))) == GPU_ONLY_LADDER,
            "GPU-only support ladder drift")
    for resolution in GPU_ONLY_LADDER:
        rows = preflight["supports"][str(resolution)]
        require(len(rows) == 32, f"N={resolution} support count drift")
        feasibility_path = root / f"resolution_{resolution}_feasibility.json"
        result_path = root / f"resolution_{resolution}.json"
        if not feasibility_path.exists():
            require(not result_path.exists(), f"N={resolution} result exists without feasibility")
            continue
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        if feasibility["status"] == "passed":
            require(all(feasibility["checks"].values()), f"N={resolution} feasibility gate drift")
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        require(result["sample_ids"] == binding["development_subset"]["sample_ids"],
                f"N={resolution} sample order drift")
        negative = {"test_accessed", "sealed_accessed", "training_executed"}
        gates = result["implementation_hard_gates"]
        require(all(value is True for key, value in gates.items() if key not in negative),
                f"N={resolution} GPU hard gate failed")
        require(all(gates[key] is False for key in negative), f"N={resolution} role violation")
        require("cross_backend_real_edge_topology_exact" not in gates,
                "cross-backend diagnostic remains a hard gate")
        require(result["cross_backend_graph_diagnostic"]["gate_role"] == "report_only",
                "cross-backend diagnostic role drift")
        require(result["cross_backend_graph_diagnostic"]["executed_this_run"] is False,
                "CPU graph diagnostic unexpectedly executed")
        require(result["cached_uncached_prediction_equivalence"]
                ["same_gpu_within_amended_numeric_tolerance"] is True,
                "same-GPU cached/fresh prediction gate failed")
        require(result["protocol_amendment"]["formal_backend"] == "gpu",
                "result protocol backend drift")
        require(finite_tree(result["support_metrics"]), "non-finite support metrics")
        require(finite_tree(result["full_field_model_plus_reconstruction"]),
                "non-finite full-field metrics")
        require(finite_tree(result["oracle_sampling_reconstruction_floor"]),
                "non-finite oracle floor")

    closeout = json.loads((root / "gpu_only_high_n_closeout.json").read_text(encoding="utf-8"))
    require(closeout["status"] in {
        "passed_core_and_optional_ladder", "passed_core_optional_stopped"
    }, "GPU-only mandatory ladder did not close")
    require(closeout["cross_backend_graph_topology"] == "report_only_not_executed",
            "closeout cross-backend role drift")
    resolutions = [row["resolution"] for row in closeout["rows"]]
    require(resolutions[0] == 1024, "curve lacks frozen 1024 baseline")
    require(all(resolution in resolutions for resolution in GPU_ONLY_MANDATORY),
            "curve lacks mandatory GPU resolutions")
    require(resolutions == sorted(resolutions), "curve resolution order drift")
    require(closeout["role_contract"]["test_accessed"] is False, "test accessed")
    require(closeout["role_contract"]["sealed_accessed"] is False, "sealed accessed")
    require(closeout["role_contract"]["training_executed"] is False, "training executed")
    require((root / "gpu_only_high_n_accuracy_resolution_latency.csv").is_file(),
            "GPU-only curve CSV missing")
    require((root / "gpu_only_high_n_closeout.md").is_file(), "GPU-only Markdown missing")
    return {"preflight": preflight, "closeout": closeout, "amendment": amendment}


def check_gpu_only_archive() -> dict[str, Any]:
    config = ROOT / "configs/heat3d_v6_p1i"
    closeout = json.loads(
        (config / "v6_p1i_gpu_only_high_n_closeout.json").read_text(encoding="utf-8")
    )
    execution = json.loads(
        (config / "v6_p1i_gpu_only_high_n_execution_manifest.json").read_text(encoding="utf-8")
    )
    amendment_path = config / "v6_p1i_gpu_only_high_n_protocol_amendment.json"
    require(closeout["status"] == "passed_core_and_optional_ladder",
            "archived GPU-only closeout did not pass")
    require([row["resolution"] for row in closeout["rows"]]
            == [1024, 4096, 8192, 16384, 32768, 65536],
            "archived accuracy-resolution ladder drift")
    require(all(row["status"] == "passed" for row in closeout["rows"]),
            "archived resolution status failed")
    require(all(finite_tree(row) for row in closeout["rows"]), "archived metric is non-finite")
    require(closeout["protocol_amendment"]["sha256"] == sha256(amendment_path),
            "archived protocol amendment hash drift")
    require(execution["protocol_amendment_sha256"] == sha256(amendment_path),
            "execution manifest protocol hash drift")
    require(execution["status"] == closeout["status"], "execution/closeout status mismatch")
    require(execution["historical_failure_artifact_preserved"] is True,
            "historical failure preservation missing")
    require(closeout["accuracy_used_as_gate"] is False, "accuracy used as gate")
    for resolution in GPU_ONLY_LADDER:
        feasibility = json.loads(
            (config / f"v6_p1i_gpu_only_high_n_feasibility_{resolution}.json")
            .read_text(encoding="utf-8")
        )
        require(feasibility["status"] == "passed", f"N={resolution} feasibility failed")
        require(all(feasibility["checks"].values()), f"N={resolution} feasibility check drift")
    manifest_lines = (
        config / "v6_p1i_gpu_only_high_n_artifact_sha256.txt"
    ).read_text(encoding="utf-8").splitlines()[1:]
    manifest = {Path(line.split(maxsplit=2)[2]).name: line.split(maxsplit=2)[1]
                for line in manifest_lines}
    for row in closeout["rows"][1:]:
        require(manifest[f"resolution_{row['resolution']}.json"] == row["result_sha256"],
                f"N={row['resolution']} result SHA manifest mismatch")
    for payload in (closeout["role_contract"], execution["role_contract"]):
        require(payload["test_accessed"] is False, "test accessed")
        require(payload["sealed_accessed"] is False, "sealed accessed")
        require(payload["training_executed"] is False, "training executed")
        require(payload["three_seed_valid128_executed"] is False,
                "three-seed valid128 executed")
    return {"closeout": closeout, "execution": execution}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_high_n_implementation_binding.json")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--expected-failure-resolution", type=int, choices=(4096,))
    parser.add_argument("--gpu-only-amendment", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--check-gpu-only-archive", action="store_true")
    args = parser.parse_args()
    binding = check_static(args.binding)
    if args.check_gpu_only_archive:
        check_gpu_only_archive()
    if args.results_root is not None:
        if args.gpu_only_amendment is not None:
            require(args.baseline_root is not None, "--baseline-root required for GPU-only check")
            check_gpu_only_results(
                args.results_root, binding, args.gpu_only_amendment, args.baseline_root
            )
        elif args.expected_failure_resolution == 4096:
            check_failure_results(args.results_root, binding)
        else:
            check_results(args.results_root, binding)
    print(json.dumps({
        "status": "passed", "static_binding": True,
        "results_checked": args.results_root is not None,
        "expected_failure_resolution": args.expected_failure_resolution,
        "mandatory_resolutions": list(MANDATORY),
        "gpu_only": args.gpu_only_amendment is not None,
        "gpu_only_archive_checked": args.check_gpu_only_archive,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
