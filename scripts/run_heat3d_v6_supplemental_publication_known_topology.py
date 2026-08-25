#!/usr/bin/env python3
"""Execute S1/S2/S3 through the frozen E/U publication runner entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_heat3d_v6_p1i_anchor_high_n_development as highn  # noqa: E402
from heat3d_v6_supplemental_input_adapter import (  # noqa: E402
    array_sha256, load_input_only_example, make_case,
)


ROUTES = (
    "E16384_reconstruction", "U_v2_16384_reconstruction",
    "U_v2_direct240825", "E240825_direct_control",
)
MODES = ("fresh_new_case", "known_topology_new_physics")


class OrchestrationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol", "publication_protocol", "binding", "frozen_artifact_root",
        "dataset_root", "manifest", "full_fields", "run_dir",
        "native_padding_result", "query_padding_result", "work_root", "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--phase", choices=("S1", "S2", "S3"), required=True)
    parser.add_argument("--s1-result", type=Path)
    parser.add_argument("--s2-result", type=Path)
    return parser.parse_args()


def load_full(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as archive:
        return {
            "coords": np.asarray(archive["shared/coords_m"], dtype=np.float64),
            "cv": np.asarray(archive["shared/control_volume_m3"], dtype=np.float64),
            "layer": np.asarray(archive["shared/layer_id"], dtype=np.int32),
        }


def build_cases(args: argparse.Namespace, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {str(row["sample_id"]): row for row in manifest["samples"]}
    full = load_full(args.full_fields)
    cases = []
    for registered in protocol["geometries"]:
        base = load_input_only_example(args.dataset_root, by_id[registered["sample_id"]])
        if int(base.meta["source_region_count"]) != int(registered["source_count"]):
            raise OrchestrationError("frozen geometry stratum drift")
        anchor_indices, distance = highn._anchor_indices(base, full["coords"], 1e-12)
        if distance > 1e-12:
            raise OrchestrationError("anchor coordinate drift")
        _, full_k, full_q, _ = highn._physics_fields(base, full)
        cases.append(make_case(
            base=base, full=full, full_k=full_k, full_q=full_q,
            sweep="identity_null", quantile=None, alpha=1.0, blend_fraction=0.5,
            protocol=protocol, anchor_indices=anchor_indices,
        ))
        quantiles = protocol["sweeps"]["K_only"]["k_target_quantiles"]
        alphas = protocol["sweeps"]["K_plus_Q_scale"]["q_multipliers"]
        for index, quantile in enumerate(quantiles):
            cases.append(make_case(
                base=base, full=full, full_k=full_k, full_q=full_q,
                sweep="K_only", quantile=float(quantile), alpha=1.0,
                blend_fraction=float(protocol["sweeps"]["K_only"]["base_to_target_blend"]),
                protocol=protocol, anchor_indices=anchor_indices,
            ))
            cases.append(make_case(
                base=base, full=full, full_k=full_k, full_q=full_q,
                sweep="K_plus_Q_scale", quantile=float(quantile), alpha=float(alphas[index]),
                blend_fraction=float(protocol["sweeps"]["K_plus_Q_scale"]["base_to_target_blend"]),
                protocol=protocol, anchor_indices=anchor_indices,
            ))
    return cases


def groups(cases: list[dict[str, Any]], phase: str) -> list[list[dict[str, Any]]]:
    if phase == "S1":
        return [[row for row in cases if row["sweep"] == "identity_null"]]
    if phase == "S2":
        return [[row for row in cases if row["sweep"] == name and row["quantile"] == 0.2]
                for name in ("K_only", "K_plus_Q_scale")]
    return [[row for row in cases if row["sweep"] == name and row["quantile"] == quantile]
            for name in ("K_only", "K_plus_Q_scale") for quantile in (0.2, 0.4, 0.6, 0.8)]


def write_plan(root: Path, cases: list[dict[str, Any]]) -> Path:
    rows = []
    for case in cases:
        sample_id = case["base_sample_id"]
        arrays = root / "inputs" / f"{sample_id}_anchor.npz"
        physics = root / "inputs" / f"{sample_id}_full.npz"
        arrays.parent.mkdir(parents=True, exist_ok=True)
        with arrays.open("wb") as handle:
            np.savez_compressed(handle, anchor_k=case["anchor_k"], anchor_q=case["anchor_q"])
        with physics.open("wb") as handle:
            np.savez_compressed(handle, k_xyz=case["full_k"], q_W_m3=case["full_q"])
        rows.append({
            "case_id": case["case_id"], "base_sample_id": sample_id,
            "sweep": case["sweep"], "quantile": case["quantile"], "alpha": case["alpha"],
            "dynamic_arrays": str(arrays), "dynamic_arrays_sha256": sha256(arrays),
            "full_physics": str(physics), "full_physics_sha256": sha256(physics),
            "total_power_W": case["total_power_W"],
            "dynamic_physics_sha256": case["dynamic_physics_sha256"],
        })
    plan = root / "input_plan.json"
    plan.write_text(json.dumps({
        "schema_version": "heat3d_v6_supplemental_input_plan_v1",
        "status": "frozen_before_runner_execution", "cases": rows,
    }, indent=2, sort_keys=True) + "\n")
    return plan


def command(args: argparse.Namespace, mode: str, plan: Path, output: Path, prediction: Path) -> list[str]:
    common = [
        sys.executable,
        str(ROOT / "scripts" / (
            "benchmark_heat3d_v6_p1i_final_e_service.py" if args.route.startswith("E")
            else "benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py")),
        "--protocol", str(args.publication_protocol), "--binding", str(args.binding),
        "--artifact-root", str(args.frozen_artifact_root),
        "--dataset-root", str(args.dataset_root), "--manifest", str(args.manifest),
        "--full-fields", str(args.full_fields), "--run-dir", str(args.run_dir),
        "--native-padding-result", str(args.native_padding_result),
        "--query-padding-result", str(args.query_padding_result),
        "--checkpoint-sha256", args.checkpoint_sha256,
        "--sample-count", "4", "--order-seed", "20260825",
        "--supplemental-input-plan", str(plan), "--output", str(output),
    ]
    if mode == "known_topology_new_physics":
        common.append("--supplemental-static-reuse")
    if args.route.startswith("E"):
        common += ["--route", args.route, "--service-mode", "serial",
                   "--resident-repeats", "1", "--cache-hot-repeats", "1"]
    else:
        common += [
            "--resolution", "16384" if "16384" in args.route else "240825",
            "--asymmetric-mode", "u_v2", "--timing-only", "--timing-regression-audit",
            "--repeats", "1", "--prediction-output", str(prediction),
        ]
    return common


def run_one(args: argparse.Namespace, mode: str, plan: Path, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{mode}.json"
    prediction = root / f"{mode}.predictions.npz"
    cmd = command(args, mode, plan, output, prediction)
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (root / f"{mode}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise OrchestrationError(f"{args.route}/{mode} runner failed; see {root}/{mode}.log")
    result = json.loads(output.read_text(encoding="utf-8"))
    if args.route.startswith("E"):
        capture = result["supplemental_capture"]
        with np.load(capture["prediction_path"], allow_pickle=False) as payload:
            ids = [str(value) for value in payload["sample_ids"].tolist()]
            predictions = np.asarray(payload["full_deltaT_K"], dtype=np.float32)
        rows = {row_id: dict(row, prediction=predictions[index])
                for index, (row_id, row) in enumerate(zip(ids, capture["rows"], strict=True))}
    else:
        with np.load(prediction, allow_pickle=False) as payload:
            ids = [str(value) for value in payload["sample_ids"].tolist()]
            predictions = np.asarray(payload["full_deltaT_K"], dtype=np.float32)
        sample_rows = {row["sample_id"]: row for row in result["samples"]}
        rows = {row_id: {
            "prepared_payload_sha256": sample_rows[row_id]["prepared_payload_sha256"],
            "static_artifacts": sample_rows[row_id]["supplemental_static_artifacts"],
            "elapsed_seconds": sample_rows[row_id]["stages"]["matched_continuous_e2e"],
            "stages": sample_rows[row_id]["stages"], "prediction": predictions[index],
        } for index, row_id in enumerate(ids)}
    return {"rows": rows, "runner_output": str(output), "command": cmd}


def validate_prior(path: Path | None, phase: str, route: str, commit: str) -> dict[str, Any]:
    if path is None:
        raise OrchestrationError(f"{phase} result required")
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("phase") != phase or row.get("status") != "passed" or row.get("route") != route:
        raise OrchestrationError(f"{phase} gate is not PASS")
    if row.get("execution_commit") != commit:
        raise OrchestrationError(f"{phase} execution commit drift")
    return row


def timing_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "median_s": float(np.median(array)),
        "p95_s": float(np.quantile(array, 0.95)),
        "throughput_samples_per_second": float(array.size / np.sum(array)),
    }


def main() -> int:
    args = parse()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "s0_passed_ready_for_gpu":
        raise OrchestrationError("S0 is not PASS")
    if sha256(args.manifest) != protocol["dataset"]["manifest_sha256"]:
        raise OrchestrationError("manifest drift")
    if sha256(args.full_fields) != protocol["dataset"]["full_field_archive_sha256"]:
        raise OrchestrationError("full-field drift")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    s1 = validate_prior(args.s1_result, "S1", args.route, commit) if args.phase in {"S2", "S3"} else None
    if args.phase == "S3":
        validate_prior(args.s2_result, "S2", args.route, commit)
    cases = build_cases(args, protocol)
    same_shape_floor = float(
        protocol["numerical_equivalence"]["same_shape_floor_K_by_route"][args.route])
    numerical_limit = max(1e-3, 20.0 * same_shape_floor)
    all_rows, gate_rows = [], []
    for index, group in enumerate(groups(cases, args.phase)):
        root = args.work_root / args.phase / args.route / f"group_{index}"
        plan = write_plan(root, group)
        fresh = run_one(args, "fresh_new_case", plan, root / "fresh")
        known = run_one(args, "known_topology_new_physics", plan, root / "known")
        for case in group:
            sample_id = case["base_sample_id"]
            left, right = fresh["rows"][sample_id], known["rows"][sample_id]
            delta = np.asarray(left["prediction"], dtype=np.float64) - np.asarray(right["prediction"], dtype=np.float64)
            maximum = float(np.max(np.abs(delta)))
            static_exact = left["static_artifacts"] == right["static_artifacts"]
            payload_exact = left["prepared_payload_sha256"] == right["prepared_payload_sha256"]
            dynamic_changed = True
            if args.phase == "S2":
                identity = {row["sample_id"]: row for row in s1["gate_rows"]}[sample_id]
                dynamic_changed = left["prepared_payload_sha256"] != identity["fresh_prepared_payload_sha256"]
            passed = static_exact and payload_exact and maximum <= numerical_limit and dynamic_changed
            gate_rows.append({
                "case_id": case["case_id"], "sample_id": sample_id,
                "sweep": case["sweep"], "static_artifacts_exact": static_exact,
                "prepared_payload_exact": payload_exact,
                "fresh_prepared_payload_sha256": left["prepared_payload_sha256"],
                "known_prepared_payload_sha256": right["prepared_payload_sha256"],
                "dynamic_payload_changed_from_identity": dynamic_changed,
                "prediction_max_abs_K": maximum,
                "prediction_rmse_K": float(np.sqrt(np.mean(delta * delta))),
                "same_shape_floor_K": same_shape_floor,
                "numerical_limit_K": numerical_limit, "passed": passed,
            })
            if not passed:
                raise OrchestrationError(f"{args.phase}/{args.route}/{case['case_id']} failed")
            for mode, source in (("fresh_new_case", left), ("known_topology_new_physics", right)):
                all_rows.append({
                    "case_id": case["case_id"], "sample_id": sample_id,
                    "sweep": case["sweep"], "mode": mode,
                    "elapsed_seconds": float(source["elapsed_seconds"]),
                    "stages": source["stages"],
                })
    summary = []
    if args.phase == "S3":
        for sweep in ("K_only", "K_plus_Q_scale"):
            rows_by_mode = {}
            for mode in MODES:
                selected = [row for row in all_rows if row["sweep"] == sweep and row["mode"] == mode]
                timing = timing_stats([row["elapsed_seconds"] for row in selected])
                stage_names = sorted({key for row in selected for key in row["stages"]})
                item = {
                    "sweep": sweep, "mode": mode, "timing": timing,
                    "stage_median_seconds": {
                        key: float(np.median([row["stages"].get(key, 0.0) for row in selected]))
                        for key in stage_names
                    },
                }
                summary.append(item); rows_by_mode[mode] = item
            rows_by_mode["known_topology_new_physics"]["speedup_vs_fresh_median"] = float(
                rows_by_mode["fresh_new_case"]["timing"]["median_s"]
                / rows_by_mode["known_topology_new_physics"]["timing"]["median_s"])
    result = {
        "schema_version": "heat3d_v6_supplemental_known_topology_result_v1",
        "status": "passed", "phase": args.phase, "route": args.route,
        "execution_commit": commit, "protocol_sha256": sha256(args.protocol),
        "gate_rows": gate_rows, "timing_rows": all_rows if args.phase == "S3" else [],
        "summary": summary,
        "guardrails": {"training": False, "FVM": False, "valid": False,
                       "test": False, "sealed": False, "temperature_labels": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "phase": args.phase, "route": args.route}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
