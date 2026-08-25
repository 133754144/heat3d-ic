#!/usr/bin/env python3
"""Fail-closed merge-readiness checker for the V6 supplemental publication bundle."""

from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import os


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/heat3d_v6_supplemental_publication"
RESULTS = BASE / "known_topology_results_8a81261"
PROTOCOL = BASE / "known_topology_new_physics_protocol.json"
CLOSEOUT = BASE / "known_topology_publication_closeout.json"
CSV_PATH = BASE / "known_topology_publication_results.csv"
ENVIRONMENT = BASE / "known_topology_environment_provenance.json"
MANIFEST = BASE / "integration_manifest.json"
RECEIPT = BASE / "integration_receipt.json"
BASE_MAIN = "6922e80c392385a8ae3d09b720c5307aaee1fffd"
EXECUTION = "8a812619ab0112b4ecfc37ef18189f731180059d"
FROZEN = "04dc85c6ec1b620f026ea546f28a045cd43bbc9c"
INVALIDATED = "171cd5e468fdae3ee599eac11bc1508097f7dd7e"
ROUTES = (
    "E16384_reconstruction",
    "U_v2_16384_reconstruction",
    "U_v2_direct240825",
    "E240825_direct_control",
)
PHASE_COUNTS = {"S1": 4, "S2": 8, "S3": 32}
GUARDRAILS = (
    "training", "FVM", "valid_access", "test_access", "sealed_access",
    "temperature_labels", "resident_optimization", "accuracy_selection",
)
ARTIFACT_GUARDRAILS = ("training", "FVM", "valid", "test", "sealed", "temperature_labels")
E_RUNNER = "scripts/benchmark_heat3d_v6_p1i_final_e_service.py"
U_RUNNER = "scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py"
FROZEN_RUNNER_SHA = {
    E_RUNNER: "ef8087d3ffe19d4d3d044097baa14d5de39f029652cab8a4064105d53695f326",
    U_RUNNER: "ec916aa2e4bf37cbc2fb27d6a862f610b185e9b918781ae82b3c6c4e5fb6a834",
}
CURRENT_RUNNER_DIFF_SHA = {
    E_RUNNER: "184f376b0a381578cbeb02662dd5bc765c017002d913efd1db3d95e6b333d49c",
    U_RUNNER: "3f51ee770f7f617e0e866ce3a94a2fa1919e33c724cddcb4ed9b7507ffaf6499",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_env() -> dict[str, str]:
    env = dict(os.environ)
    dotgit = ROOT / ".git"
    if dotgit.is_file():
        text = dotgit.read_text(encoding="utf-8").strip()
        require(text.startswith("gitdir: "), "malformed .git pointer")
        env["GIT_DIR"] = text.split(": ", 1)[1]
        env["GIT_WORK_TREE"] = str(ROOT)
    return env


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, env=git_env(), text=True).strip()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, env=git_env()
    )


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def assert_close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {actual} != {expected}")


def parser_defaults_are_disabled(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags: dict[str, dict[str, object]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if name not in {"--supplemental-input-plan", "--supplemental-static-reuse"}:
            continue
        flags[str(name)] = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords if keyword.arg and isinstance(keyword.value, ast.Constant)
        }
    require(set(flags) == {"--supplemental-input-plan", "--supplemental-static-reuse"},
            f"supplemental flags absent: {path}")
    require(flags["--supplemental-input-plan"].get("default") in {None},
            f"input-plan default enabled: {path}")
    require(flags["--supplemental-static-reuse"].get("action") == "store_true",
            f"static-reuse default enabled: {path}")
    text = path.read_text(encoding="utf-8")
    require("supplemental_plan = None" in text, f"supplemental plan is not disabled: {path}")
    require("if args.supplemental_input_plan is not None:" in text,
            f"supplemental plan lacks explicit guard: {path}")


def validate_runner_compatibility(protocol: dict[str, object]) -> None:
    for path, frozen_sha in FROZEN_RUNNER_SHA.items():
        frozen = git_bytes(FROZEN, path)
        require(sha256_bytes(frozen) == frozen_sha, f"frozen runner SHA drift: {path}")
        current = (ROOT / path).read_bytes()
        diff = "".join(difflib.unified_diff(
            frozen.decode().splitlines(keepends=True),
            current.decode().splitlines(keepends=True),
            fromfile=f"frozen/{path}", tofile=f"current/{path}", n=3,
        )).encode()
        require(sha256_bytes(diff) == CURRENT_RUNNER_DIFF_SHA[path],
                f"supplemental runner diff drift: {path}")
        require(sha256_bytes(current) == protocol["S0"]["execution_files_sha256"][path],
                f"execution runner SHA drift: {path}")
        parser_defaults_are_disabled(ROOT / path)
    s0 = subprocess.run(
        ["python3", str(ROOT / "scripts/check_heat3d_v6_supplemental_known_topology_s0.py")],
        cwd=ROOT, env=git_env(), check=True, capture_output=True, text=True,
    )
    payload = json.loads(s0.stdout.splitlines()[-1])
    require(payload["S0"] == "PASS" and payload["numeric_primitives_ast_exact"] is True,
            "S0 numerical compatibility failed")
    require(payload["shared_code_exact"] is True, "S0 shared-code compatibility failed")


def validate_artifacts(protocol: dict[str, object], closeout: dict[str, object]) -> list[dict[str, object]]:
    require(closeout["execution_commit"] == EXECUTION, "execution commit drift")
    require(protocol["frozen_publication_runtime_protocol"]["sha256"] ==
            "325dd80dffadae0f56c547ec84902a717a59615c9d32bac2036d121bae17790b",
            "publication protocol SHA drift")
    all_timing: list[dict[str, object]] = []
    for phase, expected_count in PHASE_COUNTS.items():
        for route in ROUTES:
            path = RESULTS / phase / f"{route}.json"
            require(sha256(path) == closeout["result_artifact_sha256"][phase][route],
                    f"artifact SHA drift: {phase}/{route}")
            data = json.loads(path.read_text(encoding="utf-8"))
            require("171cd5e" not in path.read_text(encoding="utf-8"),
                    f"invalidated attempt referenced by result artifact: {phase}/{route}")
            require(data["execution_commit"] == EXECUTION, f"execution drift: {phase}/{route}")
            require(data["status"] == "passed" and data["phase"] == phase and data["route"] == route,
                    f"artifact identity/status drift: {phase}/{route}")
            require(len(data["gate_rows"]) == expected_count, f"case count drift: {phase}/{route}")
            require(set(data["guardrails"]) == set(ARTIFACT_GUARDRAILS) and
                    all(data["guardrails"].get(key) is False for key in ARTIFACT_GUARDRAILS),
                    f"guardrail drift: {phase}/{route}")
            for row in data["gate_rows"]:
                require(row["static_artifacts_exact"] is True, f"static exactness failed: {phase}/{route}")
                require(row["prepared_payload_exact"] is True, f"payload exactness failed: {phase}/{route}")
                require(row["passed"] is True, f"numerical gate failed: {phase}/{route}")
                require(float(row["prediction_max_abs_K"]) <= float(row["numerical_limit_K"]),
                        f"numerical limit exceeded: {phase}/{route}")
            if phase != "S3":
                require(data["timing_rows"] == [] and data["summary"] == [],
                        f"unexpected timing population: {phase}/{route}")
                continue
            require(len(data["timing_rows"]) == 64 and len(data["summary"]) == 4,
                    f"S3 timing population drift: {route}")
            all_timing.extend(data["timing_rows"])
            for summary in data["summary"]:
                rows = [row for row in data["timing_rows"]
                        if row["sweep"] == summary["sweep"] and row["mode"] == summary["mode"]]
                require(len(rows) == 16, f"summary population drift: {route}/{summary['sweep']}/{summary['mode']}")
                values = [float(row["elapsed_seconds"]) for row in rows]
                timing = summary["timing"]
                require(int(timing["count"]) == 16, "summary count drift")
                assert_close(statistics.median(values), float(timing["median_s"]), "median")
                assert_close(quantile(values, 0.95), float(timing["p95_s"]), "p95")
                assert_close(len(values) / sum(values), float(timing["throughput_samples_per_second"]), "throughput")
            for sweep in ("K_only", "K_plus_Q_scale"):
                fresh = next(row for row in data["summary"] if row["sweep"] == sweep and row["mode"] == "fresh_new_case")
                known = next(row for row in data["summary"] if row["sweep"] == sweep and row["mode"] == "known_topology_new_physics")
                assert_close(
                    float(fresh["timing"]["median_s"]) / float(known["timing"]["median_s"]),
                    float(known["speedup_vs_fresh_median"]), "speedup",
                )
    require(len(all_timing) == 256, "four-route S3 timing row population drift")
    return all_timing


def validate_closeout_table(closeout: dict[str, object]) -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 16 and len(closeout["result_table"]) == 16,
            "closeout table population drift")
    keyed_json = {(row["route"], row["sweep"], row["mode"]): row for row in closeout["result_table"]}
    for row in csv_rows:
        key = (row["route"], row["sweep"], row["mode"])
        require(key in keyed_json, f"CSV key absent from closeout: {key}")
        reference = keyed_json[key]
        require(int(row["count"]) == int(reference["count"]), f"CSV count drift: {key}")
        for field in ("median_seconds", "p95_seconds", "throughput_samples_per_second"):
            assert_close(float(row[field]), float(reference[field]), f"CSV {field}: {key}")
        expected_speedup = reference["speedup_vs_fresh_median"]
        if expected_speedup is None:
            require(row["speedup_vs_fresh_median"] == "", f"fresh speedup must be empty: {key}")
        else:
            assert_close(float(row["speedup_vs_fresh_median"]), float(expected_speedup), f"CSV speedup: {key}")


def validate_environment() -> dict[str, object]:
    data = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    require(data["status"] == "complete_from_existing_artifacts_no_inference", "environment status")
    require(data["execution_commit"] == EXECUTION, "environment execution commit")
    require(data["machine"]["role"] == "devbox" and data["machine"]["hostname"], "machine identity")
    require(data["machine"]["cpu"] != "N/A" and data["machine"]["gpu"] != "N/A", "hardware evidence incomplete")
    require(data["software"]["backend"] == "CUDA", "backend evidence drift")
    scope = data["evidence_scope"]
    require(scope["runner_json_count"] == 64 and scope["runner_log_count"] == 64, "environment source count")
    require(scope["input_plan_count"] == 32 and scope["gpu_runtime_record_count"] == 64,
            "environment execution evidence population")
    require(scope["gpu_inference_executed_by_collector"] is False and
            scope["physics_experiment_executed_by_collector"] is False,
            "provenance collector exceeded scope")
    require(data["execution_embedded_contract"]["independent_process_count"] == 64,
            "independent process evidence drift")
    require(len(data["source_artifacts"]) == 160, "environment source manifest population")
    return data


def validate_claim_scope() -> None:
    text = (ROOT / "docs/v6_supplemental_known_topology_publication_closeout.md").read_text(encoding="utf-8")
    required = (
        "runtime-only", "four frozen train-only geometries", "new k/q",
        "no accuracy claim", "no labels/valid/test/sealed", "single randomized workload order",
        "known-topology/new-physics is not resident or cache-hot",
    )
    for phrase in required:
        require(phrase in text, f"claim-scope phrase absent: {phrase}")


def validate_integration_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    require(manifest["target_main_base"] == BASE_MAIN, "integration base drift")
    require(git("merge-base", "--is-ancestor", BASE_MAIN, "HEAD") == "", "HEAD not based on target main")
    content_commit = receipt["content_commit"]
    require(len(content_commit) == 40, "content commit not finalized")
    require(git("merge-base", "--is-ancestor", content_commit, "HEAD") == "", "content commit not ancestor of HEAD")
    actual = set(filter(None, git("diff", "--name-only", f"{BASE_MAIN}...{content_commit}").splitlines()))
    expected = set(manifest["strict_allowlist"])
    require(actual == expected, f"strict allowlist drift: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    forbidden = ("data/", "output/", "logs/", "checkpoints/")
    require(not any(path.startswith(forbidden) for path in actual), "forbidden artifact path in integration")
    require(receipt["merge_method_required"] == "merge_commit", "merge method contract drift")
    require(receipt["pr_number"] is None or int(receipt["pr_number"]) > 0, "PR receipt drift")
    return receipt


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    require(protocol["base_main_commit"] == BASE_MAIN, "protocol base drift")
    require(protocol["S0"]["status"] == "PASS", "S0 status drift")
    require(protocol["invalidated_attempt"]["head"] == INVALIDATED, "invalidated head drift")
    require(protocol["invalidated_attempt"]["reuse_of_timing_speedup_setup_break_even"] is False,
            "invalidated results reused")
    require(all(protocol["guardrails"].get(key) is False for key in GUARDRAILS), "protocol guardrail drift")
    require(all(closeout["guardrails"].get(key) is False for key in GUARDRAILS), "closeout guardrail drift")
    validate_runner_compatibility(protocol)
    validate_artifacts(protocol, closeout)
    validate_closeout_table(closeout)
    environment = validate_environment()
    require(closeout["environment_provenance"]["path"] == str(ENVIRONMENT.relative_to(ROOT)),
            "closeout environment path drift")
    require(closeout["environment_provenance"]["sha256"] == sha256(ENVIRONMENT),
            "closeout environment SHA drift")
    claim = closeout["claim_scope"]
    require(claim == {
        "runtime_only": True,
        "geometry_population": "four_train_only_fixed_geometries",
        "physics_population": "new_k_or_new_k_plus_positive_q_scale",
        "accuracy_claim": False,
        "labels_or_valid_test_sealed_access": False,
        "randomized_workload_order_count": 1,
        "known_topology_semantics": "new_physics_not_resident_not_cache_hot",
    }, "machine-readable claim scope drift")
    validate_claim_scope()
    receipt = validate_integration_manifest()
    print(json.dumps({
        "status": "PASS",
        "execution_commit": EXECUTION,
        "S0": "PASS", "S1": "PASS", "S2": "PASS", "S3": "PASS",
        "s3_timing_rows": 256,
        "environment_artifact_set_sha256": environment["source_artifact_set_sha256"],
        "strict_allowlist": "PASS",
        "supplemental_flags_off_compatibility": "PASS",
        "receipt_status": receipt["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
