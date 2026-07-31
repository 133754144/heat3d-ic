#!/usr/bin/env python3
"""Deterministic closeout checker for V6-P1i requalification."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import py_compile
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/heat3d_v6_p1i"
FORMAL_PREFIX = "v6_p1i_formal1024_v1"
FORMAL_DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"

V0_FROZEN_SHA256 = {
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_acceptance.json":
        "6707e2aa24e0c366f5547aed90e68261de4bfb9709a265727d826f398ecae394",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0.yaml":
        "a2bc00e4618786914fb72e7341fd05808210f0be76180261ff09e77e976eddf3",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0_artifact_manifest.json":
        "f4c393a00896eea0bbcd061d3a06d90618095a38effb1039212280ffe7c0815f",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0_closeout.json":
        "6812dbf281a9e2ea541b7ed363c8b9547f7096a7d3fb18a61062032d24c25356",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0_distribution_audit.json":
        "83b13c4eee334e28750fcd79e3d157feb661554cdd9450fa5aeeaeac617a5e2a",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0_freeze_manifest.json":
        "fe35cb90b08b4f5fe9888e32d1b7dae0bc039211beb3c129cea5a553ddf1307a",
    "configs/heat3d_v6_p1i/v6_p1i_formal1024_v0_manifest.json":
        "6fb5971e54834f3c48fa84c9eab8b0386e36baf0d86cae8afa60bb1d9a2ed520",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _check_v0_immutable() -> None:
    for relative, expected in V0_FROZEN_SHA256.items():
        path = ROOT / relative
        _assert(path.is_file(), f"missing formal1024_v0 artifact: {relative}")
        _assert(_sha256(path) == expected, f"formal1024_v0 SHA drift: {relative}")
    audit = _load_json(
        CONFIG / "v6_p1i_formal1024_v0_distribution_audit.json"
    )
    closeout = _load_json(CONFIG / "v6_p1i_formal1024_v0_closeout.json")
    _assert(audit["status"] == "failed", "formal1024_v0 audit must remain failed")
    _assert(
        closeout["status"] == "qualification_failed"
        and not closeout["decision"]["formal_dataset_qualified"],
        "formal1024_v0 lifecycle must remain permanent qualification failure",
    )


def _check_freeze(path: Path) -> None:
    payload = _load_json(path)
    _assert(
        payload["status"] == "frozen_before_generation",
        f"bad freeze status: {path.name}",
    )
    for row in payload["artifacts"]:
        artifact = ROOT / row["path"]
        _assert(artifact.is_file(), f"missing frozen artifact: {row['path']}")
        _assert(
            _sha256(artifact) == row["sha256"],
            f"frozen SHA drift: {row['path']}",
        )


def _check_attempt_lifecycle() -> None:
    attempts_path = CONFIG / "v6_p1i_generation_attempts.csv"
    lines = attempts_path.read_text(encoding="utf-8").splitlines(keepends=True)
    v0_stop = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("formal1024_v0,")
    )
    historical_prefix = "".join(lines[: v0_stop + 1]).encode("utf-8")
    _assert(
        hashlib.sha256(historical_prefix).hexdigest()
        == "9b63c1009e666d81ac3d2928b1dd42aaa712c0e6791ae4605e964311585adbad",
        "formal1024_v0 attempt-registry prefix drift",
    )
    with attempts_path.open(encoding="utf-8", newline="") as handle:
        rows = {row["attempt_id"]: row for row in csv.DictReader(handle)}
    for version in range(3, 13):
        _assert(
            rows[f"pilot128_v{version}"]["formal1024_allowed"] == "false",
            f"failed pilot128_v{version} must not authorize formal generation",
        )
    _assert(
        rows["pilot128_v13"]["status"] == "aborted_engineering_id_collision",
        "pilot128_v13 lifecycle",
    )
    _assert(
        rows["pilot128_v14"]["status"]
        == "generated_complete_qualification_passed",
        "pilot128_v14 lifecycle",
    )
    _assert(
        rows["formal1024_v1_input_preflight_0"]["status"]
        == "failed_input_power_floor",
        "formal input preflight failure must be retained",
    )
    _assert(
        rows["pilot128_v15"]["status"]
        == "generated_complete_qualification_passed",
        "pilot128_v15 lifecycle",
    )
    _assert(
        rows["formal1024_v1"]["status"]
        == "generated_complete_qualification_passed"
        and rows["formal1024_v1"]["formal1024_allowed"] == "true",
        "formal1024_v1 lifecycle",
    )


def _check_parent_and_formal_config() -> None:
    pilot = yaml.safe_load(
        (CONFIG / "v6_p1i_pilot128_v15.yaml").read_text(encoding="utf-8")
    )
    formal = yaml.safe_load(
        (CONFIG / "v6_p1i_formal1024_v1.yaml").read_text(encoding="utf-8")
    )
    _assert(
        formal["provenance"]["parent_attempt"]
        == "heat3d_v6_p1i_continuous_physics128_v15",
        "formal parent",
    )
    _assert(
        formal["sample_count"] == 1024
        and formal["split_counts"]
        == {"train": 768, "valid_iid": 128, "test_iid": 128},
        "formal counts",
    )
    _assert(
        formal["sample_id_prefix"] == "v6p1if1_"
        and formal["sampling"]["seed"] == 612819,
        "formal namespace/seed",
    )
    for key in ("physics", "projection", "guardrails"):
        _assert(formal[key] == pilot[key], f"formal/pilot scientific drift: {key}")
    pilot_sampling = dict(pilot["sampling"])
    formal_sampling = dict(formal["sampling"])
    pilot_sampling.pop("seed")
    formal_sampling.pop("seed")
    _assert(
        formal_sampling == pilot_sampling,
        "formal/pilot global sampling-rule drift",
    )


def _check_formal_results() -> None:
    audit = _load_json(CONFIG / f"{FORMAL_PREFIX}_distribution_audit.json")
    closeout = _load_json(CONFIG / f"{FORMAL_PREFIX}_closeout.json")
    manifest = _load_json(CONFIG / f"{FORMAL_PREFIX}_manifest.json")
    split = _load_json(CONFIG / f"{FORMAL_PREFIX}_split_manifest.json")
    preflight = _load_json(CONFIG / f"{FORMAL_PREFIX}_preflight.json")
    freeze = _load_json(CONFIG / f"{FORMAL_PREFIX}_freeze_manifest.json")
    _assert(_all_finite(audit), "formal audit contains non-finite JSON values")
    _assert(
        audit["status"] == "passed" and all(audit["checks"].values()),
        "formal audit gates",
    )
    _assert(
        closeout["status"] == "generated_complete_qualification_passed"
        and closeout["decision"]["formal_dataset_qualified"],
        "formal closeout",
    )
    _assert(
        not closeout["decision"]["training_allowed_by_this_closeout"],
        "qualification closeout must not authorize training",
    )
    _assert(
        preflight["status"] == "passed"
        and all(preflight["checks"].values()),
        "formal preflight",
    )
    _assert(
        split["target_values_used"] is False
        and split["solver_results_used"] is False
        and split["model_error_used"] is False,
        "target-independent split",
    )
    _assert(
        freeze["guardrails"]["solver_runs_before_freeze"] == 0,
        "solver ran before freeze",
    )
    _assert(
        manifest["dataset_id"] == FORMAL_DATASET_ID
        and manifest["sample_count"] == 1024
        and manifest["split_role_counts"]
        == {"train": 768, "valid_iid": 128, "test_iid": 128},
        "formal manifest identity/counts",
    )
    payload = dict(manifest)
    expected_payload_sha = payload.pop("manifest_payload_sha256")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    _assert(
        hashlib.sha256(canonical).hexdigest() == expected_payload_sha,
        "formal manifest payload SHA",
    )
    dataset_root = ROOT / manifest["dataset_root"]
    _assert(dataset_root.is_dir(), "formal dataset root")
    _assert(len(manifest["samples"]) == 1024, "formal manifest sample rows")
    seen: set[str] = set()
    roles: Counter[str] = Counter()
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        _assert(sample_id.startswith("v6p1if1_"), "formal sample namespace")
        _assert(sample_id not in seen, "duplicate formal sample ID")
        seen.add(sample_id)
        roles[row["split_role"]] += 1
        sample_dir = dataset_root / row["relative_path"]
        for filename, expected_sha in row["file_sha256"].items():
            path = sample_dir / filename
            _assert(path.is_file(), f"missing {sample_id}/{filename}")
            _assert(_sha256(path) == expected_sha, f"SHA drift {sample_id}/{filename}")
        coords = np.load(sample_dir / "coords.npy", allow_pickle=False)
        k_field = np.load(sample_dir / "k_field.npy", allow_pickle=False)
        q_field = np.load(sample_dir / "q_field.npy", allow_pickle=False)
        temperature = np.load(
            sample_dir / "temperature.npy", allow_pickle=False
        )
        _assert(coords.shape == (1024, 3), "coords shape")
        _assert(k_field.shape == (1024, 3), "k shape")
        _assert(q_field.shape == (1024, 1), "q shape")
        _assert(temperature.shape == (1024,), "temperature shape")
        _assert(
            all(
                np.all(np.isfinite(array))
                for array in (coords, k_field, q_field, temperature)
            ),
            "non-finite formal arrays",
        )
    _assert(
        roles == Counter({"train": 768, "valid_iid": 128, "test_iid": 128}),
        "formal realized split counts",
    )
    artifact_manifest_path = CONFIG / f"{FORMAL_PREFIX}_artifact_manifest.json"
    artifact_manifest = _load_json(artifact_manifest_path)
    _assert(
        artifact_manifest["status"]
        == "generated_complete_qualification_passed",
        "artifact-manifest lifecycle",
    )
    _assert(
        artifact_manifest["dataset_size_bytes"]
        == sum(
            path.stat().st_size
            for path in dataset_root.rglob("*")
            if path.is_file()
        ),
        "formal dataset logical size",
    )
    for row in artifact_manifest["artifacts"]:
        path = ROOT / row["path"]
        _assert(path.is_file(), f"missing closeout artifact: {row['path']}")
        _assert(
            _sha256(path) == row["sha256"],
            f"closeout artifact SHA drift: {row['path']}",
        )
    requalification = _load_json(CONFIG / "v6_p1i_requalification_manifest.json")
    _assert(requalification["status"] == "complete", "requalification status")
    _assert(
        requalification["formal1024_v1"]["artifact_manifest_sha256"]
        == _sha256(artifact_manifest_path),
        "requalification/artifact-manifest binding",
    )
    _assert(
        requalification["guardrails"]
        == {
            "training_runs": 0,
            "model_inference_runs": 0,
            "formal1024_v0_modified": False,
            "v6_or_p1h_modified": False,
            "per_sample_Rth_backsolve": False,
            "post_solve_filtering_or_replacement": False,
        },
        "requalification guardrails",
    )


def main() -> int:
    _check_v0_immutable()
    _check_attempt_lifecycle()
    _check_parent_and_formal_config()
    _check_freeze(CONFIG / "v6_p1i_pilot128_v15_freeze_manifest.json")
    _check_freeze(CONFIG / "v6_p1i_formal1024_v1_freeze_manifest.json")
    _check_formal_results()
    for path in (
        ROOT / "scripts/audit_heat3d_v6_p1i_postmortem.py",
        ROOT / "scripts/heat3d_v6_p1i_split.py",
        ROOT / "scripts/generate_heat3d_v6_p1i_v15.py",
        ROOT / "scripts/prepare_heat3d_v6_p1i_formal_v1.py",
        ROOT / "scripts/audit_heat3d_v6_p1i_formal_v1.py",
        Path(__file__),
    ):
        py_compile.compile(str(path), doraise=True)
    print(
        json.dumps(
            {
                "status": "passed",
                "formal1024_v0": "permanent_failed_qualification_unchanged",
                "pilot128_v15": "passed",
                "formal1024_v1": "generated_complete_qualification_passed",
                "sample_count": 1024,
                "training_runs": 0,
                "model_inference_runs": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
