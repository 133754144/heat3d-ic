#!/usr/bin/env python3
"""Deterministic checker for the frozen V6-P1i training preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PREFIX = "v6_p1i_formal1024_v1"
DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
CONFIG = Path("configs/heat3d_v6_p1i")
FROZEN = {
    CONFIG / f"{PREFIX}.yaml":
        "1e15a77fe51eea7ec64614566bb6bb12bfcf05948f3b7c8c6f3c85ec759a58f8",
    CONFIG / f"{PREFIX}_manifest.json":
        "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514",
    CONFIG / f"{PREFIX}_samples.csv":
        "5f0305e5994c8a8ec43a31ee9844526f5fad262a46c73a730cdf4285cd2d4018",
    CONFIG / f"{PREFIX}_input_definitions.csv":
        "afea524f5002814a4ec2e8dbac0bc6d2c60f18de9e2cf16c80c1ffe63feac498",
    CONFIG / f"{PREFIX}_regions.csv":
        "bb2fa33b822e4366f106488d24fcf60d7357bd46c4b196572ae50e66e7820acf",
    CONFIG / "v6_p1i_background_k_contract.csv":
        "a0f504ceea7aa9b0a7ac1eea70a7be31a849a561e114d2fe8e5859c8cadbf703",
}
REQUIRED_OUTPUTS = (
    CONFIG / f"{PREFIX}_training_preflight_audit.json",
    CONFIG / f"{PREFIX}_training_authorization.json",
    CONFIG / f"{PREFIX}_training_preflight_distribution_summary.csv",
    CONFIG / f"{PREFIX}_training_preflight_distribution_bins.csv",
    CONFIG / f"{PREFIX}_training_preflight_joint_coverage.csv",
    CONFIG / f"{PREFIX}_training_preflight_joint_occupancy.csv",
    CONFIG / f"{PREFIX}_training_preflight_split_summary.csv",
    CONFIG / f"{PREFIX}_training_preflight_split_comparison.csv",
    CONFIG / f"{PREFIX}_training_preflight_outputs_manifest.json",
    CONFIG / "v6_p1i_literature_registry_v1.json",
    CONFIG / "v6_p1i_literature_id_crosswalk.csv",
    CONFIG / "v6_p1i_cross_family_contact_evidence.json",
    CONFIG / f"{PREFIX}_archive_manifest.json",
    CONFIG / f"{PREFIX}_clean_checkout_replay.json",
    Path("docs") / f"{PREFIX}_training_preflight_audit.md",
    Path("docs") / f"{PREFIX}_training_preflight_distributions.png",
    Path("docs") / f"{PREFIX}_training_preflight_joint_coverage.png",
    Path("docs") / f"{PREFIX}_training_preflight_split_ecdf.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def canonical_tree_sha(manifest: dict[str, Any]) -> tuple[str, int]:
    rows = []
    for sample in manifest["samples"]:
        base = Path(sample["relative_path"])
        for filename, digest in sorted(sample["file_sha256"].items()):
            rows.append(
                {
                    "path": str(base / filename),
                    "sha256": digest,
                }
            )
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)


def check(repo_root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for relative, expected in FROZEN.items():
        path = repo_root / relative
        expect(path.is_file(), f"missing frozen input: {relative}")
        expect(sha256(path) == expected, f"frozen SHA drift: {relative}")
    checks["frozen_inputs_unchanged"] = True

    for relative in REQUIRED_OUTPUTS:
        expect((repo_root / relative).is_file(), f"missing output: {relative}")
    checks["required_outputs_present"] = True

    config = yaml.safe_load(
        (repo_root / CONFIG / f"{PREFIX}.yaml").read_text(encoding="utf-8")
    )
    expect(config["dataset_id"] == DATASET_ID, "dataset ID")
    expect(config["sample_count"] == 1024, "sample count")
    expect(
        config["split_counts"]
        == {"train": 768, "valid_iid": 128, "test_iid": 128},
        "split counts",
    )
    expect(
        config["physics"]["contact"]
        == {"type": "perfect", "R_contact_m2K_W": 0.0},
        "P1i perfect-contact contract",
    )
    checks["formal_config_contract"] = True

    manifest = load_json(repo_root / CONFIG / f"{PREFIX}_manifest.json")
    expect(manifest["sample_count"] == 1024, "manifest count")
    expect(
        manifest["manifest_payload_sha256"]
        == "27d2ea3b7ec4e4ce9c6d068471cd19036ac8148b6cd57da325219d718c7e5ed5",
        "manifest payload SHA",
    )
    tree_sha, file_count = canonical_tree_sha(manifest)
    checks["formal_manifest_binding"] = True

    samples = read_csv(repo_root / CONFIG / f"{PREFIX}_samples.csv")
    expect(len(samples) == 1024, "samples CSV count")
    roles = {role: 0 for role in ("train", "valid_iid", "test_iid")}
    for row in samples:
        roles[row["split_role"]] += 1
    expect(
        roles == {"train": 768, "valid_iid": 128, "test_iid": 128},
        "samples CSV split counts",
    )
    peak = np.asarray([float(row["peak_deltaT_K"]) for row in samples])
    mean = np.asarray([float(row["mean_deltaT_K"]) for row in samples])
    rms = np.asarray([float(row["cv_rms_deltaT_K"]) for row in samples])
    power = np.asarray([float(row["package_total_power_W"]) for row in samples])
    reff = peak / power
    expect(np.all(np.isfinite(np.column_stack((peak, mean, rms, reff)))), "finite")
    checks["source_table_finite"] = True

    audit = load_json(
        repo_root / CONFIG / f"{PREFIX}_training_preflight_audit.json"
    )
    expect(all_finite(audit), "audit JSON non-finite")
    expect(audit["status"] == "passed", "audit status")
    broad = audit["distribution_audit"]["continuous_broad_coverage"]
    expect(broad["passed"] is True, "continuous broad coverage")
    expect("strict uniformity is not supported" in broad["conclusion"], "uniform claim")
    metric_arrays = {
        "peak_deltaT_K": peak,
        "mean_deltaT_K": mean,
        "cv_rms_deltaT_K": rms,
        "Reff_peak_K_W": reff,
    }
    for name, values in metric_arrays.items():
        recorded = audit["distribution_audit"]["metrics"][name]
        expect(
            abs(recorded["summary"]["q00"] - float(np.min(values))) <= 1.0e-12,
            f"{name} minimum",
        )
        expect(
            abs(recorded["summary"]["q100"] - float(np.max(values))) <= 1.0e-12,
            f"{name} maximum",
        )
        expect(recorded["histogram"]["occupied_bins"] == 12, f"{name} bins")
    checks["distribution_formulas_recomputed"] = True

    power_top = audit["joint_input_coverage"]["power_top_h_assessment"]
    expect(
        power_top["conclusion"].startswith("artificially_coupled_not_deconfounded"),
        "power-top_h coupling conclusion",
    )
    expect(power_top["physical_6x6_empty_cells"] > 0, "power-top_h missing cells")
    expect(power_top["high_power_low_top_h_count"] == 0, "power-top_h corner")
    checks["joint_coverage_limitation_registered"] = True

    comparisons = read_csv(
        repo_root
        / CONFIG
        / f"{PREFIX}_training_preflight_split_comparison.csv"
    )
    expect(
        len(comparisons) == 15 * 3,
        "split comparison row count",
    )
    expect(
        all(row["test_used_for_rule_adjustment"] == "False" for row in comparisons),
        "test used for rule adjustment",
    )
    expect(
        audit["split_audit"]["test_iid_use"]
        == "descriptive_only_no_rule_adjustment",
        "test role declaration",
    )
    checks["split_descriptive_only"] = True

    authorization = load_json(
        repo_root / CONFIG / f"{PREFIX}_training_authorization.json"
    )
    expect(
        authorization["decision"]
        == "authorized_for_training_with_frozen_applicability_boundaries"
        and authorization["authorized"] is True,
        "training authorization",
    )
    expect(
        authorization["guardrails"]
        == {
            "dataset_regeneration_runs": 0,
            "formal1024_v1_modified": False,
            "model_inference_runs": 0,
            "sample_filter_or_replacement_runs": 0,
            "training_runs": 0,
        },
        "authorization guardrails",
    )
    checks["training_authorization_scoped"] = True

    contact = load_json(
        repo_root / CONFIG / "v6_p1i_cross_family_contact_evidence.json"
    )
    expect(
        contact["contact_contract"]
        == {
            "interface_model": "perfect_contact",
            "R_contact_m2K_W": 0.0,
            "finite_contact_resistance_sampled": False,
        },
        "cross-family contact contract",
    )
    expect(
        {row["family"] for row in contact["evidence"]}
        == {"P1h", "P1i", "V6 random-block"},
        "contact families",
    )
    expect(
        contact["applicability_boundary"]["future_dataset_only_proposal"]["status"]
        == "proposal_not_implemented",
        "future contact proposal lifecycle",
    )
    checks["contact_boundary_registered"] = True

    crosswalk = read_csv(
        repo_root / CONFIG / "v6_p1i_literature_id_crosswalk.csv"
    )
    mapping = {row["legacy_id"]: row["canonical_id"] for row in crosswalk}
    expect(mapping["P1I-L01"] == "V6-LIT-009", "literature crosswalk L01")
    expect(mapping["P1I-L02"] == "V6-LIT-004", "literature crosswalk L02")
    expect(len(set(mapping.values())) == 12, "canonical literature source count")
    background = read_csv(
        repo_root / CONFIG / "v6_p1i_background_k_contract.csv"
    )
    unresolved = {
        source_id
        for row in background
        for source_id in row["source_ids"].split("|")
        if source_id not in mapping
    }
    expect(not unresolved, f"unresolved background-k source IDs: {unresolved}")
    registry = load_json(
        repo_root / CONFIG / "v6_p1i_literature_registry_v1.json"
    )
    expect(registry["source_count"] == 12, "literature registry count")
    expect(
        registry["bindings"]["formal_manifest"]["sha256"]
        == FROZEN[CONFIG / f"{PREFIX}_manifest.json"],
        "literature manifest binding",
    )
    checks["literature_ids_and_bindings"] = True

    outputs_manifest = load_json(
        repo_root
        / CONFIG
        / f"{PREFIX}_training_preflight_outputs_manifest.json"
    )
    for row in outputs_manifest["artifacts"]:
        path = repo_root / row["path"]
        expect(path.is_file(), f"missing manifest output: {row['path']}")
        expect(sha256(path) == row["sha256"], f"output SHA drift: {row['path']}")
        expect(path.stat().st_size == row["size_bytes"], f"output size drift")
    checks["output_manifest_hashes"] = True

    archive = load_json(
        repo_root / CONFIG / f"{PREFIX}_archive_manifest.json"
    )
    expect(
        archive["status"] == "archived_external_immutable_revision",
        "archive status",
    )
    expect(
        archive["dataset_tree"]["canonical_file_list_sha256"] == tree_sha,
        "archive tree SHA",
    )
    expect(
        archive["dataset_tree"]["file_count"] == file_count,
        "archive file count",
    )
    expect(
        len(archive["external_archive"]["commit_sha"]) == 40,
        "HF commit SHA",
    )
    expect(
        archive["external_archive"]["verified_file_count"] == file_count + 4,
        "HF verified file count",
    )
    checks["external_archive_bound"] = True

    replay = load_json(
        repo_root / CONFIG / f"{PREFIX}_clean_checkout_replay.json"
    )
    expect(replay["status"] == "passed", "clean replay status")
    expect(replay["mismatched_artifacts"] == [], "clean replay mismatch")
    expect(replay["training_runs"] == 0, "replay training")
    expect(replay["model_inference_runs"] == 0, "replay inference")
    checks["clean_checkout_replay"] = True

    scan_paths = [
        path
        for path in REQUIRED_OUTPUTS
        if path.suffix.lower() in {".json", ".csv", ".md", ".yaml", ".yml"}
    ]
    forbidden = ("/private/tmp/", "/Users/", "\\\\Users\\\\")
    for relative in scan_paths:
        text = (repo_root / relative).read_text(encoding="utf-8")
        expect(
            not any(token in text for token in forbidden),
            f"local absolute path leaked: {relative}",
        )
    checks["no_local_absolute_paths"] = True

    result = {
        "schema_version": "heat3d_v6_p1i_training_preflight_check_v1",
        "dataset_id": DATASET_ID,
        "status": "passed",
        "checks": checks,
        "guardrails": {
            "formal1024_v1_modified": False,
            "dataset_regeneration_runs": 0,
            "sample_filter_or_replacement_runs": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
    }
    expect(all(checks.values()), "one or more checks false")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()
    result = check(args.repo_root.resolve())
    if args.write_result is not None:
        output = args.write_result
        if not output.is_absolute():
            output = args.repo_root.resolve() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
