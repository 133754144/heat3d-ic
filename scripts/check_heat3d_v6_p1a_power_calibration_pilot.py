#!/usr/bin/env python3
"""Validate the fixed V6-P1a power-calibration dataset and audit artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_heat3d_v6_p1a_power_calibration_pilot as generator  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_cases.yaml"
DEFAULT_DATASET = REPO_ROOT / "data/heat3d_v6_p1a_power_calibration16_v0"
DEFAULT_AUDIT = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_manifest.json"
DEFAULT_SAMPLES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_samples.csv"
DEFAULT_SOURCES_CSV = REPO_ROOT / "configs/heat3d_v6/v6_p1a_power_calibration_sources.csv"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _expect(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _check_registry(registry: Mapping[str, Any]) -> None:
    _expect(registry["sample_count"] == 16, "registry sample count must be 16")
    scope = registry["scope"]
    _expect(scope["layered_stacks_only"] is True, "only layered stacks are allowed")
    _expect(scope["peak_deltaT_filtering"] is False, "DeltaT filtering must be disabled")
    _expect(scope["peak_deltaT_resampling"] is False, "DeltaT resampling must be disabled")
    _expect(scope["model_training"] is False and scope["model_inference"] is False, "model work forbidden")
    physics = registry["physics"]
    bc = physics["boundary_conditions"]
    _expect(bc["top"] == {"type": "robin", "h_W_m2K": 500.0, "T_inf_K": 300.0}, "top BC mismatch")
    _expect(bc["bottom"] == {"type": "dirichlet", "T_K": 300.0}, "bottom BC mismatch")
    _expect(bc["sides"] == {"type": "adiabatic"}, "side BC mismatch")
    _expect(physics["contact"] == {"type": "perfect", "R_contact_m2K_W": 0.0}, "contact mismatch")
    _expect(physics["solver_mesh_intervals_xyz"] == [64, 64, 32], "native mesh mismatch")
    _expect(sum(int(layer["z_intervals"]) for layer in physics["layers_bottom_to_top"]) == 32, "z intervals mismatch")
    _expect(all(int(layer["z_intervals"]) >= 4 for layer in physics["layers_bottom_to_top"]), "underresolved layer")
    _expect(sum(physics["operator_projection"]["strata"].values()) == 1024, "point strata mismatch")

    cases = registry["cases"]
    _expect(len(cases) == 16, "exactly 16 cases required")
    expected_totals = [0.0005, 0.001, 0.002, 0.001, 0.004, 0.008, 0.013, 0.013, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 20.0, 20.0]
    realized_totals = [sum(float(source["power_W"]) for source in case["sources"]) for case in cases]
    _expect(np.allclose(realized_totals, expected_totals, rtol=0.0, atol=1.0e-15), "frozen powers changed")
    for case in cases:
        layers = {source["layer"] for source in case["sources"]}
        _expect(1 <= len(layers) <= 2, f"{case['id']}: active layer count out of range")
        _expect(layers <= {"active_lower", "active_upper"}, f"{case['id']}: source outside active layer")
        if case["literature_id"] == "L02":
            _expect(
                all(float(source["power_W"]) in {0.0005, 0.001, 0.002} for source in case["sources"]),
                f"{case['id']}: non-cited L02 component power",
            )
        elif case["literature_id"] == "L19":
            layer_power: dict[str, float] = {}
            for source in case["sources"]:
                layer_power[source["layer"]] = layer_power.get(source["layer"], 0.0) + float(source["power_W"])
            _expect(max(layer_power.values()) <= 10.0, f"{case['id']}: exceeds cited active-layer power")
        else:
            raise AssertionError(f"{case['id']}: unsupported literature power source")


def _check_generator_static_contract() -> None:
    signature = inspect.signature(generator._sample_points_before_labels)
    forbidden = {"temperature", "delta_t", "solver_error", "interpolation_error"}
    _expect(not (set(signature.parameters) & forbidden), "point sampler accepts label-derived inputs")
    source = inspect.getsource(generator.generate)
    _expect(
        source.index("points, point_strata, point_seed") < source.index("temperature, solver_audit"),
        "point coordinates must freeze before the temperature solve",
    )
    parsed = ast.parse(Path(generator.__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    _expect(not ({"jax", "flax", "optax", "rigno"} & imported_roots), "generator imports model/training stack")


def _check_sample(
    sample_dir: Path,
    manifest_row: Mapping[str, Any],
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, expected_hash in manifest_row["file_sha256"].items():
        path = sample_dir / name
        _expect(path.is_file(), f"{sample_dir.name}: missing {name}")
        _expect(_sha256(path) == expected_hash, f"{sample_dir.name}: hash mismatch {name}")
    coords = np.load(sample_dir / "coords.npy")
    temperature = np.load(sample_dir / "temperature.npy")
    delta_t = np.load(sample_dir / "deltaT.npy")
    k = np.load(sample_dir / "k_field.npy")
    q = np.load(sample_dir / "q_field.npy")
    layer = np.load(sample_dir / "layer_id.npy")
    bc = np.load(sample_dir / "bc_features.npy")
    _expect(coords.shape == (1024, 3), f"{sample_dir.name}: coords shape")
    _expect(temperature.shape == delta_t.shape == q.shape == layer.shape == (1024, 1), f"{sample_dir.name}: scalar array shape")
    _expect(k.shape == (1024, 3) and bc.shape == (1024, 4), f"{sample_dir.name}: feature shape")
    _expect(all(np.all(np.isfinite(array)) for array in (coords, temperature, delta_t, k, q, layer, bc)), f"{sample_dir.name}: non-finite array")
    _expect(len(np.unique(coords, axis=0)) == 1024, f"{sample_dir.name}: duplicate points")
    _expect(np.max(np.abs(delta_t[:, 0] - (temperature[:, 0] - 300.0))) < 1.0e-12, f"{sample_dir.name}: DeltaT mismatch")
    _expect(np.allclose(temperature[bc[:, 1] == 1.0, 0], 300.0, atol=1.0e-10), f"{sample_dir.name}: bottom Dirichlet mismatch")
    _expect(int(np.sum(bc[:, 0])) == 64 and int(np.sum(bc[:, 1])) == 64, f"{sample_dir.name}: boundary strata mismatch")

    meta = _read_json(sample_dir / "sample_meta.json")
    _expect(meta["power_was_Rth_inferred"] is False, f"{sample_dir.name}: power inference forbidden")
    _expect(meta["guardrails"]["peak_deltaT_filtering"] is False, f"{sample_dir.name}: filtered")
    _expect(meta["guardrails"]["peak_deltaT_resampling"] is False, f"{sample_dir.name}: resampled")
    _expect(meta["guardrails"]["training_runs"] == 0 and meta["guardrails"]["model_inference_runs"] == 0, f"{sample_dir.name}: model work")
    _expect(meta["contact"]["R_contact_m2K_W"] == 0.0, f"{sample_dir.name}: non-perfect contact")
    projection = meta["operator_projection"]
    _expect(projection["point_coordinates_frozen_before_temperature_solve"] is True, f"{sample_dir.name}: point freeze")
    _expect(projection["label_inputs_used_for_point_selection"] == [], f"{sample_dir.name}: point label leak")
    _expect(projection["strata_counts"] == {"bottom": 64, "interface": 128, "source": 256, "top": 64, "volume": 512}, f"{sample_dir.name}: strata")
    _expect(_array_sha256(coords) == projection["point_coordinates_sha256"], f"{sample_dir.name}: point SHA")
    _expect(projection["point_coordinates_sha256"] == manifest_row["point_coordinates_sha256"], f"{sample_dir.name}: manifest point SHA")

    sources = meta["sources"]
    _expect(len(sources) == len(case["sources"]), f"{sample_dir.name}: source count")
    realized_source_contract = sorted(
        (source["active_layer"], float(source["source_power_W"])) for source in sources
    )
    expected_source_contract = sorted(
        (source["layer"], float(source["power_W"])) for source in case["sources"]
    )
    _expect(realized_source_contract == expected_source_contract, f"{sample_dir.name}: source powers differ from registry")
    source_sum = 0.0
    layer_sums: dict[str, float] = {}
    for source in sources:
        power = float(source["source_power_W"])
        volume = float(source["source_volume_m3"])
        density = float(source["q_W_m3"])
        _expect(math.isclose(power, volume * density, rel_tol=1.0e-12, abs_tol=1.0e-14), f"{sample_dir.name}: q-volume-power closure")
        _expect(int(source["covered_control_volume_count"]) >= 256, f"{sample_dir.name}: underresolved source")
        source_sum += power
        layer_name = source["active_layer"]
        layer_sums[layer_name] = layer_sums.get(layer_name, 0.0) + power
    package_power = float(meta["metrics"]["package_total_power_W"])
    _expect(math.isclose(source_sum, package_power, rel_tol=1.0e-12), f"{sample_dir.name}: package power closure")
    for name, value in layer_sums.items():
        _expect(math.isclose(value, float(meta["active_layer_power_W"][name]), rel_tol=1.0e-12), f"{sample_dir.name}: layer power closure")
    metrics = meta["metrics"]
    _expect(all(math.isfinite(float(value)) for key, value in metrics.items() if key != "in_30_80_K_window"), f"{sample_dir.name}: non-finite metrics")
    _expect(abs(float(metrics["energy_balance_relative_error"])) < 1.0e-8, f"{sample_dir.name}: energy imbalance")
    _expect(abs(float(metrics["top_heat_fraction"]) + float(metrics["bottom_heat_fraction"]) - 1.0) < 1.0e-8, f"{sample_dir.name}: flux closure")
    return meta, sources


def check(args: argparse.Namespace) -> dict[str, Any]:
    registry = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    _check_registry(registry)
    _check_generator_static_contract()
    manifest = _read_json(args.dataset / "manifest.json")
    tracked_manifest = _read_json(args.manifest_json)
    _expect(tracked_manifest == manifest, "tracked manifest differs from generated dataset manifest")
    _expect(manifest["sample_count"] == 16 and len(manifest["samples"]) == 16, "manifest sample count")
    _expect(manifest["case_registry_sha256"] == _sha256(args.cases), "case-registry SHA mismatch")
    _expect(manifest["literature_matrix_sha256"] == registry["literature_matrix"]["sha256"], "literature SHA mismatch")
    _expect(manifest["guardrails"] == {"model_inference_runs": 0, "peak_deltaT_filtering": False, "peak_deltaT_resampling": False, "training_runs": 0}, "manifest guardrails")
    cases = {case["id"]: case for case in registry["cases"]}
    manifest_ids = [row["sample_id"] for row in manifest["samples"]]
    _expect(len(set(manifest_ids)) == 16 and set(manifest_ids) == set(cases), "manifest sample IDs mismatch")
    dataset_dirs = {path.name for path in args.dataset.iterdir() if path.is_dir()}
    _expect(dataset_dirs == set(cases), "dataset contains missing or extra sample directories")
    sample_metas: list[dict[str, Any]] = []
    source_count = 0
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        _expect(sample_id in cases, f"unexpected sample {sample_id}")
        meta, sources = _check_sample(args.dataset / sample_id, row, cases[sample_id])
        sample_metas.append(meta)
        source_count += len(sources)
    audit = _read_json(args.audit_json)
    _expect(audit["dataset_manifest_sha256"] == _sha256(args.dataset / "manifest.json"), "dataset manifest SHA mismatch")
    _expect(audit["tracked_manifest_sha256"] == _sha256(args.manifest_json), "tracked manifest SHA mismatch")
    _expect(audit["sample_count"] == 16 and audit["source_count"] == source_count, "audit counts")
    _expect(audit["guardrails"] == {"expanded_samples": 0, "generated_samples": 16, "model_inference_runs": 0, "peak_deltaT_filtering": False, "peak_deltaT_resampling": False, "training_runs": 0}, "audit guardrails")
    _expect(audit["integrity"]["all_metrics_finite"] is True, "audit finite flag")
    _expect(audit["integrity"]["min_source_control_volume_count"] >= 256, "audit source resolution")
    with args.samples_csv.open(encoding="utf-8", newline="") as handle:
        sample_csv = list(csv.DictReader(handle))
    with args.sources_csv.open(encoding="utf-8", newline="") as handle:
        source_csv = list(csv.DictReader(handle))
    _expect(len(sample_csv) == 16 and len(source_csv) == source_count, "CSV row counts")
    return {
        "schema_version": "heat3d_v6_p1a_checker_v1",
        "passed": True,
        "sample_count": 16,
        "source_count": source_count,
        "window_hit_count": audit["window_hit_count"],
        "max_abs_energy_balance_relative_error": audit["integrity"]["max_abs_energy_balance_relative_error"],
        "min_source_control_volume_count": audit["integrity"]["min_source_control_volume_count"],
        "training_runs": 0,
        "model_inference_runs": 0,
        "expanded_samples": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--sources-csv", type=Path, default=DEFAULT_SOURCES_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("cases", "dataset", "audit_json", "manifest_json", "samples_csv", "sources_csv"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (REPO_ROOT / value).resolve())
    result = check(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
