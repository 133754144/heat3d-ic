#!/usr/bin/env python3
"""Replay frozen P1i inputs into a non-mutating 240825-node HDF5 sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import heat3d_v6_p1i_continuous_core as core  # noqa: E402


DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
MANIFEST = ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json"
CONFIG = ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1.yaml"
MANIFEST_SHA = "f19987c659968c2ac14eade1f1ef7e206c8f7eeb94f58fde5897d6e765978514"
CONFIG_SHA = "1e15a77fe51eea7ec64614566bb6bb12bfcf05948f3b7c8c6f3c85ec759a58f8"
AMBIENT_K = 300.0
NODE_COUNT = 240825


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / str(row["relative_path"])


def _verify_source(path: Path, row: dict[str, Any]) -> None:
    for name, digest in row["file_sha256"].items():
        if not (path / name).is_file() or core.file_sha256(path / name) != digest:
            raise RuntimeError(f"frozen source hash mismatch: {path / name}")


def _rebuild(meta: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    mesh = core.build_mesh(meta["physics"])
    overlap_count = 0
    for left in meta["k_blocks"]:
        for right in meta["q_blocks"]:
            if left["layer"] != right["layer"]:
                continue
            lx0, lx1, ly0, ly1 = map(float, left["bbox_fraction_xy"])
            rx0, rx1, ry0, ry1 = map(float, right["bbox_fraction_xy"])
            overlap_count += int(
                min(lx1, rx1) > max(lx0, rx0)
                and min(ly1, ry1) > max(ly0, ry0)
            )
    group = {
        "group_id": meta["group_id"],
        "split_role": meta["split_role"],
        "q_blocks": meta["q_blocks"],
        "k_blocks": meta["k_blocks"],
        "cross_family_overlap_pair_count": overlap_count,
    }
    layout = core.validate_layout(group, mesh)
    k_field, q_field, _ = core.build_case_fields(meta, group, mesh, layout)
    return mesh, k_field, q_field


def _support_indices(mesh: dict[str, Any], coords: np.ndarray) -> np.ndarray:
    axes = [np.asarray(mesh[name]) for name in ("x", "y", "z")]
    found = [np.searchsorted(axis, coords[:, dim]) for dim, axis in enumerate(axes)]
    for axis, values, indices in zip(axes, coords.T, found, strict=True):
        if np.any(indices >= axis.size) or not np.array_equal(axis[indices], values):
            raise RuntimeError("saved support is not an exact solver-node subset")
    flat = np.asarray(mesh["grid"])[found[0], found[1], found[2]]
    if not np.array_equal(np.asarray(mesh["coords"])[flat], coords):
        raise RuntimeError("saved coordinate order could not be replayed")
    return flat


def _create_archive(path: Path, rows: list[dict[str, Any]], mesh: dict[str, Any]) -> h5py.File:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sidecar archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = h5py.File(path, "w")
    archive.attrs.update({
        "schema_version": "heat3d_v6_p1i_full_field_sidecar_v1",
        "dataset_id": DATASET_ID,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_config_sha256": CONFIG_SHA,
        "ambient_K": AMBIENT_K,
        "label_dtype": "float32",
    })
    shared = archive.create_group("shared")
    for name, value in (
        ("coords_m", mesh["coords"]),
        ("control_volume_m3", mesh["weights"]),
        ("layer_id", mesh["layer_ids"]),
        ("boundary_flags", core.boundary_flags(mesh["coords"], mesh)),
    ):
        shared.create_dataset(name, data=np.asarray(value), compression="gzip", shuffle=True)
    samples = archive.create_group("samples")
    string = h5py.string_dtype("utf-8")
    samples.create_dataset("sample_id", data=np.asarray([r["sample_id"] for r in rows], dtype=object), dtype=string)
    samples.create_dataset("split_role", data=np.asarray([r["split_role"] for r in rows], dtype=object), dtype=string)
    shape, chunks = (len(rows), NODE_COUNT), (1, NODE_COUNT)
    for name in ("temperature_K", "deltaT_K"):
        samples.create_dataset(name, shape=shape, chunks=chunks, dtype=np.float32,
                               compression="gzip", compression_opts=4, shuffle=True)
    for name in ("top_heat_flux_W", "bottom_heat_flux_W", "linear_residual",
                 "peak_deltaT_K", "mean_deltaT_K", "cv_rms_deltaT_K",
                 "projection_max_abs_error_K", "float32_cast_max_abs_error_K"):
        samples.create_dataset(name, shape=(len(rows),), dtype=np.float64)
    samples.create_dataset("cg_iterations", shape=(len(rows),), dtype=np.int32)
    return archive


def build(dataset_root: Path, output_root: Path, manifest_output: Path, limit: int | None) -> dict[str, Any]:
    started = time.perf_counter()
    if core.file_sha256(MANIFEST) != MANIFEST_SHA or core.file_sha256(CONFIG) != CONFIG_SHA:
        raise RuntimeError("frozen P1i config/manifest SHA drift")
    source_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = list(source_manifest["samples"])
    if source_manifest.get("dataset_id") != DATASET_ID or len(rows) != 1024:
        raise RuntimeError("unexpected frozen P1i manifest identity")
    selected = rows if limit is None else rows[: int(limit)]
    first_dir = _source_dir(dataset_root, selected[0])
    _verify_source(first_dir, selected[0])
    first_meta = json.loads((first_dir / "sample_meta.json").read_text(encoding="utf-8"))
    shared_mesh, _, _ = _rebuild(first_meta)
    if int(shared_mesh["node_count"]) != NODE_COUNT:
        raise RuntimeError("solver node count drift")
    archive_path = output_root / "full_fields.h5"
    archive = _create_archive(archive_path, selected, shared_mesh)
    maxima = {name: 0.0 for name in ("coords_m", "k_W_mK", "q_W_m3",
                                      "projected_temperature_K", "field_metric", "float32_cast_K")}
    row_hashes: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected):
            sample_started = time.perf_counter()
            sample_dir = _source_dir(dataset_root, row)
            _verify_source(sample_dir, row)
            meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
            mesh, k_field, q_field = _rebuild(meta)
            for key in ("coords", "weights", "layer_ids"):
                if not np.array_equal(np.asarray(mesh[key]), np.asarray(shared_mesh[key])):
                    raise RuntimeError(f"{row['sample_id']}: shared mesh drift")
            coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
            support = _support_indices(mesh, coords)
            errors = [
                float(np.max(np.abs(np.asarray(mesh["coords"])[support] - coords))),
                float(np.max(np.abs(k_field[support] - np.load(sample_dir / "k_field.npy")))),
                float(np.max(np.abs(q_field[support] - np.load(sample_dir / "q_field.npy").reshape(-1)))),
            ]
            temperature, solver = core.solve_case(
                mesh, k_field, q_field,
                top_h=float(meta["top_h_W_m2K"]),
                bottom_h=float(meta["bottom_h_W_m2K"]), ambient_K=AMBIENT_K,
            )
            projection_error = float(np.max(np.abs(
                temperature[support] - np.load(sample_dir / "temperature.npy").reshape(-1))))
            metrics = core.case_metrics(mesh, temperature, q_field, solver, ambient_K=AMBIENT_K)
            metric_keys = ("peak_deltaT_K", "mean_deltaT_K", "cv_rms_deltaT_K",
                           "top_heat_fraction", "bottom_heat_fraction",
                           "energy_balance_relative_error", "linear_residual")
            metric_error = max(abs(float(metrics[k]) - float(meta[k])) for k in metric_keys)
            temperature32 = temperature.astype(np.float32)
            delta32 = (temperature - AMBIENT_K).astype(np.float32)
            cast_error = max(float(np.max(np.abs(temperature32.astype(np.float64) - temperature))),
                             float(np.max(np.abs(delta32.astype(np.float64) - (temperature - AMBIENT_K)))))
            archive["samples/temperature_K"][index] = temperature32
            archive["samples/deltaT_K"][index] = delta32
            for name, value in (
                ("top_heat_flux_W", solver["top_heat_flux_W"]),
                ("bottom_heat_flux_W", solver["bottom_heat_flux_W"]),
                ("linear_residual", solver["linear_residual"]),
                ("peak_deltaT_K", metrics["peak_deltaT_K"]),
                ("mean_deltaT_K", metrics["mean_deltaT_K"]),
                ("cv_rms_deltaT_K", metrics["cv_rms_deltaT_K"]),
                ("projection_max_abs_error_K", projection_error),
                ("float32_cast_max_abs_error_K", cast_error),
            ):
                archive[f"samples/{name}"][index] = float(value)
            archive["samples/cg_iterations"][index] = int(solver["cg_iterations"])
            archive.flush()
            row_hashes.append({"sample_id": row["sample_id"], "split_role": row["split_role"],
                               "temperature_float32_raw_sha256": _array_sha(temperature32),
                               "deltaT_float32_raw_sha256": _array_sha(delta32)})
            for name, value in zip(maxima, (*errors, projection_error, metric_error, cast_error), strict=True):
                maxima[name] = max(maxima[name], value)
            print(f"[{index + 1:04d}/{len(selected):04d}] {row['sample_id']} "
                  f"projection={projection_error:.3e}K elapsed={time.perf_counter()-sample_started:.2f}s", flush=True)
    finally:
        archive.close()
    with h5py.File(archive_path, "r") as archive:
        shared_hashes = {name: _array_sha(archive[f"shared/{name}"][:])
                         for name in ("coords_m", "control_volume_m3", "layer_id", "boundary_flags")}
    complete = len(selected) == 1024
    payload = {
        "schema_version": "heat3d_v6_p1i_full_field_sidecar_manifest_v1",
        "status": "complete" if complete else "smoke_subset",
        "dataset_id": DATASET_ID,
        "source_dataset_policy_path": f"data/{DATASET_ID}",
        "source_manifest_path": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": MANIFEST_SHA,
        "source_manifest_payload_sha256": source_manifest["manifest_payload_sha256"],
        "source_config_path": str(CONFIG.relative_to(ROOT)),
        "source_config_sha256": CONFIG_SHA,
        "sidecar_policy_path": f"data/{DATASET_ID}_full_fields",
        "archive_file": "full_fields.h5",
        "archive_sha256": core.file_sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "sample_count": len(selected), "solver_node_count": NODE_COUNT,
        "label_dtype": "float32", "ambient_K": AMBIENT_K,
        "shared_raw_sha256": shared_hashes, "samples": row_hashes,
        "replay_max_abs_error": maxima,
        "tolerances": {"coords_m": 0.0, "k_W_mK": 0.0, "q_W_m3": 0.0,
                       "projected_temperature_K": 1.0e-8, "field_metric": 1.0e-8,
                       "float32_cast_K": 5.0e-5},
        "elapsed_seconds": time.perf_counter() - started,
        "guardrails": {"source_1024_point_files_modified": False,
                       "sample_or_split_rules_modified": False, "training_runs": 0,
                       "model_inference_runs": 0, "test_labels_used_for_selection": False},
        "hf_archive": None,
    }
    _write_json(output_root / "sidecar_manifest.json", payload)
    _write_json(manifest_output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / DATASET_ID)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / f"{DATASET_ID}_full_fields")
    parser.add_argument("--manifest-output", type=Path,
                        default=ROOT / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_full_field_sidecar_manifest.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = build(args.dataset_root.resolve(), args.output_root.resolve(),
                   args.manifest_output.resolve(), args.limit)
    print(json.dumps({k: result[k] for k in ("status", "sample_count", "archive_sha256", "archive_size_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
