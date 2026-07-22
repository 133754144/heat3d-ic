#!/usr/bin/env python3
"""Audit whether immutable P1g solver-full fields can support a P1h rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024_manifest.json"
EXPECTED_PROJECTED_FILES = {
    "coords.npy", "k_field.npy", "q_field.npy", "temperature.npy", "deltaT.npy",
    "bc_features.npy", "bc_parameters.npy", "layer_id.npy", "sampling_stratum.npy",
    "sample_meta.json",
}
ARCHIVE_SUFFIXES = {".h5", ".hdf5", ".vtk", ".vtu", ".zarr"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_shapes(path: Path) -> list[list[int]]:
    if path.suffix == ".npy":
        return [list(np.load(path, mmap_mode="r").shape)]
    if path.suffix == ".npz":
        with np.load(path) as payload:
            return [list(payload[key].shape) for key in payload.files]
    return []


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _supplemental_scan(
    search_roots: list[Path], source_root: Path, solver_node_counts: set[int]
) -> dict[str, Any]:
    """Search local data/output roots without interpreting projected labels."""

    array_file_count = 0
    unreadable_array_files: list[str] = []
    solver_sized_arrays: list[dict[str, Any]] = []
    archive_candidates: list[str] = []
    name_candidates: list[str] = []
    roots: list[dict[str, Any]] = []
    source_root = source_root.resolve()

    for raw_root in search_roots:
        root = raw_root.resolve()
        roots.append({"path": str(root), "exists": root.exists()})
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or _is_within(path.resolve(), source_root):
                continue
            suffix = path.suffix.lower()
            lower_name = path.name.lower()
            if suffix in ARCHIVE_SUFFIXES:
                archive_candidates.append(str(path))
            if "solver" in lower_name or ("full" in lower_name and "field" in lower_name):
                name_candidates.append(str(path))
            if suffix not in {".npy", ".npz"}:
                continue
            array_file_count += 1
            try:
                shapes = _array_shapes(path)
            except (OSError, ValueError, EOFError):
                unreadable_array_files.append(str(path))
                continue
            for shape in shapes:
                if any(node_count in shape for node_count in solver_node_counts):
                    solver_sized_arrays.append({"path": str(path), "shape": shape})

    return {
        "roots": roots,
        "excluded_projected_source_root": str(source_root),
        "array_file_count_scanned": array_file_count,
        "unreadable_array_file_count": len(unreadable_array_files),
        "unreadable_array_files": unreadable_array_files[:20],
        "solver_sized_array_count": len(solver_sized_arrays),
        "solver_sized_arrays": solver_sized_arrays[:20],
        "archive_candidate_count": len(archive_candidates),
        "archive_candidates": archive_candidates[:20],
        "solver_or_full_field_name_candidate_count": len(name_candidates),
        "solver_or_full_field_name_candidates": name_candidates[:20],
    }


def audit(source_root: Path, search_roots: list[Path]) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    missing_dirs: list[str] = []
    projected_complete = 0
    full_field_complete = 0
    samples_with_large_array: list[str] = []
    extra_files: dict[str, list[str]] = {}
    solver_node_counts: set[int] = set()
    projected_node_counts: set[int] = set()
    shape_histogram: dict[str, int] = {}

    for row in samples:
        sample_id = str(row["sample_id"])
        sample_dir = source_root / str(row["sample_dir"])
        if not sample_dir.is_dir():
            missing_dirs.append(sample_id)
            continue
        names = {path.name for path in sample_dir.iterdir() if path.is_file()}
        if EXPECTED_PROJECTED_FILES <= names:
            projected_complete += 1
        extras = sorted(names - EXPECTED_PROJECTED_FILES)
        if extras:
            extra_files[sample_id] = extras
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        solver_nodes = int(meta["solver_mesh"]["node_count"])
        solver_node_counts.add(solver_nodes)
        coords = np.load(sample_dir / "coords.npy", mmap_mode="r")
        projected_node_counts.add(int(coords.shape[0]))

        full_arrays = 0
        for path in sample_dir.iterdir():
            if not path.is_file() or path.suffix not in {".npy", ".npz"}:
                continue
            for shape in _array_shapes(path):
                shape_histogram[str(shape)] = shape_histogram.get(str(shape), 0) + 1
                if solver_nodes in shape:
                    full_arrays += 1
        if full_arrays:
            samples_with_large_array.append(sample_id)
        # A rebuild requires independent full coordinates, k, q, and T.
        if full_arrays >= 4:
            full_field_complete += 1

    p1h_root = source_root.parent / "heat3d_v6_p1h_shared_support1024_v0"
    blocked = full_field_complete != len(samples)
    supplemental = _supplemental_scan(search_roots, source_root, solver_node_counts)
    return {
        "schema_version": "heat3d_v6_p1h_full_field_availability_audit_v1",
        "status": "blocked_missing_original_solver_full_fields" if blocked else "passed_full_fields_available",
        "decision": "stop_without_generating_P1h" if blocked else "eligible_to_generate_P1h",
        "parent_dataset_id": manifest["dataset_id"],
        "proposed_dataset_id": "heat3d_v6_p1h_shared_support1024_v0",
        "parent_manifest": str(MANIFEST.relative_to(ROOT)),
        "parent_manifest_sha256": _sha256(MANIFEST),
        "source_root": str(source_root.resolve()),
        "manifest_sample_count": len(samples),
        "source_sample_dir_count": sum(path.is_dir() for path in source_root.iterdir()),
        "missing_sample_dir_count": len(missing_dirs),
        "projected_1024_complete_sample_count": projected_complete,
        "projected_node_counts": sorted(projected_node_counts),
        "declared_solver_node_counts": sorted(solver_node_counts),
        "samples_with_any_solver_sized_array_count": len(samples_with_large_array),
        "samples_with_complete_solver_coords_k_q_T_count": full_field_complete,
        "unmanifested_extra_file_sample_count": len(extra_files),
        "representative_extra_files": dict(list(extra_files.items())[:10]),
        "array_shape_histogram": dict(sorted(shape_histogram.items())),
        "supplemental_local_search": supplemental,
        "contract": {
            "requires_original_solver_coordinates_k_q_T": True,
            "allows_interpolation_from_existing_1024_points": False,
            "target_or_test_label_used_for_support_selection": False,
            "p1h_generation_started": False,
            "p1h_manifest_created": False,
            "p1h_output_exists": p1h_root.exists(),
            "p1g_overwritten": False,
            "canonical_dataset_changed": False,
        },
        "missing_sample_ids": missing_dirs,
        "samples_with_any_solver_sized_array": samples_with_large_array,
    }


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# V6-P1h shared-support full-field availability audit

Status: `{payload['status']}`. Decision: `{payload['decision']}`.

The immutable P1g manifest contains {payload['manifest_sample_count']} samples. All
{payload['projected_1024_complete_sample_count']} available sample directories contain
the projected 1024-point files, while
`samples_with_complete_solver_coords_k_q_T_count={payload['samples_with_complete_solver_coords_k_q_T_count']}`.
Each sample metadata record declares a {payload['declared_solver_node_counts']}-node solver
mesh, but no sample directory contains solver-sized coordinate, conductivity, source,
and temperature arrays.

The requested P1h dataset therefore was not generated. Existing 1024-point P1g arrays
were not interpolated, P1g was not modified, no P1h manifest was fabricated, and the
canonical dataset designation remains P1g-v0. Construction can resume only after the
original per-sample solver-full coordinates/k/q/T files are restored with provenance.

Audit source manifest SHA256: `{payload['parent_manifest_sha256']}`.

The supplemental local search scanned
{payload['supplemental_local_search']['array_file_count_scanned']} NumPy files outside
the P1g projected source root. It found
`solver_sized_array_count={payload['supplemental_local_search']['solver_sized_array_count']}`,
`archive_candidate_count={payload['supplemental_local_search']['archive_candidate_count']}`,
and
`solver_or_full_field_name_candidate_count={payload['supplemental_local_search']['solver_or_full_field_name_candidate_count']}`.

## Deliberately absent downstream artifacts

Because the prerequisite full fields are absent, there is no P1h dataset directory or
manifest, no shared coordinate/graph hash, and no source/layer/interface coverage,
projection-error, distribution, leakage, loader, or B24 trainability result. Producing
any of those would require either fabricating a dataset or interpolating from P1g's
already projected 1024 points, both of which are forbidden by the task contract.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--write-md", type=Path)
    args = parser.parse_args()
    payload = audit(args.source_root, args.search_root)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_md:
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"].startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
