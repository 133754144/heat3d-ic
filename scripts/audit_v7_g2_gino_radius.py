#!/usr/bin/env python3
"""Geometry-only GINO radius audit for role-scoped P1i coordinate files.

The audit never opens targets or physical features. It maps coordinates with
train-only bounds, applies the frozen 32^3 latent grid, and reproduces
NeuralOperator's closed-ball ``distance <= radius`` neighbor semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ALLOWED_ROLES = {"train", "valid_iid"}
FORBIDDEN_ROLE_TOKENS = ("test", "sealed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(counts: np.ndarray) -> dict[str, float | int]:
    return {
        "zero_neighbor_fraction": float(np.mean(counts == 0)),
        "mean_neighbors": float(np.mean(counts)),
        "median_neighbors": float(np.median(counts)),
        "p5_neighbors": float(np.percentile(counts, 5)),
        "p95_neighbors": float(np.percentile(counts, 95)),
        "min_neighbors": int(np.min(counts)),
        "max_neighbors": int(np.max(counts)),
        "edge_count": int(np.sum(counts, dtype=np.int64)),
    }


def role_audit(
    coordinates: list[np.ndarray],
    latent_grid: np.ndarray,
    input_radius: float,
    output_radius: float,
) -> dict[str, Any]:
    input_query_counts: list[np.ndarray] = []
    output_query_counts: list[np.ndarray] = []
    input_source_counts: list[np.ndarray] = []
    output_source_counts: list[np.ndarray] = []
    nearest_input: list[np.ndarray] = []
    nearest_output: list[np.ndarray] = []
    latent_tree = cKDTree(latent_grid)
    for points in coordinates:
        point_tree = cKDTree(points)
        input_counts = point_tree.query_ball_point(
            latent_grid, input_radius, return_length=True
        )
        output_counts = latent_tree.query_ball_point(
            points, output_radius, return_length=True
        )
        input_query_counts.append(np.asarray(input_counts, dtype=np.int64))
        output_query_counts.append(np.asarray(output_counts, dtype=np.int64))
        # An input point participates if at least one latent query finds it.
        input_source_counts.append(
            np.asarray(
                latent_tree.query_ball_point(points, input_radius, return_length=True),
                dtype=np.int64,
            )
        )
        # A latent point participates in output GNO if at least one P1i query
        # finds it. This count equals the input-query count geometrically.
        output_source_counts.append(
            np.asarray(
                point_tree.query_ball_point(
                    latent_grid, output_radius, return_length=True
                ),
                dtype=np.int64,
            )
        )
        nearest_input.append(point_tree.query(latent_grid, k=1)[0])
        nearest_output.append(latent_tree.query(points, k=1)[0])
    input_counts = np.concatenate(input_query_counts)
    output_counts = np.concatenate(output_query_counts)
    input_sources = np.concatenate(input_source_counts)
    output_sources = np.concatenate(output_source_counts)
    return {
        "input_gno": {
            "radius": input_radius,
            "queries_are": "32^3 latent grid points",
            "sources_are": "P1i points",
            **summarize(input_counts),
            "query_coverage_fraction": float(np.mean(input_counts > 0)),
            "source_participation_fraction": float(np.mean(input_sources > 0)),
        },
        "output_gno": {
            "radius": output_radius,
            "queries_are": "P1i points",
            "sources_are": "32^3 latent grid points",
            **summarize(output_counts),
            "query_coverage_fraction": float(np.mean(output_counts > 0)),
            "source_participation_fraction": float(np.mean(output_sources > 0)),
        },
        "nearest_source_distance": {
            "input_gno_p50_p95_p99_max": [
                float(value)
                for value in np.percentile(
                    np.concatenate(nearest_input), [50, 95, 99, 100]
                )
            ],
            "output_gno_p50_p95_p99_max": [
                float(value)
                for value in np.percentile(
                    np.concatenate(nearest_output), [50, 95, 99, 100]
                )
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--all-allowed-from-split", action="store_true")
    parser.add_argument("--input-radius", type=float)
    parser.add_argument("--output-radius", type=float)
    parser.add_argument("--symmetric-reference-radius", type=float)
    # Kept only so the P2 receipt remains reproducible. New formal receipts use
    # the explicit asymmetric arguments above.
    parser.add_argument("--radius", action="append", type=float)
    parser.add_argument("--latent-resolution", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    assignment = split["assignment"]
    rows = {row["sample_id"]: row for row in dataset["samples"]}
    by_role: dict[str, list[np.ndarray]] = {"train": [], "valid_iid": []}
    verified: list[dict[str, str]] = []
    if args.all_allowed_from_split:
        if args.sample_id:
            parser.error("do not combine --all-allowed-from-split and --sample-id")
        sample_ids = sorted(
            sample_id
            for sample_id, role in assignment.items()
            if role in ALLOWED_ROLES
        )
        if len(sample_ids) != 896:
            raise ValueError(f"expected 896 open-role geometries, got {len(sample_ids)}")
    else:
        if not args.sample_id:
            parser.error("provide --sample-id or --all-allowed-from-split")
        sample_ids = args.sample_id
    for sample_id in sample_ids:
        role = assignment.get(sample_id)
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{sample_id}: forbidden or unknown role {role!r}")
        if any(token in sample_id.lower() for token in FORBIDDEN_ROLE_TOKENS):
            raise ValueError(f"{sample_id}: closed role token in identifier")
        path = args.coords_root / sample_id / "coords.npy"
        expected = rows[sample_id]["file_sha256"]["coords.npy"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{sample_id}: coords SHA mismatch")
        coords = np.load(path, allow_pickle=False).astype(np.float64)
        if coords.shape != (1024, 3):
            raise ValueError(f"{sample_id}: unexpected coordinates {coords.shape}")
        by_role[role].append(coords)
        verified.append({"sample_id": sample_id, "role": role, "coords_sha256": actual})
    if not by_role["train"] or not by_role["valid_iid"]:
        raise ValueError("audit requires at least one train and one valid_iid geometry")

    train_stack = np.concatenate(by_role["train"], axis=0)
    lower = train_stack.min(axis=0)
    upper = train_stack.max(axis=0)
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError("degenerate train coordinate bounds")
    unit = {
        role: [(coords - lower) / span for coords in values]
        for role, values in by_role.items()
    }
    axis = np.linspace(0.0, 1.0, args.latent_resolution)
    latent_grid = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    if args.radius:
        audits = {
            str(radius): {
                role: role_audit(unit[role], latent_grid, radius, radius)
                for role in ("train", "valid_iid")
            }
            for radius in args.radius
        }
        radius_contract: dict[str, Any] = {"legacy_symmetric_audits": audits}
    else:
        if args.input_radius is None or args.output_radius is None:
            parser.error("provide --input-radius and --output-radius")
        asymmetric = {
            role: role_audit(
                unit[role], latent_grid, args.input_radius, args.output_radius
            )
            for role in ("train", "valid_iid")
        }
        reference_radius = args.symmetric_reference_radius
        reference = None
        if reference_radius is not None:
            reference = {
                role: role_audit(
                    unit[role], latent_grid, reference_radius, reference_radius
                )
                for role in ("train", "valid_iid")
            }
        role_ratios: dict[str, float] = {}
        totals = {"asymmetric": 0, "symmetric_reference": 0}
        if reference is not None:
            for role in ("train", "valid_iid"):
                asymmetric_edges = sum(
                    asymmetric[role][direction]["edge_count"]
                    for direction in ("input_gno", "output_gno")
                )
                reference_edges = sum(
                    reference[role][direction]["edge_count"]
                    for direction in ("input_gno", "output_gno")
                )
                role_ratios[role] = asymmetric_edges / reference_edges
                totals["asymmetric"] += asymmetric_edges
                totals["symmetric_reference"] += reference_edges
        radius_contract = {
            "asymmetric": asymmetric,
            "symmetric_reference": reference,
            "asymmetric_vs_symmetric_graph_size_ratio": {
                "by_role": role_ratios,
                "combined": (
                    totals["asymmetric"] / totals["symmetric_reference"]
                    if totals["symmetric_reference"]
                    else None
                ),
                "edge_totals": totals,
            },
        }
    verified_payload = json.dumps(
        verified, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema_version": "heat3d_v7_g2_gino_geometry_radius_audit_v2",
        "method": "geometry_only_scipy_ckdtree_closed_ball_distance_le_radius",
        "split_manifest_sha256": sha256(args.split_manifest),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "latent_grid": [args.latent_resolution] * 3,
        "train_only_coordinate_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "verified_samples": {
            "count": len(verified),
            "by_role": {
                role: sum(row["role"] == role for row in verified)
                for role in ("train", "valid_iid")
            },
            "ordered_sample_id_role_coords_sha256_index_sha256": hashlib.sha256(
                verified_payload
            ).hexdigest(),
            "individual_hash_source": "frozen dataset manifest referenced above",
        },
        "forbidden_arrays_opened": False,
        "formal_accuracy_observed": False,
        "radius_contract": radius_contract,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
