#!/usr/bin/env python3
"""Prepare the nested, label-independent V6 source-aware resolution ladder."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
RESOLUTIONS = (1024, 2048, 4096, 8192, 16384, 32768)
NEXT_RESOLUTION = 65536
SEED = 2026072801
RATIOS = {
    "source": 0.50,
    "volume": 0.25,
    "interface": 0.125,
    "top": 0.0625,
    "bottom": 0.0625,
}
CODE_TO_NAME = {0: "volume", 1: "source", 2: "interface", 3: "top", 4: "bottom"}
MANIFEST_SHA256 = "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
ARCHIVE_SHA256 = "f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _interleaved_permutation(
    groups: Iterable[np.ndarray], *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(np.asarray(group, dtype=np.int32)) for group in groups]
    result: list[int] = []
    cursor = 0
    while True:
        added = False
        for group in shuffled:
            if cursor < len(group):
                result.append(int(group[cursor]))
                added = True
        if not added:
            break
        cursor += 1
    return np.asarray(result, dtype=np.int32)


def _quotas(resolution: int) -> dict[str, int]:
    values = {name: int(round(fraction * resolution)) for name, fraction in RATIOS.items()}
    if sum(values.values()) != resolution:
        raise RuntimeError(f"{resolution}: stratum quotas do not sum to resolution")
    return values


def _source_box_coverage(
    *,
    indices: np.ndarray,
    coords: np.ndarray,
    layer_id: np.ndarray,
    groups: list[dict[str, Any]],
    layer_index: dict[str, int],
) -> dict[str, Any]:
    selected_coords = coords[indices]
    selected_layers = layer_id[indices]
    counts: list[int] = []
    for group in groups:
        for source in group["sources"]:
            x0, x1, y0, y1 = map(float, source["bbox_fraction_xy"])
            active_layer = layer_index[str(source["layer"])]
            mask = (
                (selected_layers == active_layer)
                & (selected_coords[:, 0] >= 0.01 * x0)
                & (selected_coords[:, 0] <= 0.01 * x1)
                & (selected_coords[:, 1] >= 0.01 * y0)
                & (selected_coords[:, 1] <= 0.01 * y1)
            )
            counts.append(int(np.sum(mask)))
    values = np.asarray(counts, dtype=np.int32)
    return {
        "source_box_count": int(len(values)),
        "zero_covered_source_box_count": int(np.sum(values == 0)),
        "all_source_boxes_covered": bool(np.all(values > 0)),
        "node_count_per_source_box": {
            "min": int(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": int(np.max(values)),
        },
        "source_box_inputs": (
            "registered geometry-group source boxes only; split_role and all "
            "temperature/model fields are excluded"
        ),
    }


def build(dataset_root: Path, manifest_path: Path, config_path: Path) -> dict[str, Any]:
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("P1h manifest SHA256 mismatch")
    archive_path = dataset_root / "full_fields.h5"
    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise RuntimeError("P1h full-field archive SHA256 mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = []
    for row in config["geometry_groups"]:
        groups.append(
            {
                "group_id": str(row["group_id"]),
                "sources": [
                    {
                        "layer": str(source["layer"]),
                        "bbox_fraction_xy": list(source["bbox_fraction_xy"]),
                    }
                    for source in row["sources"]
                ],
            }
        )
    with h5py.File(archive_path, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        layer_id = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        cv = np.asarray(handle["mesh/control_volume"], dtype=np.float64)
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
        anchors = np.asarray(handle["support/indices"], dtype=np.int32)
        anchor_codes = np.asarray(handle["support/stratum_code"], dtype=np.int8)
    if coords.shape != (240825, 3) or anchors.shape != (1024,):
        raise RuntimeError("P1h solver mesh/anchor shape drifted")
    anchor_strata = np.asarray([CODE_TO_NAME[int(code)] for code in anchor_codes])
    if Counter(anchor_strata) != {
        "source": 512,
        "volume": 256,
        "interface": 128,
        "top": 64,
        "bottom": 64,
    }:
        raise RuntimeError("canonical P1h anchor ratio drifted")
    anchor_set = set(map(int, anchors))
    z = coords[:, 2]
    top_mask = np.isclose(z, float(np.max(z)), atol=1e-15)
    bottom_mask = np.isclose(z, float(np.min(z)), atol=1e-15)
    interface_masks = [
        np.isclose(z, float(value), atol=1e-15)
        for value in boundaries[1:-1]
    ]
    interface_union = np.any(np.stack(interface_masks, axis=1), axis=1)
    layer_names = [
        "pcb_fr4_equivalent",
        "bt_substrate_with_vias",
        "silicon_interposer_tsv_0p1",
        "bump_underfill_under_interposer",
        "silicon_die_lower",
        "tim_between_dies",
        "silicon_die_upper",
        "tim_to_spreader",
        "spreader",
    ]
    layer_index = {name: index for index, name in enumerate(layer_names)}
    source_union = np.isin(
        layer_id,
        [layer_index["silicon_die_lower"], layer_index["silicon_die_upper"]],
    )
    all_indices = np.arange(len(coords), dtype=np.int32)
    not_anchor = ~np.isin(all_indices, anchors)

    # Physical domains are disjoint by priority; existing canonical anchors
    # retain their frozen stratum codes even when located on another domain.
    top_pool = all_indices[top_mask & not_anchor]
    bottom_pool = all_indices[bottom_mask & not_anchor]
    interface_groups = [
        all_indices[mask & ~top_mask & ~bottom_mask & not_anchor]
        for mask in interface_masks
    ]
    source_groups = [
        all_indices[
            (layer_id == active)
            & ~top_mask
            & ~bottom_mask
            & ~interface_union
            & not_anchor
        ]
        for active in (
            layer_index["silicon_die_lower"],
            layer_index["silicon_die_upper"],
        )
    ]
    reserved = top_mask | bottom_mask | interface_union | source_union
    volume_groups = [
        all_indices[(layer_id == index) & ~reserved & not_anchor]
        for index in range(9)
    ]
    sequences = {
        "top": _interleaved_permutation([top_pool], seed=SEED + 11),
        "bottom": _interleaved_permutation([bottom_pool], seed=SEED + 13),
        "interface": _interleaved_permutation(interface_groups, seed=SEED + 17),
        "source": _interleaved_permutation(source_groups, seed=SEED + 19),
        "volume": _interleaved_permutation(volume_groups, seed=SEED + 23),
    }
    anchor_by_stratum = {
        name: anchors[anchor_strata == name] for name in RATIOS
    }
    capacity = {
        name: int(len(anchor_by_stratum[name]) + len(sequences[name]))
        for name in RATIOS
    }

    probes: dict[str, Any] = {}
    prior: set[int] = set()
    for resolution in RESOLUTIONS:
        quotas = _quotas(resolution)
        selected_parts = [anchors]
        added_counts = {}
        for name in RATIOS:
            needed = quotas[name] - len(anchor_by_stratum[name])
            if needed < 0 or needed > len(sequences[name]):
                raise RuntimeError(
                    f"{resolution}: {name} quota {quotas[name]} exceeds capacity {capacity[name]}"
                )
            selected_parts.append(sequences[name][:needed])
            added_counts[name] = int(needed)
        indices = np.concatenate(selected_parts).astype(np.int32)
        if len(indices) != resolution or len(np.unique(indices)) != resolution:
            raise RuntimeError(f"{resolution}: support uniqueness/count failed")
        current = set(map(int, indices))
        if not anchor_set <= current:
            raise RuntimeError(f"{resolution}: canonical anchors not retained")
        if prior and not prior < current:
            raise RuntimeError(f"{resolution}: support is not strictly nested")
        prior = current
        selected_coords = coords[indices]
        selected_layers = layer_id[indices]
        interface_counts = [
            int(np.sum(np.isclose(selected_coords[:, 2], value, atol=1e-15)))
            for value in boundaries[1:-1]
        ]
        coverage = _source_box_coverage(
            indices=indices,
            coords=coords,
            layer_id=layer_id,
            groups=groups,
            layer_index=layer_index,
        )
        if (
            not coverage["all_source_boxes_covered"]
            or not all(interface_counts)
            or len(set(selected_layers.tolist())) != 9
        ):
            raise RuntimeError(f"{resolution}: source/layer/interface coverage failed")
        probes[str(resolution)] = {
            "probe_id": f"v6_p1h_source_aware_nested_{resolution}_v0",
            "node_count": resolution,
            "anchor_count": 1024,
            "indices": indices.tolist(),
            "indices_sha256": _array_sha256(indices),
            "coordinate_sha256": _array_sha256(selected_coords),
            "control_volume_sha256": _array_sha256(cv[indices]),
            "stratum_quotas": quotas,
            "stratum_ratios": RATIOS,
            "added_counts": added_counts,
            "metric_weight_policy": "equal_weight_fixed_source_aware_strata_v1",
            "metric_weights": np.full(resolution, 1.0 / resolution).tolist(),
            "layer_counts": {
                str(index): int(np.sum(selected_layers == index))
                for index in range(9)
            },
            "all_layers_covered": True,
            "interface_counts": interface_counts,
            "all_interfaces_covered": all(value > 0 for value in interface_counts),
            "top_count": int(np.sum(np.isclose(selected_coords[:, 2], np.max(z), atol=1e-15))),
            "bottom_count": int(np.sum(np.isclose(selected_coords[:, 2], np.min(z), atol=1e-15))),
            "source_box_coverage": coverage,
        }
    next_quotas = _quotas(NEXT_RESOLUTION)
    infeasible = {
        name: {
            "required": next_quotas[name],
            "capacity": capacity[name],
            "shortfall": max(0, next_quotas[name] - capacity[name]),
        }
        for name in RATIOS
    }
    if not any(row["shortfall"] > 0 for row in infeasible.values()):
        raise RuntimeError("expected next-resolution physical capacity limit was not found")
    return {
        "schema_version": "heat3d_v6_source_aware_resolution_ladder_v1",
        "status": "frozen_for_valid_iid_cpu_evaluation",
        "dataset_id": DATASET_ID,
        "solver_node_count": len(coords),
        "resolutions": list(RESOLUTIONS),
        "next_resolution": NEXT_RESOLUTION,
        "next_resolution_status": "infeasible_exact_stratum_ratio_on_unique_solver_nodes",
        "next_resolution_capacity_audit": infeasible,
        "seed": SEED,
        "strictly_nested": True,
        "canonical_1024_retained_exactly": True,
        "stratum_ratios": RATIOS,
        "selection_policy": (
            "canonical anchors plus frozen per-stratum/per-layer permutations; "
            "no temperature, prediction, error, or split-role input"
        ),
        "selection_inputs": [
            "canonical P1h anchor indices and stratum codes",
            "solver coordinates/layer IDs/interfaces",
            "registered source-allowed active layers and source-box geometry",
            "fixed RNG seed",
        ],
        "forbidden_selection_inputs": [
            "temperature or any target",
            "model predictions/errors",
            "split role",
            "test/hard identity",
        ],
        "source_box_audit_uses_registered_geometry_only": True,
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "manifest_sha256": MANIFEST_SHA256,
        "full_field_archive_sha256": ARCHIVE_SHA256,
        "anchor_indices_sha256": _array_sha256(anchors),
        "stratum_capacity": capacity,
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_p1g_geometry_deconfounded1024.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_source_aware_resolution_ladder.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build(args.dataset.resolve(), args.manifest.resolve(), args.config.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("source-aware resolution ladder is stale")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "resolutions": list(RESOLUTIONS),
                "next_resolution_status": payload["next_resolution_status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
