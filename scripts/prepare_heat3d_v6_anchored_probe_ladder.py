#!/usr/bin/env python3
"""Prepare the label-independent anchored V6 high-resolution probe ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (1024, 2048, 4096, 8192)
SEED = 2026072701
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
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


def _systematic_pps_order(weights: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Select ``count`` unique complement nodes by circular systematic PPS."""

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(weights))
    ordered = weights[permutation]
    total = float(np.sum(ordered))
    spacing = total / count
    if float(np.max(ordered)) >= spacing:
        raise RuntimeError("PPS no-certainty-unit contract failed")
    offset = float(rng.random() * spacing)
    thresholds = offset + np.arange(count, dtype=np.float64) * spacing
    positions = np.searchsorted(np.cumsum(ordered), thresholds, side="right")
    selected = permutation[positions]
    if len(np.unique(selected)) != count:
        raise RuntimeError("PPS selection contains duplicates")
    return selected.astype(np.int32)


def build(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_ids = [
        str(row["sample_id"])
        for row in manifest["samples"]
        if row["split_role"] == "valid"
    ]
    if manifest["dataset_id"] != DATASET_ID or len(valid_ids) != 128:
        raise RuntimeError("P1h/valid_iid binding drifted")
    archive = dataset_root / "full_fields.h5"
    if _sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("full-field archive SHA256 mismatch")
    with h5py.File(archive, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        cv = np.asarray(handle["mesh/control_volume"], dtype=np.float64)
        layer = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
        anchors = np.asarray(handle["support/indices"], dtype=np.int32)
    if anchors.shape != (1024,) or len(np.unique(anchors)) != 1024:
        raise RuntimeError("frozen P1h anchor support drifted")

    complement = np.setdiff1d(
        np.arange(len(coords), dtype=np.int32), anchors, assume_unique=False
    )
    master_extra = _systematic_pps_order(
        cv[complement], max(RESOLUTIONS) - 1024, SEED
    )
    master_extra = complement[master_extra]
    probes: dict[str, Any] = {}
    previous: set[int] = set()
    for resolution in RESOLUTIONS:
        extra_count = resolution - 1024
        indices = np.concatenate((anchors, master_extra[:extra_count])).astype(np.int32)
        current = set(indices.tolist())
        if previous and not previous < current:
            raise RuntimeError("anchored supports are not strictly nested")
        previous = current
        selected_coords = coords[indices]
        selected_layers = layer[indices]
        # The 1024 anchor measure is the frozen training/evaluation measure.
        # Added complement nodes use a PPS expansion measure. The evaluator
        # reports both this query measure and the canonical 1024 anchor result.
        if extra_count:
            expansion = np.concatenate(
                (
                    cv[anchors],
                    np.full(
                        extra_count,
                        float(np.sum(cv[complement])) / extra_count,
                        dtype=np.float64,
                    ),
                )
            )
            metric_policy = "anchor_cv_plus_complement_pps_expansion_v1"
        else:
            expansion = np.full(1024, 1.0 / 1024.0, dtype=np.float64)
            metric_policy = "canonical_equal_weight_frozen_p1h_anchor_measure"
        layer_counts = {
            str(index): int(np.sum(selected_layers == index)) for index in range(9)
        }
        interface_counts = [
            int(np.sum(np.isclose(selected_coords[:, 2], z, atol=1.0e-15)))
            for z in boundaries[1:-1]
        ]
        probes[str(resolution)] = {
            "probe_id": f"v6_p1h_anchored_volume_pps_{resolution}_v0",
            "node_count": resolution,
            "anchor_count": 1024,
            "added_query_count": extra_count,
            "indices": indices.tolist(),
            "indices_sha256": _array_sha256(indices),
            "coordinate_sha256": _array_sha256(selected_coords),
            "layer_id_sha256": _array_sha256(selected_layers),
            "metric_weights": expansion.tolist(),
            "metric_weights_sha256": _array_sha256(expansion),
            "metric_weight_policy": metric_policy,
            "layer_counts": layer_counts,
            "interface_counts": interface_counts,
            "top_count": int(np.sum(np.isclose(selected_coords[:, 2], coords[:, 2].max()))),
            "bottom_count": int(np.sum(np.isclose(selected_coords[:, 2], coords[:, 2].min()))),
            "all_layers_covered": all(value > 0 for value in layer_counts.values()),
            "all_interfaces_covered": all(value > 0 for value in interface_counts),
        }
    return {
        "schema_version": "heat3d_v6_anchored_probe_ladder_v1",
        "status": "frozen_for_valid_iid_cpu_evaluation",
        "dataset_id": DATASET_ID,
        "evaluation_role": "valid_iid",
        "conditioning_support": "frozen_source_aware_p1h_1024_anchors",
        "query_support": "anchors_plus_label_independent_volume_pps_nodes",
        "global_context_source": "anchors_only",
        "seed": SEED,
        "selection_inputs": [
            "frozen P1h support indices",
            "solver control volume",
            "frozen RNG seed",
        ],
        "forbidden_selection_inputs": [
            "temperature",
            "q/source layout",
            "split labels except valid-only access guard",
            "model prediction/error",
        ],
        "label_independent": True,
        "test_hard_accessed": False,
        "training_executed": False,
        "manifest_sha256": MANIFEST_SHA256,
        "full_field_archive_sha256": ARCHIVE_SHA256,
        "anchor_indices_sha256": _array_sha256(anchors),
        "valid_sample_ids_sha256": hashlib.sha256(
            "\n".join(valid_ids).encode()
        ).hexdigest(),
        "strictly_nested": True,
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
        "--output",
        type=Path,
        default=ROOT / "configs/heat3d_v6/v6_anchored_probe_ladder.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build(args.dataset.resolve(), args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("anchored probe ladder is stale")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": "passed", "resolutions": list(RESOLUTIONS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
