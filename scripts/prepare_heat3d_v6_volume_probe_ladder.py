#!/usr/bin/env python3
"""Prepare nested label-independent volume-representative V6 solver probes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTANCE = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_acceptance.json"
)
DEFAULT_MANIFEST = (
    ROOT / "configs/heat3d_v6/v6_p1h_shared_support1024_manifest.json"
)
DEFAULT_LADDER = (
    ROOT / "configs/heat3d_v6/v6_volume_representative_probe_ladder.json"
)
DEFAULT_PROBE4096 = (
    ROOT / "configs/heat3d_v6/v6_volume_representative_probe4096.json"
)
DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
RESOLUTIONS = (1024, 2048, 4096, 8192)
MASTER_RESOLUTION = max(RESOLUTIONS)
SEED = 2026072501
MANIFEST_SHA256 = (
    "324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5"
)
ARCHIVE_SHA256 = (
    "f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _valid_ids_sha256(manifest: dict[str, Any]) -> str:
    valid_ids = [
        str(row["sample_id"])
        for row in manifest["samples"]
        if row["split_role"] == "valid"
    ]
    if len(valid_ids) != 128:
        raise RuntimeError("valid_iid sample count drifted")
    return hashlib.sha256("\n".join(valid_ids).encode("utf-8")).hexdigest()


def _circular_systematic_pps(
    control_volume: np.ndarray,
    *,
    resolution: int,
    master_offset_m3: float,
    permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(control_volume, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    if (
        order.shape != (len(weights),)
        or len(np.unique(order)) != len(weights)
        or int(np.min(order)) != 0
        or int(np.max(order)) != len(weights) - 1
    ):
        raise RuntimeError("PPS permutation is invalid")
    total = float(np.sum(weights))
    if np.any(weights <= 0.0) or not np.isfinite(total):
        raise RuntimeError("solver control volumes must be finite and positive")
    spacing = total / float(resolution)
    if float(np.max(weights)) >= spacing:
        raise RuntimeError(
            "PPS no-certainty-unit contract failed: max CV exceeds spacing"
        )
    master_spacing = total / float(MASTER_RESOLUTION)
    stride = MASTER_RESOLUTION // resolution
    slots = np.arange(0, MASTER_RESOLUTION, stride, dtype=np.int64)
    thresholds = np.mod(
        master_offset_m3 + slots.astype(np.float64) * master_spacing,
        total,
    )
    positions = np.searchsorted(
        np.cumsum(weights[order]), np.sort(thresholds), side="right"
    )
    indices = order[positions].astype(np.int32)
    indices = np.sort(indices)
    if indices.shape != (resolution,) or len(np.unique(indices)) != resolution:
        raise RuntimeError(f"resolution {resolution}: PPS selection is not unique")
    inclusion = resolution * weights[indices] / total
    expansion = weights[indices] / inclusion
    if np.any(inclusion <= 0.0) or np.any(inclusion >= 1.0):
        raise RuntimeError(f"resolution {resolution}: invalid inclusion probability")
    if not np.allclose(expansion, total / resolution, rtol=1.0e-13, atol=0.0):
        raise RuntimeError(f"resolution {resolution}: expansion weights drifted")
    return indices, inclusion, expansion


def _coverage(
    *,
    coords: np.ndarray,
    layer_id: np.ndarray,
    control_volume: np.ndarray,
    boundaries: np.ndarray,
    indices: np.ndarray,
    expansion: np.ndarray,
) -> dict[str, Any]:
    selected_coords = coords[indices]
    selected_layers = layer_id[indices]
    total_volume = float(np.sum(control_volume))
    layer_rows = []
    for layer in range(9):
        mask_all = layer_id == layer
        mask_selected = selected_layers == layer
        actual = float(np.sum(control_volume[mask_all]) / total_volume)
        estimated = float(np.sum(expansion[mask_selected]) / total_volume)
        layer_rows.append(
            {
                "layer_index": layer,
                "point_count": int(np.sum(mask_selected)),
                "actual_volume_fraction": actual,
                "estimated_volume_fraction": estimated,
                "absolute_fraction_error": abs(estimated - actual),
            }
        )
    interfaces = []
    for index, boundary in enumerate(boundaries[1:-1]):
        interfaces.append(
            {
                "interface_index": index,
                "z_m": float(boundary),
                "point_count": int(
                    np.sum(
                        np.isclose(
                            selected_coords[:, 2],
                            float(boundary),
                            atol=1.0e-15,
                        )
                    )
                ),
            }
        )
    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))
    return {
        "all_layers_covered": all(row["point_count"] > 0 for row in layer_rows),
        "all_interfaces_covered": all(
            row["point_count"] > 0 for row in interfaces
        ),
        "layer_distribution": layer_rows,
        "max_layer_volume_fraction_error": max(
            row["absolute_fraction_error"] for row in layer_rows
        ),
        "interfaces": interfaces,
        "top_point_count": int(
            np.sum(np.isclose(selected_coords[:, 2], z_max, atol=1.0e-15))
        ),
        "bottom_point_count": int(
            np.sum(np.isclose(selected_coords[:, 2], z_min, atol=1.0e-15))
        ),
        "selected_layer_counts": {
            str(key): int(value)
            for key, value in sorted(
                Counter(selected_layers.tolist()).items()
            )
        },
    }


def build(
    acceptance_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if acceptance["dataset_id"] != DATASET_ID or manifest["dataset_id"] != DATASET_ID:
        raise RuntimeError("P1h dataset binding drifted")
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("P1h manifest SHA256 mismatch")
    dataset_root = Path(acceptance["data_paths"]["durable_copy"])
    archive_path = dataset_root / "full_fields.h5"
    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise RuntimeError("P1h full-field archive SHA256 mismatch")

    with h5py.File(archive_path, "r") as handle:
        coords = np.asarray(handle["mesh/coords"], dtype=np.float64)
        layer_id = np.asarray(handle["mesh/layer_id"], dtype=np.int32)
        control_volume = np.asarray(
            handle["mesh/control_volume"], dtype=np.float64
        )
        boundaries = np.asarray(handle["mesh/boundaries"], dtype=np.float64)
    if coords.shape != (240825, 3) or layer_id.shape != (240825,):
        raise RuntimeError("P1h solver mesh shape drifted")

    total_volume = float(np.sum(control_volume))
    rng = np.random.default_rng(SEED)
    master_offset = float(rng.random() * total_volume)
    permutation = rng.permutation(len(coords)).astype(np.int32)
    probes: dict[str, Any] = {}
    prior_indices: set[int] = set()
    for resolution in RESOLUTIONS:
        indices, inclusion, expansion = _circular_systematic_pps(
            control_volume,
            resolution=resolution,
            master_offset_m3=master_offset,
            permutation=permutation,
        )
        current = set(indices.tolist())
        if prior_indices and not prior_indices < current:
            raise RuntimeError("resolution ladder supports are not strictly nested")
        prior_indices = current
        selected_coords = coords[indices]
        selected_layers = layer_id[indices]
        selected_cv = control_volume[indices]
        coverage = _coverage(
            coords=coords,
            layer_id=layer_id,
            control_volume=control_volume,
            boundaries=boundaries,
            indices=indices,
            expansion=expansion,
        )
        if (
            not coverage["all_layers_covered"]
            or not coverage["all_interfaces_covered"]
            or coverage["top_point_count"] == 0
            or coverage["bottom_point_count"] == 0
        ):
            raise RuntimeError(f"resolution {resolution}: required coverage failed")
        probes[str(resolution)] = {
            "schema_version": "heat3d_v6_volume_probe_v1",
            "status": "prepared_not_evaluated",
            "probe_id": f"v6_volume_representative_solver_node{resolution}_v0",
            "dataset_id": DATASET_ID,
            "evaluation_role": "valid_iid",
            "sample_count": 128,
            "node_count": resolution,
            "solver_node_count": len(coords),
            "seed": SEED,
            "selection_policy": "nested_circular_systematic_pps_by_control_volume_v1",
            "selection_inputs": [
                "mesh.control_volume",
                "solver node index under a frozen random permutation",
                "mesh.coords for coverage audit only",
                "mesh.layer_id for coverage audit only",
                "mesh.boundaries for coverage audit only",
            ],
            "forbidden_selection_inputs": [
                "temperature",
                "q",
                "source metadata or source layout",
                "sample split labels",
                "model predictions",
                "model errors",
            ],
            "label_independent": True,
            "source_dense_quota_fraction": 0.0,
            "test_hard_accessed": False,
            "training_executed": False,
            "inference_executed": False,
            "manifest_sha256": MANIFEST_SHA256,
            "full_field_archive_sha256": ARCHIVE_SHA256,
            "valid_sample_ids_sha256": _valid_ids_sha256(manifest),
            "master_offset_m3": master_offset,
            "solver_index_permutation_sha256": _array_sha256(permutation),
            "total_control_volume_m3": total_volume,
            "pps_spacing_m3": total_volume / resolution,
            "inclusion_probability_formula": "pi_i=n*cv_i/sum_all_solver_cv",
            "expansion_weight_formula": "w_i=cv_i/pi_i=sum_all_solver_cv/n",
            "metric_weight_policy": (
                "Horvitz-Thompson expansion weights estimate full solver "
                "control-volume integrals"
            ),
            "indices": indices.tolist(),
            "inclusion_probabilities": inclusion.tolist(),
            "expansion_weights_m3": expansion.tolist(),
            "support_index_sha256": _array_sha256(indices),
            "coordinate_sha256": _array_sha256(selected_coords),
            "layer_id_sha256": _array_sha256(selected_layers),
            "control_volume_sha256": _array_sha256(selected_cv),
            "inclusion_probability_sha256": _array_sha256(inclusion),
            "expansion_weight_sha256": _array_sha256(expansion),
            "coverage": coverage,
        }
    return {
        "schema_version": "heat3d_v6_volume_probe_ladder_v1",
        "status": "prepared_not_evaluated",
        "dataset_id": DATASET_ID,
        "resolutions": list(RESOLUTIONS),
        "selection_policy": "nested_circular_systematic_pps_by_control_volume_v1",
        "supports_are_nested": True,
        "label_independent": True,
        "source_dense_quota_fraction": 0.0,
        "evaluation_role": "valid_iid",
        "test_hard_accessed": False,
        "training_executed": False,
        "formal_inference_executed": False,
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ladder-output", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--probe4096-output", type=Path, default=DEFAULT_PROBE4096)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build(args.acceptance.resolve(), args.manifest.resolve())
    probe4096 = dict(payload["probes"]["4096"])
    probe4096["ladder_parent"] = str(DEFAULT_LADDER.relative_to(ROOT))
    if args.write:
        args.ladder_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.probe4096_output.write_text(
            json.dumps(probe4096, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "resolutions": payload["resolutions"],
                "supports_are_nested": payload["supports_are_nested"],
                "coverage": {
                    key: {
                        "layers": value["coverage"]["all_layers_covered"],
                        "interfaces": value["coverage"]["all_interfaces_covered"],
                        "top": value["coverage"]["top_point_count"],
                        "bottom": value["coverage"]["bottom_point_count"],
                        "max_layer_fraction_error": value["coverage"][
                            "max_layer_volume_fraction_error"
                        ],
                    }
                    for key, value in payload["probes"].items()
                },
                "test_hard_accessed": payload["test_hard_accessed"],
                "formal_inference_executed": payload["formal_inference_executed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
