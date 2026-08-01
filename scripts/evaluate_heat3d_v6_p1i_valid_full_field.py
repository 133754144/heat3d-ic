#!/usr/bin/env python3
"""Valid-only P1i 1024-support to 240825-node full-field evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v6_full_field import build_reconstruction_map


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _source_mask(coords: np.ndarray, layer_id: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    layer_names = [row["id"] for row in meta["physics"]["layers_bottom_to_top"]]
    mask = np.zeros(len(coords), dtype=bool)
    x_fraction = coords[:, 0] / float(meta["physics"]["footprint_m"][0])
    y_fraction = coords[:, 1] / float(meta["physics"]["footprint_m"][1])
    for block in meta["q_blocks"]:
        x0, x1, y0, y1 = map(float, block["bbox_fraction_xy"])
        current = (
            (layer_id == layer_names.index(block["layer"]))
            & (x_fraction >= x0 - 1.0e-12)
            & (x_fraction <= x1 + 1.0e-12)
            & (y_fraction >= y0 - 1.0e-12)
            & (y_fraction <= y1 + 1.0e-12)
        )
        mask |= current
    if not np.any(mask):
        raise RuntimeError(f"{meta['sample_id']}: reconstructed source mask is empty")
    return mask


def _boundaries(meta: dict[str, Any], minimum_z: float) -> np.ndarray:
    thicknesses = [
        float(row["thickness_m"])
        for row in meta["physics"]["layers_bottom_to_top"]
    ]
    return minimum_z + np.concatenate(([0.0], np.cumsum(thicknesses)))


class Metrics:
    def __init__(self, cv: np.ndarray, coords: np.ndarray, layer_id: np.ndarray, boundaries: np.ndarray):
        self.cv = np.asarray(cv, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.layer_id = np.asarray(layer_id, dtype=np.int32)
        self.boundaries = np.asarray(boundaries, dtype=np.float64)
        self.rows: list[dict[str, Any]] = []

    def add(self, sample_id: str, prediction: np.ndarray, truth: np.ndarray, source: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        error = prediction - truth
        cv = self.cv
        low_background = (~source) & (truth <= np.quantile(truth, 0.5))
        top = np.isclose(self.coords[:, 2], np.max(self.coords[:, 2]), atol=1e-15)
        bottom = np.isclose(self.coords[:, 2], np.min(self.coords[:, 2]), atol=1e-15)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "sse": float(np.sum(error**2)),
            "energy": float(np.sum(truth**2)),
            "cv_sse": float(np.sum(cv * error**2)),
            "cv_energy": float(np.sum(cv * truth**2)),
            "cv_measure": float(np.sum(cv)),
            "sample_first_cv_relative_rmse_pct": float(
                100.0 * math.sqrt(np.sum(cv * error**2) / np.sum(cv * truth**2))
            ),
            "peak_error_K": float(np.max(prediction) - np.max(truth)),
        }
        for name, mask in (
            ("source", source),
            ("background", ~source),
            ("low_deltaT_background", low_background),
            ("top", top),
            ("bottom", bottom),
        ):
            row[f"{name}_cv_sse"] = float(np.sum(cv[mask] * error[mask] ** 2))
            row[f"{name}_cv_measure"] = float(np.sum(cv[mask]))
            row[f"{name}_cv_bias_numerator"] = float(np.sum(cv[mask] * error[mask]))
        true_layer_means, predicted_layer_means = [], []
        for layer in range(9):
            mask = self.layer_id == layer
            true_layer_means.append(_weighted(truth, cv, mask))
            predicted_layer_means.append(_weighted(prediction, cv, mask))
        layer_mean_error = np.asarray(predicted_layer_means) - np.asarray(true_layer_means)
        row["layer_mean_squared_error_sum"] = float(np.sum(layer_mean_error**2))
        row["layer_mean_count"] = int(len(layer_mean_error))
        layer_drop_error = np.diff(predicted_layer_means) - np.diff(true_layer_means)
        row["layer_drop_squared_error_sum"] = float(np.sum(layer_drop_error**2))
        row["layer_drop_count"] = int(len(layer_drop_error))
        interface_errors = []
        for boundary in self.boundaries[1:-1]:
            mask = np.isclose(self.coords[:, 2], boundary, atol=1e-15)
            interface_errors.append(_weighted(error, cv, mask))
        row["interface_mean_squared_error_sum"] = float(np.sum(np.square(interface_errors)))
        row["interface_count"] = int(len(interface_errors))
        self.rows.append(row)

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        sse = sum(row["sse"] for row in rows)
        energy = sum(row["energy"] for row in rows)
        cv_sse = sum(row["cv_sse"] for row in rows)
        cv_energy = sum(row["cv_energy"] for row in rows)
        cv_measure = sum(row["cv_measure"] for row in rows)
        result: dict[str, Any] = {
            "sample_count": len(rows),
            "full_node_count": len(self.cv),
            "point_global_true_rms_relative_rmse_pct": float(100 * math.sqrt(sse / energy)),
            "point_global_cv_relative_rmse_pct": float(100 * math.sqrt(cv_sse / cv_energy)),
            "sample_first_cv_relative_rmse_pct": float(
                np.mean([row["sample_first_cv_relative_rmse_pct"] for row in rows])
            ),
            "raw_cv_weighted_rmse_K": float(math.sqrt(cv_sse / cv_measure)),
            "peak_rmse_K": float(math.sqrt(np.mean([row["peak_error_K"] ** 2 for row in rows]))),
        }
        for name in ("source", "background", "low_deltaT_background", "top", "bottom"):
            region_sse = sum(row[f"{name}_cv_sse"] for row in rows)
            measure = sum(row[f"{name}_cv_measure"] for row in rows)
            bias = sum(row[f"{name}_cv_bias_numerator"] for row in rows)
            result[name] = {
                "cv_weighted_rmse_K": float(math.sqrt(region_sse / measure)),
                "cv_weighted_bias_K": float(bias / measure),
            }
        for name in ("layer_mean", "layer_drop", "interface_mean"):
            result[f"{name}_rmse_K"] = float(
                math.sqrt(
                    sum(row[f"{name}_squared_error_sum"] for row in rows)
                    / sum(row[f"{name}_count"] for row in rows)
                )
            )
        worst = sorted(rows, key=lambda row: row["cv_sse"], reverse=True)[:10]
        result["worst_samples_by_cv_sse"] = [
            {
                "sample_id": row["sample_id"],
                "cv_sse": row["cv_sse"],
                "sample_first_cv_relative_rmse_pct": row["sample_first_cv_relative_rmse_pct"],
            }
            for row in worst
        ]
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-fields", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, help="label=path")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    valid_rows = [row for row in manifest["samples"] if row["split_role"] == "valid_iid"]
    if len(valid_rows) != 128:
        raise RuntimeError("valid_iid must contain exactly 128 samples")
    prediction_sets: dict[str, Any] = {}
    prediction_paths: dict[str, str] = {}
    for value in args.predictions:
        label, path_text = value.split("=", 1)
        path = Path(path_text)
        prediction_sets[label] = np.load(path, allow_pickle=False)
        prediction_paths[label] = str(path)
    with h5py.File(args.full_fields, "r") as archive:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in archive["samples/sample_id"][:]]
        index_by_id = {sample_id: index for index, sample_id in enumerate(ids)}
        full_coords = np.asarray(archive["shared/coords_m"][:], dtype=np.float64)
        full_cv = np.asarray(archive["shared/control_volume_m3"][:], dtype=np.float64)
        full_layer = np.asarray(archive["shared/layer_id"][:], dtype=np.int32)
        floor_metrics = None
        model_metrics: dict[str, Metrics] = {}
        map_audits = []
        full_tree = cKDTree(full_coords)
        for row_index, row in enumerate(valid_rows):
            sample_id = row["sample_id"]
            sample_dir = args.dataset_root / row["relative_path"]
            meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
            support_coords = np.asarray(np.load(sample_dir / "coords.npy"), dtype=np.float64)
            support_truth = np.asarray(np.load(sample_dir / "deltaT.npy"), dtype=np.float64).reshape(-1)
            distance, support_indices = full_tree.query(support_coords, k=1)
            if float(np.max(distance)) > 1.0e-14:
                raise RuntimeError(f"{sample_id}: support is not a subset of solver nodes")
            boundaries = _boundaries(meta, float(np.min(full_coords[:, 2])))
            mapping, map_audit = build_reconstruction_map(
                coords=full_coords,
                layer_id=full_layer,
                boundaries=boundaries,
                support_indices=np.asarray(support_indices, dtype=np.int32),
                empty_domain_fallback="same_layer",
            )
            if row_index in (0, len(valid_rows) - 1):
                map_audits.append({"sample_id": sample_id, **map_audit})
            truth = np.asarray(archive["samples/deltaT_K"][index_by_id[sample_id]], dtype=np.float64)
            source = _source_mask(full_coords, full_layer, meta)
            if floor_metrics is None:
                floor_metrics = Metrics(full_cv, full_coords, full_layer, boundaries)
            reconstructed_truth = mapping.reconstruct(support_truth)
            floor_metrics.add(sample_id, reconstructed_truth, truth, source)
            for label, predictions in prediction_sets.items():
                if sample_id not in predictions.files:
                    raise RuntimeError(f"{label}: missing valid sample {sample_id}")
                if label not in model_metrics:
                    model_metrics[label] = Metrics(full_cv, full_coords, full_layer, boundaries)
                predicted_temperature = np.asarray(predictions[sample_id], dtype=np.float64).reshape(-1)
                reference = float(meta["physics"]["ambient_K"])
                reconstructed_prediction = mapping.reconstruct(predicted_temperature - reference)
                model_metrics[label].add(sample_id, reconstructed_prediction, truth, source)
    for predictions in prediction_sets.values():
        predictions.close()
    payload = {
        "schema_version": "heat3d_v6_p1i_valid_full_field_v1",
        "accessed_roles": ["valid_iid"],
        "test_accessed": False,
        "sealed_accessed": False,
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": _sha256(args.manifest),
        "full_field_archive_sha256": _sha256(args.full_fields),
        "prediction_paths": prediction_paths,
        "reconstruction_only_sampling_floor": floor_metrics.summary(),
        "model_plus_reconstruction": {label: metrics.summary() for label, metrics in model_metrics.items()},
        "reconstruction_map_audit_examples": map_audits,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["kind", "point_global_true_rms_relative_rmse_pct", "point_global_cv_relative_rmse_pct", "sample_first_cv_relative_rmse_pct", "raw_cv_weighted_rmse_K", "peak_rmse_K", "source_rmse_K", "background_rmse_K", "layer_mean_rmse_K", "layer_drop_rmse_K", "interface_mean_rmse_K", "top_rmse_K", "bottom_rmse_K"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, summary in [("reconstruction_only", payload["reconstruction_only_sampling_floor"]), *payload["model_plus_reconstruction"].items()]:
            writer.writerow({
                "kind": label,
                "point_global_true_rms_relative_rmse_pct": summary["point_global_true_rms_relative_rmse_pct"],
                "point_global_cv_relative_rmse_pct": summary["point_global_cv_relative_rmse_pct"],
                "sample_first_cv_relative_rmse_pct": summary["sample_first_cv_relative_rmse_pct"],
                "raw_cv_weighted_rmse_K": summary["raw_cv_weighted_rmse_K"],
                "peak_rmse_K": summary["peak_rmse_K"],
                "source_rmse_K": summary["source"]["cv_weighted_rmse_K"],
                "background_rmse_K": summary["background"]["cv_weighted_rmse_K"],
                "layer_mean_rmse_K": summary["layer_mean_rmse_K"],
                "layer_drop_rmse_K": summary["layer_drop_rmse_K"],
                "interface_mean_rmse_K": summary["interface_mean_rmse_K"],
                "top_rmse_K": summary["top"]["cv_weighted_rmse_K"],
                "bottom_rmse_K": summary["bottom"]["cv_weighted_rmse_K"],
            })
    lines = [
        "# P1i old seed0 valid full-field closeout",
        "",
        "Only `valid_iid` was read. Test and sealed IID remained closed. The oracle row",
        "is the reconstruction-only sampling floor from exact original 1024 solver nodes;",
        "model rows combine prediction error and the same 240825-node reconstruction.",
        "",
        "| kind | point-global true-RMS % | sample-first CV % | raw CV RMSE K | peak RMSE K | source/background K | layer mean/drop K | interface K | top/bottom K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = [("reconstruction_only", payload["reconstruction_only_sampling_floor"]), *payload["model_plus_reconstruction"].items()]
    for label, summary in summaries:
        lines.append(
            f"| {label} | {summary['point_global_true_rms_relative_rmse_pct']:.6f} | "
            f"{summary['sample_first_cv_relative_rmse_pct']:.6f} | "
            f"{summary['raw_cv_weighted_rmse_K']:.6f} | {summary['peak_rmse_K']:.6f} | "
            f"{summary['source']['cv_weighted_rmse_K']:.6f}/{summary['background']['cv_weighted_rmse_K']:.6f} | "
            f"{summary['layer_mean_rmse_K']:.6f}/{summary['layer_drop_rmse_K']:.6f} | "
            f"{summary['interface_mean_rmse_K']:.6f} | "
            f"{summary['top']['cv_weighted_rmse_K']:.6f}/{summary['bottom']['cv_weighted_rmse_K']:.6f} |"
        )
    lines.extend(["", "Worst samples are recorded in the machine-readable JSON.", ""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
