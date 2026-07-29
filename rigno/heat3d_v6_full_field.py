"""Layer/interface-aware reconstruction and full-field V6 metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


ALGORITHM = "layer_interface_knn_inverse_distance_v1"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReconstructionMap:
    support_indices: np.ndarray
    neighbor_local_indices: np.ndarray
    neighbor_weights: np.ndarray
    domain_code: np.ndarray
    domain_names: tuple[str, ...]

    def reconstruct(self, support_values: np.ndarray) -> np.ndarray:
        values = np.asarray(support_values, dtype=np.float64)
        if values.shape != (len(self.support_indices),):
            raise ValueError("support value shape does not match reconstruction map")
        gathered = values[self.neighbor_local_indices]
        return np.sum(gathered * self.neighbor_weights, axis=1)


def _domain_partition(
    coords: np.ndarray,
    layer_id: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    z = coords[:, 2]
    names = ["top", "bottom"]
    masks = [
        np.isclose(z, float(np.max(z)), atol=1.0e-15),
        np.isclose(z, float(np.min(z)), atol=1.0e-15),
    ]
    for index, value in enumerate(boundaries[1:-1], start=1):
        names.append(f"interface_{index:02d}")
        masks.append(np.isclose(z, float(value), atol=1.0e-15))
    reserved = np.any(np.stack(masks, axis=1), axis=1)
    for index in range(int(np.max(layer_id)) + 1):
        names.append(f"layer_{index:02d}")
        masks.append((layer_id == index) & ~reserved)
    code = np.full(len(coords), -1, dtype=np.int16)
    for index, mask in enumerate(masks):
        unassigned = mask & (code < 0)
        code[unassigned] = index
    if np.any(code < 0):
        raise RuntimeError("full-field domain partition left unassigned nodes")
    return code, tuple(names)


def build_reconstruction_map(
    *,
    coords: np.ndarray,
    layer_id: np.ndarray,
    boundaries: np.ndarray,
    support_indices: np.ndarray,
) -> tuple[ReconstructionMap, dict[str, Any]]:
    started = time.perf_counter()
    coords = np.asarray(coords, dtype=np.float64)
    layer_id = np.asarray(layer_id, dtype=np.int32)
    support_indices = np.asarray(support_indices, dtype=np.int32)
    if len(np.unique(support_indices)) != len(support_indices):
        raise ValueError("support indices are not unique")
    domain_code, domain_names = _domain_partition(coords, layer_id, boundaries)
    support_code = domain_code[support_indices]
    max_neighbors = 8
    neighbors = np.empty((len(coords), max_neighbors), dtype=np.int32)
    weights = np.zeros((len(coords), max_neighbors), dtype=np.float64)
    coverage: dict[str, Any] = {}
    for code, name in enumerate(domain_names):
        query_rows = np.flatnonzero(domain_code == code)
        support_local = np.flatnonzero(support_code == code)
        if not len(query_rows):
            continue
        if not len(support_local):
            raise RuntimeError(f"{name}: support domain is empty")
        dimensions = 2 if name.startswith(("top", "bottom", "interface")) else 3
        k = min(4 if dimensions == 2 else 8, len(support_local))
        query_coords = coords[query_rows]
        candidate_coords = coords[support_indices[support_local]]
        if dimensions == 2:
            query_coords = query_coords[:, :2]
            candidate_coords = candidate_coords[:, :2]
        tree = cKDTree(candidate_coords)
        distance, local_neighbor = tree.query(query_coords, k=k)
        if k == 1:
            distance = distance[:, None]
            local_neighbor = local_neighbor[:, None]
        selected = support_local[np.asarray(local_neighbor, dtype=np.int64)]
        exact = np.asarray(distance) <= 1.0e-15
        inverse = 1.0 / np.maximum(np.asarray(distance), 1.0e-15) ** 2
        domain_weights = inverse / np.sum(inverse, axis=1, keepdims=True)
        exact_rows = np.any(exact, axis=1)
        if np.any(exact_rows):
            domain_weights[exact_rows] = exact[exact_rows] / np.sum(
                exact[exact_rows], axis=1, keepdims=True
            )
        neighbors[query_rows, :k] = selected
        neighbors[query_rows, k:] = selected[:, :1]
        weights[query_rows, :k] = domain_weights
        coverage[name] = {
            "full_node_count": int(len(query_rows)),
            "support_node_count": int(len(support_local)),
            "neighbor_count": int(k),
            "maximum_nearest_distance_m": float(np.max(np.asarray(distance)[:, 0])),
        }
    if not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("reconstruction weights do not form a partition of unity")
    result = ReconstructionMap(
        support_indices=support_indices,
        neighbor_local_indices=neighbors,
        neighbor_weights=weights,
        domain_code=domain_code,
        domain_names=domain_names,
    )
    return result, {
        "algorithm": ALGORITHM,
        "label_independent": True,
        "selection_inputs": ["coords", "layer_id", "layer_boundaries", "support_indices"],
        "target_or_split_inputs": [],
        "build_seconds": float(time.perf_counter() - started),
        "full_node_count": int(len(coords)),
        "support_node_count": int(len(support_indices)),
        "domain_coverage": coverage,
        "partition_of_unity_max_abs_error": float(
            np.max(np.abs(np.sum(weights, axis=1) - 1.0))
        ),
    }


def save_reconstruction_map(path: Path, mapping: ReconstructionMap) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("wb") as handle:
        np.savez(
            handle,
            support_indices=mapping.support_indices,
            neighbor_local_indices=mapping.neighbor_local_indices,
            neighbor_weights=mapping.neighbor_weights,
            domain_code=mapping.domain_code,
            domain_names=np.asarray(mapping.domain_names, dtype="S64"),
        )
    return {
        "save_seconds": float(time.perf_counter() - started),
        "file_bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def load_reconstruction_map(path: Path) -> tuple[ReconstructionMap, dict[str, Any]]:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as payload:
        mapping = ReconstructionMap(
            support_indices=np.asarray(payload["support_indices"], dtype=np.int32),
            neighbor_local_indices=np.asarray(
                payload["neighbor_local_indices"], dtype=np.int32
            ),
            neighbor_weights=np.asarray(payload["neighbor_weights"], dtype=np.float64),
            domain_code=np.asarray(payload["domain_code"], dtype=np.int16),
            domain_names=tuple(
                value.decode("utf-8")
                for value in np.asarray(payload["domain_names"]).tolist()
            ),
        )
    return mapping, {
        "load_seconds": float(time.perf_counter() - started),
        "file_bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FullFieldMetricAccumulator:
    """Streaming CV-weighted prediction and sampling-floor metrics."""

    def __init__(
        self,
        *,
        control_volume: np.ndarray,
        layer_id: np.ndarray,
        boundaries: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        self.cv = np.asarray(control_volume, dtype=np.float64)
        self.layer_id = np.asarray(layer_id, dtype=np.int32)
        self.boundaries = np.asarray(boundaries, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.rows: dict[str, list[dict[str, float]]] = {"model": [], "sampling_floor": []}

    def add(
        self,
        *,
        kind: str,
        sample_id: str,
        prediction_delta: np.ndarray,
        truth_delta: np.ndarray,
        q: np.ndarray,
    ) -> None:
        prediction = np.asarray(prediction_delta, dtype=np.float64)
        truth = np.asarray(truth_delta, dtype=np.float64)
        error = prediction - truth
        cv = self.cv
        total_cv = float(np.sum(cv))
        truth_energy = float(np.sum(cv * truth * truth))
        sse = float(np.sum(cv * error * error))
        row: dict[str, float] = {
            "sample_id": sample_id,
            "sse_cv": sse,
            "truth_energy_cv": truth_energy,
            "rmse_K": float(np.sqrt(sse / total_cv)),
            "relative_rmse_pct": float(np.sqrt(sse / max(truth_energy, 1.0e-30)) * 100),
            "peak_error_K": float(np.max(prediction) - np.max(truth)),
        }
        source = np.asarray(q, dtype=np.float64) > 0.0
        row["source_sse_cv"] = float(np.sum(cv[source] * error[source] ** 2))
        row["source_cv"] = float(np.sum(cv[source]))
        top = np.isclose(self.coords[:, 2], np.max(self.coords[:, 2]), atol=1.0e-15)
        bottom = np.isclose(
            self.coords[:, 2], np.min(self.coords[:, 2]), atol=1.0e-15
        )
        for name, mask in (("top", top), ("bottom", bottom)):
            row[f"{name}_sse_cv"] = float(np.sum(cv[mask] * error[mask] ** 2))
            row[f"{name}_cv"] = float(np.sum(cv[mask]))
        for layer in range(int(np.max(self.layer_id)) + 1):
            mask = self.layer_id == layer
            row[f"layer_{layer:02d}_sse_cv"] = float(
                np.sum(cv[mask] * error[mask] ** 2)
            )
            row[f"layer_{layer:02d}_cv"] = float(np.sum(cv[mask]))
        for interface, value in enumerate(self.boundaries[1:-1], start=1):
            mask = np.isclose(self.coords[:, 2], float(value), atol=1.0e-15)
            row[f"interface_{interface:02d}_sse_cv"] = float(
                np.sum(cv[mask] * error[mask] ** 2)
            )
            row[f"interface_{interface:02d}_cv"] = float(np.sum(cv[mask]))
        self.rows[kind].append(row)

    def summarize(self, kind: str) -> dict[str, Any]:
        rows = self.rows[kind]
        if not rows:
            raise RuntimeError(f"no full-field metric rows for {kind}")
        total_sse = sum(float(row["sse_cv"]) for row in rows)
        total_energy = sum(float(row["truth_energy_cv"]) for row in rows)
        total_cv = float(np.sum(self.cv)) * len(rows)
        summary: dict[str, Any] = {
            "sample_count": len(rows),
            "full_node_count": int(len(self.cv)),
            "cv_weighted_rmse_K": float(np.sqrt(total_sse / total_cv)),
            "cv_weighted_point_global_relative_rmse_pct": float(
                np.sqrt(total_sse / total_energy) * 100
            ),
            "sample_first_cv_relative_rmse_pct": float(
                np.mean([row["relative_rmse_pct"] for row in rows])
            ),
            "peak_error_rmse_K": float(
                np.sqrt(np.mean([row["peak_error_K"] ** 2 for row in rows]))
            ),
        }
        for name in ("source", "top", "bottom"):
            sse = sum(float(row[f"{name}_sse_cv"]) for row in rows)
            measure = sum(float(row[f"{name}_cv"]) for row in rows)
            summary[f"{name}_cv_weighted_rmse_K"] = float(
                np.sqrt(sse / max(measure, 1.0e-30))
            )
        summary["layer_cv_weighted_rmse_K"] = {
            f"layer_{layer:02d}": float(
                np.sqrt(
                    sum(float(row[f"layer_{layer:02d}_sse_cv"]) for row in rows)
                    / sum(float(row[f"layer_{layer:02d}_cv"]) for row in rows)
                )
            )
            for layer in range(int(np.max(self.layer_id)) + 1)
        }
        summary["interface_cv_weighted_rmse_K"] = {
            f"interface_{interface:02d}": float(
                np.sqrt(
                    sum(
                        float(row[f"interface_{interface:02d}_sse_cv"])
                        for row in rows
                    )
                    / sum(
                        float(row[f"interface_{interface:02d}_cv"])
                        for row in rows
                    )
                )
            )
            for interface in range(1, len(self.boundaries) - 1)
        }
        return summary
