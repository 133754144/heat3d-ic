"""Label-aware evaluation core for the frozen V6 Heat3D metric contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


METRIC_SCHEMA_VERSION = "heat3d_v7_evaluation_core_v1_v6_definitions"
EVALUATION_SPLIT = "valid_iid"


@dataclass(frozen=True)
class EvaluationSample:
    """One prediction/truth pair at a single evaluation resolution."""

    sample_id: str
    prediction_deltaT_K: np.ndarray
    truth_deltaT_K: np.ndarray
    control_volumes_m3: np.ndarray
    coords: np.ndarray
    layer_id: np.ndarray
    q_W_m3: np.ndarray
    split: str = EVALUATION_SPLIT

    def validated(self) -> "EvaluationSample":
        if self.split != EVALUATION_SPLIT:
            raise ValueError(
                "EvaluationCore is restricted to frozen valid_iid; "
                f"got split={self.split!r}"
            )
        sample_id = str(self.sample_id)
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        prediction = _finite_vector(self.prediction_deltaT_K, "prediction_deltaT_K")
        truth = _finite_vector(self.truth_deltaT_K, "truth_deltaT_K")
        weights = _positive_vector(self.control_volumes_m3, "control_volumes_m3")
        coords = np.asarray(self.coords, dtype=np.float64)
        layer_id = np.asarray(self.layer_id, dtype=np.int32).reshape(-1)
        q = _finite_vector(self.q_W_m3, "q_W_m3")
        if coords.ndim != 2 or coords.shape != (prediction.size, 3):
            raise ValueError(
                f"coords must have shape {(prediction.size, 3)}, got {coords.shape}"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError("coords must be finite")
        for name, value in (
            ("truth_deltaT_K", truth),
            ("control_volumes_m3", weights),
            ("layer_id", layer_id),
            ("q_W_m3", q),
        ):
            if value.size != prediction.size:
                raise ValueError(
                    f"{name} count {value.size} does not match prediction count "
                    f"{prediction.size}"
                )
        return EvaluationSample(
            sample_id=sample_id,
            prediction_deltaT_K=prediction,
            truth_deltaT_K=truth,
            control_volumes_m3=weights,
            coords=coords,
            layer_id=layer_id,
            q_W_m3=q,
            split=EVALUATION_SPLIT,
        )


class EvaluationCore:
    """Single stable implementation of the frozen V6 evaluation contract."""

    split = EVALUATION_SPLIT

    @staticmethod
    def load_valid_truth(
        sample_ids: Sequence[str],
        truth_loader: Callable[[str], Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Load truth through an explicit valid-only provider."""

        result = []
        for sample_id in sample_ids:
            row = dict(truth_loader(str(sample_id)))
            if row.get("split", EVALUATION_SPLIT) != EVALUATION_SPLIT:
                raise ValueError("truth loader returned a non-valid_iid sample")
            row["sample_id"] = str(sample_id)
            row["split"] = EVALUATION_SPLIT
            result.append(row)
        return result

    def evaluate(
        self,
        samples: Sequence[EvaluationSample | Mapping[str, Any]],
        *,
        receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("EvaluationCore requires at least one valid_iid sample")
        rows = [
            _sample_row(
                sample.validated()
                if isinstance(sample, EvaluationSample)
                else _sample_from_mapping(sample).validated()
            )
            for sample in samples
        ]
        stats = _sum_sufficient_statistics(rows)
        result: dict[str, Any] = {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "evaluation_split": EVALUATION_SPLIT,
            "sample_count": len(rows),
            "per_sample": rows,
            "sufficient_statistics": stats,
            "metrics": _metrics_from_statistics(stats),
            "labels_read_by_inference": False,
        }
        if receipt is not None:
            result["receipt"] = dict(receipt)
        return result


def _sample_from_mapping(sample: Mapping[str, Any]) -> EvaluationSample:
    return EvaluationSample(
        sample_id=str(sample.get("sample_id") or ""),
        prediction_deltaT_K=sample.get("prediction_deltaT_K"),
        truth_deltaT_K=sample.get("truth_deltaT_K"),
        control_volumes_m3=sample.get("control_volumes_m3"),
        coords=sample.get("coords"),
        layer_id=sample.get("layer_id"),
        q_W_m3=sample.get("q_W_m3"),
        split=str(sample.get("split", EVALUATION_SPLIT)),
    )


def _sample_row(sample: EvaluationSample) -> dict[str, Any]:
    prediction = sample.prediction_deltaT_K
    truth = sample.truth_deltaT_K
    weights = sample.control_volumes_m3
    error = prediction - truth
    source = sample.q_W_m3 > 0.0
    background = ~source
    layer_means = []
    for layer in sorted(np.unique(sample.layer_id)):
        mask = sample.layer_id == layer
        layer_means.append(float(np.sum(weights[mask] * error[mask]) / np.sum(weights[mask])))
    interface_errors = np.diff(layer_means)
    point_sse = float(np.sum(error * error))
    point_truth_energy = float(np.sum(truth * truth))
    cv_sse = float(np.sum(weights * error * error))
    cv_truth_energy = float(np.sum(weights * truth * truth))
    cv_volume = float(np.sum(weights))
    source_sse = float(np.sum(weights[source] * error[source] ** 2))
    source_volume = float(np.sum(weights[source]))
    background_sse = float(np.sum(weights[background] * error[background] ** 2))
    background_volume = float(np.sum(weights[background]))
    peak_error = float(np.max(prediction) - np.max(truth))
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "point_count": int(prediction.size),
        "point_sse": point_sse,
        "point_truth_energy": point_truth_energy,
        "cv_sse": cv_sse,
        "cv_truth_energy": cv_truth_energy,
        "cv_volume": cv_volume,
        "sample_cv_relative_rmse": math.sqrt(cv_sse / cv_truth_energy),
        "source_sse": source_sse,
        "source_volume": source_volume,
        "background_sse": background_sse,
        "background_volume": background_volume,
        "peak_error_K": peak_error,
        "peak_error_squared": peak_error * peak_error,
        "interface_error_sum_squared": float(np.sum(interface_errors * interface_errors)),
        "interface_error_count": int(interface_errors.size),
        "interface_error_by_boundary": [float(value) for value in interface_errors],
    }


def _sum_sufficient_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def total(name: str) -> float:
        return float(sum(float(row[name]) for row in rows))

    interface_by_boundary: dict[str, dict[str, float | int]] = {}
    max_interfaces = max(len(row["interface_error_by_boundary"]) for row in rows)
    for index in range(max_interfaces):
        values = [
            float(row["interface_error_by_boundary"][index])
            for row in rows
            if index < len(row["interface_error_by_boundary"])
        ]
        interface_by_boundary[f"interface_{index + 1:02d}"] = {
            "sse": float(sum(value * value for value in values)),
            "count": len(values),
        }
    return {
        "point_sse": total("point_sse"),
        "point_truth_energy": total("point_truth_energy"),
        "point_count": int(sum(int(row["point_count"]) for row in rows)),
        "cv_sse": total("cv_sse"),
        "cv_truth_energy": total("cv_truth_energy"),
        "cv_volume": total("cv_volume"),
        "sample_cv_relative_rmse_sum": total("sample_cv_relative_rmse"),
        "sample_cv_relative_rmse_count": len(rows),
        "source_sse": total("source_sse"),
        "source_volume": total("source_volume"),
        "background_sse": total("background_sse"),
        "background_volume": total("background_volume"),
        "peak_error_squared": total("peak_error_squared"),
        "peak_error_count": len(rows),
        "interface_error_sum_squared": total("interface_error_sum_squared"),
        "interface_error_count": int(total("interface_error_count")),
        "interface_by_boundary": interface_by_boundary,
    }


def _metrics_from_statistics(stats: Mapping[str, Any]) -> dict[str, float]:
    point_global = math.sqrt(stats["point_sse"] / stats["point_truth_energy"]) * 100.0
    sample_first = (
        stats["sample_cv_relative_rmse_sum"]
        / stats["sample_cv_relative_rmse_count"]
        * 100.0
    )
    raw_cv = math.sqrt(stats["cv_sse"] / stats["cv_volume"])
    source = math.sqrt(stats["source_sse"] / stats["source_volume"])
    peak = math.sqrt(stats["peak_error_squared"] / stats["peak_error_count"])
    interface = math.sqrt(
        stats["interface_error_sum_squared"] / stats["interface_error_count"]
    )
    return {
        "point_global_relative_rmse_pct": point_global,
        "sample_first_relative_rmse_pct": sample_first,
        "sample_first_cv_relative_rmse_pct": sample_first,
        "raw_K_CV_RMSE_K": raw_cv,
        "raw_cv_weighted_rmse_K": raw_cv,
        "source_region_RMSE_K": source,
        "source_rmse_K": source,
        "peak_RMSE_K": peak,
        "peak_rmse_K": peak,
        "interface_RMSE_K": interface,
        "interface_drop_rmse_K": interface,
    }


def _finite_vector(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _positive_vector(value: Any, name: str) -> np.ndarray:
    array = _finite_vector(value, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array
