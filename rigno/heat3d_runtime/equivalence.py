"""Old/new runtime numerical-equivalence primitives.

The default for checkpoint interpretation, normalization, feature assembly,
graph metadata, and model-input tensors is exact equality.  Prediction
comparison defaults to the existing V6 adapter/reference 1e-6 K tolerance;
callers may pass a stricter tolerance, but this module never widens one
implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TensorComparison:
    name: str
    shape_match: bool
    max_abs: float | None
    rmse: float | None
    max_abs_tolerance: float
    rmse_tolerance: float
    passed: bool


@dataclass(frozen=True)
class ComparisonReport:
    comparisons: tuple[TensorComparison, ...]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(item.passed for item in self.comparisons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "comparisons": [
                {
                    "name": item.name,
                    "shape_match": item.shape_match,
                    "max_abs": item.max_abs,
                    "rmse": item.rmse,
                    "max_abs_tolerance": item.max_abs_tolerance,
                    "rmse_tolerance": item.rmse_tolerance,
                    "passed": item.passed,
                }
                for item in self.comparisons
            ],
        }


def compare_metadata(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare JSON-safe checkpoint/runtime interpretation metadata exactly."""

    keys = tuple(sorted(set(old) | set(new)))
    changed = {
        key: {"old": old.get(key), "new": new.get(key)}
        for key in keys
        if key not in old or key not in new or old[key] != new[key]
    }
    return {"passed": not changed, "changed": changed}


def compare_named_arrays(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    max_abs_tolerance: float = 0.0,
    rmse_tolerance: float = 0.0,
    per_name_tolerance: Mapping[str, tuple[float, float]] | None = None,
) -> ComparisonReport:
    """Compare named tensors and report actual max-abs error and RMSE."""

    names = tuple(sorted(set(old) | set(new)))
    rows = []
    for name in names:
        if name not in old or name not in new:
            rows.append(
                TensorComparison(
                    name=name,
                    shape_match=False,
                    max_abs=None,
                    rmse=None,
                    max_abs_tolerance=max_abs_tolerance,
                    rmse_tolerance=rmse_tolerance,
                    passed=False,
                )
            )
            continue
        left = np.asarray(old[name])
        right = np.asarray(new[name])
        tolerance = (max_abs_tolerance, rmse_tolerance)
        if per_name_tolerance and name in per_name_tolerance:
            tolerance = tuple(float(value) for value in per_name_tolerance[name])
        if left.shape != right.shape:
            rows.append(
                TensorComparison(
                    name=name,
                    shape_match=False,
                    max_abs=None,
                    rmse=None,
                    max_abs_tolerance=tolerance[0],
                    rmse_tolerance=tolerance[1],
                    passed=False,
                )
            )
            continue
        difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0
        rows.append(
            TensorComparison(
                name=name,
                shape_match=True,
                max_abs=max_abs,
                rmse=rmse,
                max_abs_tolerance=tolerance[0],
                rmse_tolerance=tolerance[1],
                passed=bool(max_abs <= tolerance[0] and rmse <= tolerance[1]),
            )
        )
    return ComparisonReport(tuple(rows))


def compare_prediction_arrays(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    max_abs_tolerance: float = 1.0e-6,
    rmse_tolerance: float = 1.0e-6,
) -> ComparisonReport:
    """Compare prediction/scale outputs using the existing V6 1e-6 bound."""

    return compare_named_arrays(
        old,
        new,
        max_abs_tolerance=max_abs_tolerance,
        rmse_tolerance=rmse_tolerance,
    )


def snapshot_group(group: Mapping[str, Any], output: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Flatten model-visible group tensors and optional model outputs."""

    snapshot: dict[str, np.ndarray] = {}
    _collect_arrays(group, "group", snapshot)
    if output is not None:
        _collect_arrays(output, "output", snapshot)
    return snapshot


def _collect_arrays(value: Any, prefix: str, result: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_arrays(item, f"{prefix}.{key}", result)
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for key in value._fields:
            _collect_arrays(getattr(value, key), f"{prefix}.{key}", result)
        return
    if isinstance(value, (np.ndarray,)) or hasattr(value, "shape"):
        try:
            result[prefix] = np.asarray(value)
        except (TypeError, ValueError):
            return
