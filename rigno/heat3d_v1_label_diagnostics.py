"""Smoke-level label diagnostics for Heat3D v1 supervised samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = (
    "coords.npy",
    "k_field.npy",
    "q_field.npy",
    "temperature.npy",
    "sample_meta.json",
)

SUPPORTED_K_WIDTHS = {1, 3}
UNSUPPORTED_K_WIDTHS = {6}
BOTTOM_PASS_TOL_K = 1e-6
BOTTOM_WARNING_TOL_K = 1e-3
LABEL_META_REQUIRED_FIELDS = (
    "solver_name",
    "solver_version",
    "convergence_flag",
    "residual_norm",
    "bottom_dirichlet_error",
    "warnings",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def find_sample_dirs(path: str | Path) -> list[Path]:
    """Return sample directories from a sample, samples, or subset path."""

    root = Path(path)
    if root.name.startswith("sample_") and root.is_dir():
        return [root]
    if (root / "samples").is_dir():
        root = root / "samples"
    if not root.is_dir():
        return []
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith("sample_")
    )


def _status_rank(status: str) -> int:
    return {"pass": 0, "warning": 1, "fail": 2}.get(status, 2)


def _combine_status(*statuses: str) -> str:
    ranked = max((_status_rank(status), status) for status in statuses)
    return ranked[1]


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))),
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
    }


def _load_required(sample_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] | None = None

    for name in REQUIRED_FILES:
        path = sample_dir / name
        if not path.exists():
            errors.append(f"missing required file: {name}")
            continue
        if name.endswith(".npy"):
            try:
                arrays[name] = np.load(path)
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                errors.append(f"failed to load {name}: {exc}")
        else:
            try:
                meta = load_json(path)
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                errors.append(f"failed to load {name}: {exc}")

    return arrays, meta, errors


def _label_meta_checks(sample_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Read optional solver label metadata and validate core smoke fields."""

    path = sample_dir / "label_meta.json"
    if not path.exists():
        return {"present": False, "status": "not_present"}, [], []

    errors: list[str] = []
    warnings: list[str] = []
    try:
        label_meta = load_json(path)
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {
            "present": True,
            "status": "fail",
            "path": str(path),
        }, [f"failed to load label_meta.json: {exc}"], []

    missing = [field for field in LABEL_META_REQUIRED_FIELDS if field not in label_meta]
    if missing:
        errors.append(f"label_meta.json missing required fields: {missing}")

    residual_norm = label_meta.get("residual_norm")
    if residual_norm is not None:
        try:
            residual_norm = float(residual_norm)
            if not np.isfinite(residual_norm):
                errors.append("label_meta.residual_norm must be finite")
        except (TypeError, ValueError):
            errors.append("label_meta.residual_norm must be numeric")

    bottom_error = label_meta.get("bottom_dirichlet_error")
    if bottom_error is not None:
        try:
            bottom_error = float(bottom_error)
            if not np.isfinite(bottom_error):
                errors.append("label_meta.bottom_dirichlet_error must be finite")
        except (TypeError, ValueError):
            errors.append("label_meta.bottom_dirichlet_error must be numeric")

    convergence_flag = label_meta.get("convergence_flag")
    if convergence_flag is not None and not isinstance(convergence_flag, bool):
        errors.append("label_meta.convergence_flag must be boolean")
    elif convergence_flag is False:
        errors.append("label_meta.convergence_flag is false")

    solver_warnings = label_meta.get("warnings", [])
    if solver_warnings is not None and not isinstance(solver_warnings, list):
        errors.append("label_meta.warnings must be a list")
        solver_warnings = []

    status = "fail" if errors else "pass"
    summary = {
        "present": True,
        "status": status,
        "path": str(path),
        "solver_name": label_meta.get("solver_name"),
        "solver_version": label_meta.get("solver_version"),
        "convergence_flag": convergence_flag,
        "residual_norm": residual_norm,
        "bottom_dirichlet_error": bottom_error,
        "warning_count": len(solver_warnings),
        "warnings": solver_warnings,
    }
    return summary, errors, warnings


def _boundary_params(meta: dict[str, Any], boundary: str) -> dict[str, Any]:
    params = meta.get("boundary_params", {})
    if isinstance(params, dict):
        value = params.get(boundary, {})
        if isinstance(value, dict):
            return value
    return {}


def resolve_t_ref(meta: dict[str, Any]) -> dict[str, Any]:
    """Resolve non-leaking T_ref from BC metadata."""

    bottom_params = _boundary_params(meta, "bottom")
    if "fixed_temperature_K" in bottom_params:
        return {
            "value": float(bottom_params["fixed_temperature_K"]),
            "source": "bottom_dirichlet_fixed_temperature",
            "fallback": False,
        }

    top_params = _boundary_params(meta, "top")
    if "ambient_temperature_K" in top_params:
        return {
            "value": float(top_params["ambient_temperature_K"]),
            "source": "top_robin_ambient_temperature",
            "fallback": False,
        }

    return {"value": 300.0, "source": "fallback_300K", "fallback": True}


def _shape_checks(arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    summary = {name: _array_summary(array) for name, array in arrays.items()}

    coords = arrays.get("coords.npy")
    k_field = arrays.get("k_field.npy")
    q_field = arrays.get("q_field.npy")
    temperature = arrays.get("temperature.npy")

    if coords is None or k_field is None or q_field is None or temperature is None:
        return summary, errors, warnings

    if coords.ndim != 2 or coords.shape[1] != 3:
        errors.append(f"coords.npy must have shape (N, 3), found {coords.shape}")
        return summary, errors, warnings

    n_points = coords.shape[0]
    if k_field.ndim != 2 or k_field.shape[0] != n_points:
        errors.append(f"k_field.npy must have shape (N, C), found {k_field.shape}")
    elif k_field.shape[1] in UNSUPPORTED_K_WIDTHS:
        errors.append("k_field.npy has shape (N, 6), unsupported_in_current_smoke")
    elif k_field.shape[1] not in SUPPORTED_K_WIDTHS:
        errors.append(
            f"k_field.npy channel count must be 1 or 3 for current smoke, found {k_field.shape[1]}"
        )

    if q_field.ndim != 2 or q_field.shape != (n_points, 1):
        errors.append(f"q_field.npy must have shape (N, 1), found {q_field.shape}")

    if temperature.ndim != 2 or temperature.shape != (n_points, 1):
        errors.append(
            f"temperature.npy must have shape (N, 1), found {temperature.shape}"
        )

    for name, item in summary.items():
        if not item["finite"]:
            errors.append(f"{name} contains NaN or Inf")

    return summary, errors, warnings


def _temperature_stats(
    coords: np.ndarray,
    temperature: np.ndarray,
    t_ref: float,
) -> dict[str, Any]:
    delta_t = temperature[:, 0] - t_ref
    peak_index = int(np.argmax(temperature[:, 0]))
    return {
        "T_min": float(np.min(temperature)),
        "T_max": float(np.max(temperature)),
        "T_mean": float(np.mean(temperature)),
        "DeltaT_min": float(np.min(delta_t)),
        "DeltaT_max": float(np.max(delta_t)),
        "DeltaT_mean": float(np.mean(delta_t)),
        "peak_temperature": float(temperature[peak_index, 0]),
        "peak_index": peak_index,
        "peak_coord": [float(value) for value in coords[peak_index]],
    }


def _bottom_dirichlet_check(
    coords: np.ndarray,
    temperature: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    bottom_params = _boundary_params(meta, "bottom")
    if "fixed_temperature_K" not in bottom_params:
        return {
            "status": "warning",
            "reason": "bottom fixed_temperature_K missing",
            "max_abs_error_K": None,
        }

    fixed_t = float(bottom_params["fixed_temperature_K"])
    z_min = float(np.min(coords[:, 2]))
    bottom_mask = np.isclose(coords[:, 2], z_min)
    if not np.any(bottom_mask):
        return {
            "status": "fail",
            "reason": "no bottom points found from z_min",
            "z_min": z_min,
            "max_abs_error_K": None,
        }

    max_abs_error = float(np.max(np.abs(temperature[bottom_mask, 0] - fixed_t)))
    if max_abs_error <= BOTTOM_PASS_TOL_K:
        status = "pass"
    elif max_abs_error <= BOTTOM_WARNING_TOL_K:
        status = "warning"
    else:
        status = "fail"

    return {
        "status": status,
        "fixed_temperature_K": fixed_t,
        "z_min": z_min,
        "bottom_point_count": int(np.sum(bottom_mask)),
        "max_abs_error_K": max_abs_error,
        "pass_tolerance_K": BOTTOM_PASS_TOL_K,
        "warning_tolerance_K": BOTTOM_WARNING_TOL_K,
    }


def _not_computed_diagnostics() -> dict[str, dict[str, str]]:
    return {
        "top_robin_residual": {
            "status": "requires_numerical_operator",
            "reason": "requires credible boundary flux / gradient operator",
        },
        "side_adiabatic_flux": {
            "status": "requires_numerical_operator",
            "reason": "requires side-face flux operator",
        },
        "interface_flux_mismatch": {
            "status": "requires_numerical_operator",
            "reason": "requires interface pairing and conservative flux operator",
        },
        "global_energy_balance": {
            "status": "requires_numerical_operator",
            "reason": "requires discrete generation/removal flux accounting",
        },
        "pde_residual": {
            "status": "not_computed",
            "reason": "not implemented in current label diagnostics smoke",
        },
    }


def diagnose_sample(sample_dir: str | Path) -> dict[str, Any]:
    """Run smoke-level label diagnostics for one sample directory."""

    sample_path = Path(sample_dir)
    arrays, meta, load_errors = _load_required(sample_path)
    sample_id = sample_path.name
    if meta is not None:
        sample_id = str(meta.get("sample_id", sample_id))

    report: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_dir": str(sample_path),
        "split": meta.get("split") if meta else None,
        "stage": meta.get("stage") if meta else None,
        "errors": list(load_errors),
        "warnings": [],
    }

    if load_errors or meta is None:
        report["overall_status"] = "fail"
        report["not_computed"] = _not_computed_diagnostics()
        return report

    array_summary, shape_errors, shape_warnings = _shape_checks(arrays)
    report["arrays"] = array_summary
    report["errors"].extend(shape_errors)
    report["warnings"].extend(shape_warnings)

    required = ("coords.npy", "k_field.npy", "q_field.npy", "temperature.npy")
    if any(name not in arrays for name in required) or shape_errors:
        report["overall_status"] = "fail"
        report["not_computed"] = _not_computed_diagnostics()
        return report

    coords = arrays["coords.npy"]
    temperature = arrays["temperature.npy"]
    t_ref = resolve_t_ref(meta)
    report["t_ref"] = t_ref
    report["temperature"] = _temperature_stats(coords, temperature, float(t_ref["value"]))
    report["bottom_dirichlet"] = _bottom_dirichlet_check(coords, temperature, meta)
    label_meta, label_meta_errors, label_meta_warnings = _label_meta_checks(sample_path)
    report["label_meta"] = label_meta
    report["errors"].extend(label_meta_errors)
    report["warnings"].extend(label_meta_warnings)
    report["not_computed"] = _not_computed_diagnostics()

    computed_status = _combine_status(
        report["bottom_dirichlet"]["status"],
        label_meta.get("status", "pass") if label_meta.get("present") else "pass",
    )
    if report["errors"]:
        report["overall_status"] = "fail"
    elif report["warnings"] or computed_status == "warning":
        report["overall_status"] = "warning"
    else:
        report["overall_status"] = "pass"

    return report
