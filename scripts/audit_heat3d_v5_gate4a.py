#!/usr/bin/env python3
"""V5 Gate 4A offline learned scale-correction feasibility audit.

The script fits only small offline scalar models.  Frozen V4 field shapes and
raw-temperature prediction archives are never changed.  No RIGNO parameter,
formal configuration, data, label, or split assignment is modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "rigno").is_dir() and (Path.cwd() / "rigno").is_dir():
    REPO_ROOT = Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
TEMP_SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR, TEMP_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_heat3d_v5_gate3 as gate3  # noqa: E402


AUDIT_ID = "V5-Gate-4A-offline-learned-scale-correction-feasibility"
SCHEMA_VERSION = "heat3d_v5_gate4a_offline_scale_correction_v1"
GATE1_FINAL_TABLE_SHA256 = "79b7f79c32ac5c3da100e27ebafeeea25cb185088687785c6140f0359bde7de9"
GATE3_TABLE_SHA256 = "ff5ab6eaf49401079d0ddd034601ad470f22b158e85ada651d24d013344d5bd8"
ROLE_ORDER = (
    "train",
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
CLEAN_ROLES = ("train", "valid_iid", "test_iid")
HARD_ROLES = ("hard_train_holdout", "hard_challenge_valid", "hard_challenge_test")
CHECKPOINTS = ("best", "final")
PROTOCOLS = ("clean_only_zero_shot", "hard_adapted")
RIDGE_LAMBDAS = (0.0, 1.0e-4, 1.0e-2, 1.0, 10.0)
MLP_HIDDEN = 16
MLP_EPOCHS = 800
MLP_LR = 1.0e-2
MLP_WEIGHT_DECAY = 1.0e-4
MLP_SEED = 20260711
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260712
CLEAN_GUARD_RELATIVE_DEGRADATION = 0.05
EPS_K = 1.0e-12
FEATURE_STD_EPS = 1.0e-12
RECON_TOL = 1.0e-8

GLOBAL_FEATURES = (
    "P_operator_W",
    "raw_z_collapsed_1d_operator_K",
    "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W",
    "q_low_k_overlap_fraction",
    "source_concentration",
    "source_z_centroid_normalized",
    "source_layer_kz_heterogeneity_cv",
    "harmonic_kx_W_mK",
    "harmonic_ky_W_mK",
    "harmonic_kz_W_mK",
    "anisotropy_xy_over_z",
    "Lx_m",
    "Ly_m",
    "Lz_m",
    "top_area_m2",
    "top_h_W_m2K",
    "T_bottom_K",
    "T_inf_K",
    "T_inf_minus_T_bottom_K",
)
GATE1_FEATURES = frozenset(
    {
        "P_operator_W",
        "raw_z_collapsed_1d_operator_K",
        "harmonic_kx_W_mK",
        "harmonic_ky_W_mK",
        "harmonic_kz_W_mK",
        "anisotropy_xy_over_z",
        "Lx_m",
        "Ly_m",
        "Lz_m",
        "top_area_m2",
        "top_h_W_m2K",
        "T_bottom_K",
        "T_inf_K",
        "T_inf_minus_T_bottom_K",
    }
)
GATE3_FEATURES = frozenset(set(GLOBAL_FEATURES) - GATE1_FEATURES)
FORBIDDEN_FEATURE_TOKENS = ("target", "residual", "oracle", "prediction", "pred_")


class AuditError(RuntimeError):
    """Raised when the frozen Gate 4A contract is violated."""


@dataclass(frozen=True)
class FieldContext:
    sample_id: str
    target_raw_K: np.ndarray
    volumes: np.ndarray
    reference_K: float
    boundary: gate3.BoundaryContract
    frozen_shape: Mapping[str, np.ndarray]
    frozen_scale_K: Mapping[str, float]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_key(role: str) -> tuple[int, str]:
    try:
        return (ROLE_ORDER.index(role), role)
    except ValueError:
        return (len(ROLE_ORDER), role)


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise AuditError(f"{name} must be finite")
    return number


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def _read_csv_by_id(path: Path, label: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not path.is_file():
        raise AuditError(f"{label} does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise AuditError(f"{label} has no sample_id column")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in rows:
                raise AuditError(f"{label} has invalid or duplicate sample_id {sample_id!r}")
            rows[sample_id] = dict(row)
    return rows, list(reader.fieldnames)


def _load_split_map(path: Path, contract: Mapping[str, Any]) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    payload = _read_json(path)
    raw = _mapping(payload.get("sample_splits"), "sample_splits")
    assignments = {str(sample_id): str(role) for sample_id, role in raw.items()}
    if not assignments:
        raise AuditError("sample_splits is empty")
    counts = Counter(assignments.values())
    dataset_contract = _mapping(contract.get("dataset_contract"), "dataset_contract")
    if payload.get("dataset_id") != dataset_contract.get("dataset_id"):
        raise AuditError("split map dataset_id disagrees with contract")
    expected_counts = {str(key): int(value) for key, value in _mapping(dataset_contract.get("role_counts"), "role_counts").items()}
    if dict(counts) != expected_counts:
        raise AuditError("split role counts disagree with contract")
    if len(assignments) != int(dataset_contract.get("total_sample_count", -1)):
        raise AuditError("split sample count disagrees with contract")
    return assignments, sorted(counts, key=_role_key), payload


def _assert_contract(contract: Mapping[str, Any]) -> None:
    target = _mapping(contract.get("target_and_reconstruction"), "target_and_reconstruction")
    if target.get("physics_only") != "delta_s_hat = 0":
        raise AuditError("physics-only contract drift")
    if target.get("shape_training") is not False:
        raise AuditError("Gate 4A must keep shape training disabled")
    source = _mapping(contract.get("input_feature_contract"), "input_feature_contract")
    features = tuple(str(value) for value in source.get("global_physics_features", []))
    if features != GLOBAL_FEATURES:
        raise AuditError("global feature allowlist drift")
    forbidden = tuple(str(value) for value in source.get("forbidden_input_categories", []))
    if not forbidden:
        raise AuditError("contract must explicitly name forbidden label-derived inputs")
    protocols = _mapping(contract.get("protocols"), "protocols")
    clean = _mapping(protocols.get("clean_only_zero_shot"), "clean_only_zero_shot")
    hard = _mapping(protocols.get("hard_adapted"), "hard_adapted")
    if tuple(clean.get("fit_roles", [])) != ("train",) or clean.get("selection_role") != "valid_iid":
        raise AuditError("clean-only protocol drift")
    if tuple(hard.get("fit_roles", [])) != ("train", "hard_train_holdout") or hard.get("selection_role") != "hard_challenge_valid":
        raise AuditError("hard-adapted protocol drift")
    for protocol in (clean, hard):
        if tuple(protocol.get("test_roles", [])) != ("test_iid", "hard_challenge_test"):
            raise AuditError("test-role policy drift")


def _validate_feature_allowlist() -> None:
    for name in GLOBAL_FEATURES:
        normalized = name.lower()
        if any(token in normalized for token in FORBIDDEN_FEATURE_TOKENS):
            raise AuditError(f"global feature appears label-derived: {name}")
    if "raw_z_collapsed_1d_operator_K" not in GLOBAL_FEATURES:
        raise AuditError("s_phys must be a global feature")


def _load_latents(path: Path, expected_ids: set[str], checkpoint: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"{checkpoint} latent archive does not exist: {path}")
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AuditError(f"cannot read {checkpoint} latent archive: {exc}") from exc
    latents = {sample_id: np.asarray(archive[sample_id], dtype=np.float64).reshape(-1) for sample_id in archive.files}
    if set(latents) != expected_ids:
        raise AuditError(f"{checkpoint} latent IDs do not match split map")
    dimensions = {value.shape for value in latents.values()}
    if len(dimensions) != 1 or next(iter(dimensions))[0] < 1:
        raise AuditError(f"{checkpoint} pooled latent dimension is invalid")
    if not all(np.all(np.isfinite(value)) for value in latents.values()):
        raise AuditError(f"{checkpoint} pooled latent contains non-finite values")
    return latents, {"path": path.as_posix(), "sha256": _sha256(path), "dimension": int(next(iter(dimensions))[0]), "sample_count": len(latents)}


def _attach_latent_manifest(provenance: Mapping[str, Any], manifest_path: Path, checkpoint: str) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    if payload.get("latent_archive_sha256") != provenance["sha256"]:
        raise AuditError(f"{checkpoint} latent manifest archive hash mismatch")
    if int(payload.get("sample_count", -1)) != int(provenance["sample_count"]):
        raise AuditError(f"{checkpoint} latent manifest sample count mismatch")
    if int(payload.get("latent_dimension", -1)) != int(provenance["dimension"]):
        raise AuditError(f"{checkpoint} latent manifest dimension mismatch")
    if float(payload.get("max_prediction_abs_error_K", math.inf)) > 2.0e-2:
        raise AuditError(f"{checkpoint} latent export exceeds frozen prediction tolerance")
    if not payload.get("checkpoint_sha256") or not payload.get("run_config_sha256"):
        raise AuditError(f"{checkpoint} latent manifest lacks frozen checkpoint provenance")
    return {**dict(provenance), "manifest_path": manifest_path.as_posix(), "manifest_sha256": _sha256(manifest_path), "manifest": payload}


def _protocol_spec(name: str) -> dict[str, Any]:
    if name == "clean_only_zero_shot":
        return {"fit_roles": ("train",), "selection_role": "valid_iid", "clean_guard_role": None}
    if name == "hard_adapted":
        return {
            "fit_roles": ("train", "hard_train_holdout"),
            "selection_role": "hard_challenge_valid",
            "clean_guard_role": "valid_iid",
        }
    raise AuditError(f"unknown protocol {name}")


def _ridge_id(feature_set: str, value: float) -> str:
    return f"ridge_{feature_set}_l{value:.4g}"


def _candidate_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {
        "physics_only": {"kind": "fixed", "feature_set": "none", "family": "physics"},
        "v4_uncorrected_scale": {"kind": "fixed", "feature_set": "none", "family": "v4"},
    }
    for feature_set, family in (("global", "global"), ("global_latent", "global_latent")):
        for value in RIDGE_LAMBDAS:
            catalog[_ridge_id(feature_set, value)] = {
                "kind": "ridge",
                "feature_set": feature_set,
                "family": family,
                "ridge_lambda": value,
            }
        catalog[f"mlp_{feature_set}"] = {
            "kind": "mlp",
            "feature_set": feature_set,
            "family": family,
            "hidden_size": MLP_HIDDEN,
            "epochs": MLP_EPOCHS,
            "learning_rate": MLP_LR,
            "weight_decay": MLP_WEIGHT_DECAY,
            "seed": MLP_SEED,
        }
    return catalog


CANDIDATE_CATALOG = _candidate_catalog()
TRAINED_CANDIDATES = tuple(name for name, spec in CANDIDATE_CATALOG.items() if spec["kind"] != "fixed")
GLOBAL_CANDIDATES = tuple(name for name, spec in CANDIDATE_CATALOG.items() if spec["family"] == "global")
LATENT_CANDIDATES = tuple(name for name, spec in CANDIDATE_CATALOG.items() if spec["family"] == "global_latent")
ALL_CANDIDATES = tuple(CANDIDATE_CATALOG)


def _feature_columns(checkpoint: str, latent_dim: int, feature_set: str) -> list[str]:
    columns = [f"feature_{name}" for name in GLOBAL_FEATURES]
    if feature_set == "global_latent":
        columns.extend(f"{checkpoint}_pooled_latent_{index:03d}" for index in range(latent_dim))
    return columns


def _fit_standardizer(X: np.ndarray, fit_mask: np.ndarray, feature_names: Sequence[str]) -> dict[str, Any]:
    if X.ndim != 2 or fit_mask.shape != (X.shape[0],):
        raise AuditError("feature standardizer dimensions are invalid")
    if int(fit_mask.sum()) < 2:
        raise AuditError("at least two fit samples are required")
    mean = X[fit_mask].mean(axis=0)
    std = X[fit_mask].std(axis=0)
    active = np.flatnonzero(std > FEATURE_STD_EPS)
    if active.size == 0:
        raise AuditError("all model inputs are constant on the fit roles")
    return {
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "active_indices": active.astype(int).tolist(),
        "dropped_feature_names": [str(feature_names[index]) for index in range(X.shape[1]) if index not in set(active.tolist())],
    }


def _standardize(X: np.ndarray, standardizer: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    active = np.asarray(standardizer["active_indices"], dtype=np.int64)
    if X.shape[1] != mean.size or mean.shape != std.shape or active.size == 0:
        raise AuditError("stored standardizer dimensions are invalid")
    return (X[:, active] - mean[active]) / std[active]


def _fit_ridge(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, feature_names: Sequence[str], ridge_lambda: float) -> dict[str, Any]:
    standardizer = _fit_standardizer(X, fit_mask, feature_names)
    Z = _standardize(X, standardizer)
    design = np.column_stack((np.ones(int(fit_mask.sum()), dtype=np.float64), Z[fit_mask]))
    if float(ridge_lambda) == 0.0:
        # The explicit linear-regression baseline avoids normal equations: the
        # pooled latent can be collinear, and SVD least squares is stable across
        # the local and remote NumPy/BLAS implementations used for verification.
        coefficients = np.linalg.lstsq(design, y[fit_mask], rcond=1.0e-10)[0]
    else:
        regularizer = np.diag(np.concatenate(([0.0], np.full(Z.shape[1], float(ridge_lambda)))))
        try:
            coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y[fit_mask])
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(design.T @ design + regularizer, design.T @ y[fit_mask], rcond=1.0e-10)[0]
    if not np.all(np.isfinite(coefficients)):
        raise AuditError("linear/ridge fit produced non-finite coefficients")
    return {
        "kind": "ridge",
        "ridge_lambda": float(ridge_lambda),
        "standardizer": standardizer,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
    }


def _fit_mlp(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, feature_names: Sequence[str], seed: int) -> dict[str, Any]:
    standardizer = _fit_standardizer(X, fit_mask, feature_names)
    Z = _standardize(X, standardizer)
    target_mean = float(y[fit_mask].mean())
    target_std = float(y[fit_mask].std())
    if target_std <= FEATURE_STD_EPS:
        target_std = 1.0
    target = (y[fit_mask] - target_mean) / target_std
    train_X = Z[fit_mask]
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, math.sqrt(2.0 / train_X.shape[1]), size=(train_X.shape[1], MLP_HIDDEN))
    b1 = np.zeros(MLP_HIDDEN, dtype=np.float64)
    W2 = rng.normal(0.0, math.sqrt(2.0 / MLP_HIDDEN), size=(MLP_HIDDEN, 1))
    b2 = np.zeros(1, dtype=np.float64)
    parameters = [W1, b1, W2, b2]
    moments = [np.zeros_like(value) for value in parameters]
    velocities = [np.zeros_like(value) for value in parameters]
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, MLP_EPOCHS + 1):
        hidden = np.tanh(train_X @ W1 + b1)
        prediction = (hidden @ W2 + b2).reshape(-1)
        residual = prediction - target
        grad_prediction = 2.0 * residual[:, None] / train_X.shape[0]
        grad_W2 = hidden.T @ grad_prediction + 2.0 * MLP_WEIGHT_DECAY * W2
        grad_b2 = grad_prediction.sum(axis=0)
        grad_hidden = grad_prediction @ W2.T
        grad_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_W1 = train_X.T @ grad_pre + 2.0 * MLP_WEIGHT_DECAY * W1
        grad_b1 = grad_pre.sum(axis=0)
        gradients = [grad_W1, grad_b1, grad_W2, grad_b2]
        for index, gradient in enumerate(gradients):
            moments[index] = beta1 * moments[index] + (1.0 - beta1) * gradient
            velocities[index] = beta2 * velocities[index] + (1.0 - beta2) * (gradient * gradient)
            m_hat = moments[index] / (1.0 - beta1**epoch)
            v_hat = velocities[index] / (1.0 - beta2**epoch)
            parameters[index] -= MLP_LR * m_hat / (np.sqrt(v_hat) + 1.0e-8)
        W1, b1, W2, b2 = parameters
    return {
        "kind": "mlp",
        "standardizer": standardizer,
        "target_mean": target_mean,
        "target_std": target_std,
        "hidden_size": MLP_HIDDEN,
        "epochs": MLP_EPOCHS,
        "learning_rate": MLP_LR,
        "weight_decay": MLP_WEIGHT_DECAY,
        "seed": int(seed),
        "weights": [W1.tolist(), W2.tolist()],
        "biases": [b1.tolist(), b2.tolist()],
    }


def _predict_model(model: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    Z = _standardize(X, _mapping(model.get("standardizer"), "standardizer"))
    kind = str(model.get("kind"))
    if kind == "ridge":
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        if coefficients.shape != (Z.shape[1],):
            raise AuditError("ridge coefficient dimensions are invalid")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            predicted = float(model["intercept"]) + Z @ coefficients
        if not np.all(np.isfinite(predicted)):
            raise AuditError("ridge prediction produced non-finite values")
        return predicted
    if kind == "mlp":
        weights = model.get("weights")
        biases = model.get("biases")
        if not isinstance(weights, list) or not isinstance(biases, list) or len(weights) != 2 or len(biases) != 2:
            raise AuditError("MLP parameters are invalid")
        W1, W2 = (np.asarray(value, dtype=np.float64) for value in weights)
        b1, b2 = (np.asarray(value, dtype=np.float64) for value in biases)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hidden = np.tanh(Z @ W1 + b1)
            normalized = (hidden @ W2 + b2).reshape(-1)
            predicted = normalized * float(model["target_std"]) + float(model["target_mean"])
        if not np.all(np.isfinite(predicted)):
            raise AuditError("MLP prediction produced non-finite values")
        return predicted
    raise AuditError(f"unsupported model kind {kind!r}")


def _fit_model(
    *,
    candidate: str,
    X: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    spec = CANDIDATE_CATALOG[candidate]
    if spec["kind"] == "ridge":
        return _fit_ridge(X, y, fit_mask, feature_names, float(spec["ridge_lambda"]))
    if spec["kind"] == "mlp":
        return _fit_mlp(X, y, fit_mask, feature_names, int(spec["seed"]))
    raise AuditError(f"cannot fit fixed candidate {candidate}")


def _build_input_rows(
    *,
    gate1_table: Path,
    gate3_table: Path,
    assignments: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate1, gate1_columns = _read_csv_by_id(gate1_table, "Gate 1 table")
    gate3_rows, gate3_columns = _read_csv_by_id(gate3_table, "Gate 3 table")
    expected_ids = set(assignments)
    if set(gate1) != expected_ids or set(gate3_rows) != expected_ids:
        raise AuditError("Gate 1/Gate 3 CSV IDs must exactly match the split map")
    missing = [name for name in GLOBAL_FEATURES if name not in gate1_columns and name not in gate3_columns]
    if missing:
        raise AuditError(f"required global feature columns are missing: {missing}")
    required_gate1 = ("raw_z_collapsed_1d_operator_K", "role", "input_fingerprint", "full_fingerprint", "provenance_source_id")
    required_gate3 = ("target_scale_cv_rms_K", "reference_temperature_K", "role")
    if any(name not in gate1_columns for name in required_gate1) or any(name not in gate3_columns for name in required_gate3):
        raise AuditError("predecessor CSV schema lacks required Gate 4A values")
    result: list[dict[str, Any]] = []
    for sample_id in sorted(expected_ids):
        first = gate1[sample_id]
        third = gate3_rows[sample_id]
        role = assignments[sample_id]
        if first["role"] != role or third["role"] != role:
            raise AuditError(f"{sample_id}: predecessor role disagrees with split map")
        features: dict[str, float] = {}
        for feature in GLOBAL_FEATURES:
            source = first if feature in gate1_columns else third
            features[feature] = _finite(source[feature], f"{sample_id}.{feature}")
        s_phys = features["raw_z_collapsed_1d_operator_K"]
        s_true = _finite(third["target_scale_cv_rms_K"], f"{sample_id}.target_scale_cv_rms_K")
        if s_phys <= EPS_K or s_true <= EPS_K:
            raise AuditError(f"{sample_id}: s_phys and s_true must be positive")
        result.append(
            {
                "sample_id": sample_id,
                "role": role,
                "is_clean_role": int(role in CLEAN_ROLES),
                "is_hard_role": int(role in HARD_ROLES),
                "input_fingerprint": first["input_fingerprint"],
                "full_fingerprint": first["full_fingerprint"],
                "provenance_source_id": first["provenance_source_id"],
                "reference_temperature_K": _finite(third["reference_temperature_K"], f"{sample_id}.reference_temperature_K"),
                "s_phys_K": s_phys,
                "s_true_K": s_true,
                "target_delta_s": math.log(s_true / s_phys),
                **{f"feature_{name}": value for name, value in features.items()},
            }
        )
    provenance = {
        "gate1_table": {"path": gate1_table.as_posix(), "sha256": _sha256(gate1_table), "columns": gate1_columns},
        "gate3_table": {"path": gate3_table.as_posix(), "sha256": _sha256(gate3_table), "columns": gate3_columns},
    }
    return result, provenance


def _build_field_contexts(
    *,
    dataset: Path,
    input_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, FieldContext]:
    contexts: dict[str, FieldContext] = {}
    for row in input_rows:
        sample_id = str(row["sample_id"])
        sample_dir = dataset / sample_id
        meta = gate3._read_json(sample_dir / "sample_meta.json")
        arrays = gate3._load_arrays(sample_dir)
        # Gate 3 deliberately memory-maps sample arrays.  Gate 4A retains a
        # field context for every sample, so take owned copies of the three
        # arrays needed below before releasing all five mmap handles.
        coords = np.array(arrays["coords"], dtype=np.float64, copy=True)
        target_raw = np.array(arrays["temperature"], dtype=np.float64, copy=True).reshape(-1)
        bc_features = np.array(arrays["bc_features"], dtype=np.float64, copy=True)
        del arrays
        volumes, _axes, _inverse, _shape = gate3._control_volumes(coords)
        boundary = gate3._resolve_boundary_contract(
            meta=meta,
            bc_features=bc_features,
            coords=coords,
            reference_region_id="bottom",
            allow_coordinate_fallback=False,
        )
        target_delta, target_scale, _target_shape = gate3._shape_scale_decompose(
            target_raw, volumes, boundary.reference_temperature_K
        )
        if not math.isclose(target_scale, float(row["s_true_K"]), rel_tol=1.0e-10, abs_tol=1.0e-10):
            raise AuditError(f"{sample_id}: target scale disagrees with Gate 3 table")
        frozen_shape: dict[str, np.ndarray] = {}
        frozen_scale: dict[str, float] = {}
        for checkpoint in CHECKPOINTS:
            raw = np.asarray(predictions[checkpoint][sample_id], dtype=np.float64).reshape(-1)
            if raw.shape != target_raw.shape:
                raise AuditError(f"{sample_id}: {checkpoint} frozen prediction node count mismatch")
            _delta, scale, shape = gate3._shape_scale_decompose(raw, volumes, boundary.reference_temperature_K)
            frozen_shape[checkpoint] = shape
            frozen_scale[checkpoint] = scale
        contexts[sample_id] = FieldContext(
            sample_id=sample_id,
            target_raw_K=target_raw,
            volumes=volumes,
            reference_K=boundary.reference_temperature_K,
            boundary=boundary,
            frozen_shape=frozen_shape,
            frozen_scale_K=frozen_scale,
        )
        if np.max(np.abs(target_delta - (target_raw - boundary.reference_temperature_K))) > RECON_TOL:
            raise AuditError(f"{sample_id}: target DeltaT reconstruction drift")
    return contexts


def _field_cv_rmse(context: FieldContext, checkpoint: str, corrected_scale_K: float) -> float:
    raw = gate3._shape_scale_reconstruct(
        context.frozen_shape[checkpoint], corrected_scale_K, context.reference_K
    )
    projected = gate3._boundary_project_raw(raw, context.boundary)
    return gate3._weighted_rms(projected - context.target_raw_K, context.volumes)


def _base_columns(latent_dims: Mapping[str, int]) -> list[str]:
    columns = [
        "sample_id",
        "role",
        "is_clean_role",
        "is_hard_role",
        "input_fingerprint",
        "full_fingerprint",
        "provenance_source_id",
        "reference_temperature_K",
        "s_phys_K",
        "s_true_K",
        "target_delta_s",
    ]
    columns.extend(f"feature_{name}" for name in GLOBAL_FEATURES)
    for checkpoint in CHECKPOINTS:
        columns.extend(f"{checkpoint}_pooled_latent_{index:03d}" for index in range(int(latent_dims[checkpoint])))
        columns.extend(
            (
                f"{checkpoint}_v4_uncorrected_delta_s",
                f"{checkpoint}_v4_uncorrected_scale_K",
                f"{checkpoint}_v4_uncorrected_scale_log_error",
                f"{checkpoint}_v4_uncorrected_frozen_shape_field_cv_rmse_K",
            )
        )
    for protocol in PROTOCOLS:
        for checkpoint in CHECKPOINTS:
            for candidate in ALL_CANDIDATES:
                prefix = f"{protocol}_{checkpoint}_{candidate}"
                columns.extend(
                    (
                        f"{prefix}_delta_s_hat",
                        f"{prefix}_scale_hat_K",
                        f"{prefix}_scale_log_error",
                        f"{prefix}_frozen_shape_field_cv_rmse_K",
                    )
                )
    return columns


STRING_COLUMNS = {
    "sample_id",
    "role",
    "input_fingerprint",
    "full_fingerprint",
    "provenance_source_id",
}
INT_COLUMNS = {"is_clean_role", "is_hard_role"}


def _write_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["sample_id"])):
            encoded: dict[str, str] = {}
            for column in columns:
                value = row.get(column)
                if value is None:
                    encoded[column] = ""
                elif column in STRING_COLUMNS:
                    encoded[column] = str(value)
                elif column in INT_COLUMNS:
                    encoded[column] = str(int(value))
                else:
                    encoded[column] = format(float(value), ".17g")
            writer.writerow(encoded)


def _read_table(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"per-sample table does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise AuditError("per-sample table columns do not match the Gate 4A schema")
        for raw in reader:
            row: dict[str, Any] = {}
            for column in columns:
                value = raw[column]
                if value == "":
                    row[column] = None
                elif column in STRING_COLUMNS:
                    row[column] = value
                elif column in INT_COLUMNS:
                    row[column] = int(value)
                else:
                    row[column] = float(value)
            rows.append(row)
    return rows


def _candidate_prefix(protocol: str, checkpoint: str, candidate: str) -> str:
    return f"{protocol}_{checkpoint}_{candidate}"


def _candidate_values(rows: Sequence[Mapping[str, Any]], protocol: str, checkpoint: str, candidate: str, role: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["role"] == role]
    prefix = _candidate_prefix(protocol, checkpoint, candidate)
    scale = np.asarray([float(row[f"{prefix}_scale_log_error"]) for row in selected], dtype=np.float64)
    field = np.asarray([float(row[f"{prefix}_frozen_shape_field_cv_rmse_K"]) for row in selected], dtype=np.float64)
    return scale, field


def _metric_pair(scale_errors: np.ndarray, field_errors: np.ndarray) -> dict[str, float | int | None]:
    if scale_errors.size == 0 or field_errors.size == 0:
        return {"sample_count": 0, "scale_log_RMSE": None, "frozen_shape_field_CV_RMSE_K": None}
    return {
        "sample_count": int(scale_errors.size),
        "scale_log_RMSE": float(math.sqrt(np.mean(scale_errors * scale_errors))),
        "frozen_shape_field_CV_RMSE_K": float(np.mean(field_errors)),
    }


def _metrics_by_role(rows: Sequence[Mapping[str, Any]], protocol: str, checkpoint: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {
        role: [row for row in rows if row["role"] == role] for role in ROLE_ORDER
    }
    groups["clean"] = [row for row in rows if int(row["is_clean_role"]) == 1]
    groups["hard"] = [row for row in rows if int(row["is_hard_role"]) == 1]
    groups["all_samples"] = list(rows)
    for group, members in groups.items():
        candidate_metrics: dict[str, Any] = {}
        for candidate in ALL_CANDIDATES:
            prefix = _candidate_prefix(protocol, checkpoint, candidate)
            scale = np.asarray([float(row[f"{prefix}_scale_log_error"]) for row in members], dtype=np.float64)
            field = np.asarray([float(row[f"{prefix}_frozen_shape_field_cv_rmse_K"]) for row in members], dtype=np.float64)
            candidate_metrics[candidate] = _metric_pair(scale, field)
        result[group] = candidate_metrics
    return result


def _bootstrap_seed(protocol: str, checkpoint: str, comparison: str) -> int:
    protocol_offset = {"clean_only_zero_shot": 0, "hard_adapted": 100}[protocol]
    checkpoint_offset = {"best": 0, "final": 20}[checkpoint]
    comparison_offset = {
        "selected_vs_physics_only": 1,
        "selected_vs_v4_uncorrected_scale": 2,
        "global_only_vs_global_plus_latent": 3,
    }[comparison]
    return BOOTSTRAP_SEED + protocol_offset + checkpoint_offset + comparison_offset


def _paired_bootstrap(
    *,
    candidate_scale: np.ndarray,
    candidate_field: np.ndarray,
    reference_scale: np.ndarray,
    reference_field: np.ndarray,
    seed: int,
    candidate: str,
    reference: str,
    role: str,
) -> dict[str, Any]:
    if not (
        candidate_scale.shape == reference_scale.shape == candidate_field.shape == reference_field.shape
    ) or candidate_scale.ndim != 1:
        raise AuditError("paired bootstrap array shapes disagree")
    count = candidate_scale.size
    if count < 2:
        return {
            "candidate": candidate,
            "reference": reference,
            "role": role,
            "sample_count": int(count),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "scale_log_RMSE_delta": None,
            "frozen_shape_field_CV_RMSE_delta": None,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(BOOTSTRAP_RESAMPLES, count))
    scale_delta = np.sqrt(np.mean(candidate_scale[indices] ** 2, axis=1)) - np.sqrt(
        np.mean(reference_scale[indices] ** 2, axis=1)
    )
    field_delta = np.mean(candidate_field[indices], axis=1) - np.mean(reference_field[indices], axis=1)
    def payload(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point_estimate": float(point),
            "ci95": {
                "low": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "high": float(np.quantile(values, 0.975)),
            },
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "role": role,
        "sample_count": int(count),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "scale_log_RMSE_delta": payload(
            scale_delta,
            math.sqrt(np.mean(candidate_scale**2)) - math.sqrt(np.mean(reference_scale**2)),
        ),
        "frozen_shape_field_CV_RMSE_delta": payload(
            field_delta,
            float(np.mean(candidate_field) - np.mean(reference_field)),
        ),
    }


def _strictly_improves(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return (
        candidate.get("scale_log_RMSE") is not None
        and candidate.get("frozen_shape_field_CV_RMSE_K") is not None
        and float(candidate["scale_log_RMSE"]) < float(reference["scale_log_RMSE"])
        and float(candidate["frozen_shape_field_CV_RMSE_K"]) < float(reference["frozen_shape_field_CV_RMSE_K"])
    )


def _family_winner(
    metrics: Mapping[str, Any], candidates: Sequence[str]) -> str:
    available = [candidate for candidate in candidates if metrics[candidate]["scale_log_RMSE"] is not None]
    if not available:
        raise AuditError("family has no available candidate metrics")
    return min(
        available,
        key=lambda candidate: (
            float(metrics[candidate]["scale_log_RMSE"]),
            float(metrics[candidate]["frozen_shape_field_CV_RMSE_K"]),
            candidate,
        ),
    )


def _select_candidate(rows: Sequence[Mapping[str, Any]], protocol: str, checkpoint: str) -> dict[str, Any]:
    spec = _protocol_spec(protocol)
    all_metrics = _metrics_by_role(rows, protocol, checkpoint)
    selection_role = str(spec["selection_role"])
    selection_metrics = all_metrics[selection_role]
    physics = selection_metrics["physics_only"]
    eligible: list[str] = []
    clean_guard: dict[str, Any] | None = None
    for candidate in TRAINED_CANDIDATES:
        if not _strictly_improves(selection_metrics[candidate], physics):
            continue
        if spec["clean_guard_role"] is not None:
            guard_role = str(spec["clean_guard_role"])
            candidate_guard = all_metrics[guard_role][candidate]
            physics_guard = all_metrics[guard_role]["physics_only"]
            scale_ratio = float(candidate_guard["scale_log_RMSE"]) / max(float(physics_guard["scale_log_RMSE"]), EPS_K)
            field_ratio = float(candidate_guard["frozen_shape_field_CV_RMSE_K"]) / max(
                float(physics_guard["frozen_shape_field_CV_RMSE_K"]), EPS_K
            )
            if scale_ratio > 1.0 + CLEAN_GUARD_RELATIVE_DEGRADATION or field_ratio > 1.0 + CLEAN_GUARD_RELATIVE_DEGRADATION:
                continue
        eligible.append(candidate)
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                float(selection_metrics[candidate]["scale_log_RMSE"]),
                float(selection_metrics[candidate]["frozen_shape_field_CV_RMSE_K"]),
                candidate,
            ),
        )
    if spec["clean_guard_role"] is not None:
        guard_role = str(spec["clean_guard_role"])
        if selected is not None:
            candidate_guard = all_metrics[guard_role][selected]
            physics_guard = all_metrics[guard_role]["physics_only"]
            clean_guard = {
                "role": guard_role,
                "scale_log_RMSE_ratio_to_physics": float(candidate_guard["scale_log_RMSE"]) / max(float(physics_guard["scale_log_RMSE"]), EPS_K),
                "field_CV_RMSE_ratio_to_physics": float(candidate_guard["frozen_shape_field_CV_RMSE_K"]) / max(
                    float(physics_guard["frozen_shape_field_CV_RMSE_K"]), EPS_K
                ),
                "maximum_allowed_ratio": 1.0 + CLEAN_GUARD_RELATIVE_DEGRADATION,
                "passed": True,
            }
        else:
            clean_guard = {"role": guard_role, "passed": False, "reason": "no eligible hard-adapted candidate"}
    global_winner = _family_winner(selection_metrics, GLOBAL_CANDIDATES)
    latent_winner = _family_winner(selection_metrics, LATENT_CANDIDATES)
    def arrays(candidate: str, role: str) -> tuple[np.ndarray, np.ndarray]:
        return _candidate_values(rows, protocol, checkpoint, candidate, role)
    global_scale, global_field = arrays(global_winner, selection_role)
    latent_scale, latent_field = arrays(latent_winner, selection_role)
    latent_bootstrap = _paired_bootstrap(
        candidate_scale=latent_scale,
        candidate_field=latent_field,
        reference_scale=global_scale,
        reference_field=global_field,
        seed=_bootstrap_seed(protocol, checkpoint, "global_only_vs_global_plus_latent"),
        candidate=latent_winner,
        reference=global_winner,
        role=selection_role,
    )
    global_adequate = _strictly_improves(selection_metrics[global_winner], physics)
    scale_ci = latent_bootstrap["scale_log_RMSE_delta"]
    field_ci = latent_bootstrap["frozen_shape_field_CV_RMSE_delta"]
    latent_stable_gain = bool(
        scale_ci is not None
        and field_ci is not None
        and float(scale_ci["ci95"]["high"]) < 0.0
        and float(field_ci["ci95"]["high"]) < 0.0
    )
    bootstraps: dict[str, Any] = {"global_only_vs_global_plus_latent": latent_bootstrap}
    if selected is not None:
        selected_scale, selected_field = arrays(selected, selection_role)
        physics_scale, physics_field = arrays("physics_only", selection_role)
        v4_scale, v4_field = arrays("v4_uncorrected_scale", selection_role)
        bootstraps["selected_vs_physics_only"] = _paired_bootstrap(
            candidate_scale=selected_scale,
            candidate_field=selected_field,
            reference_scale=physics_scale,
            reference_field=physics_field,
            seed=_bootstrap_seed(protocol, checkpoint, "selected_vs_physics_only"),
            candidate=selected,
            reference="physics_only",
            role=selection_role,
        )
        bootstraps["selected_vs_v4_uncorrected_scale"] = _paired_bootstrap(
            candidate_scale=selected_scale,
            candidate_field=selected_field,
            reference_scale=v4_scale,
            reference_field=v4_field,
            seed=_bootstrap_seed(protocol, checkpoint, "selected_vs_v4_uncorrected_scale"),
            candidate=selected,
            reference="v4_uncorrected_scale",
            role=selection_role,
        )
    return {
        "protocol": protocol,
        "checkpoint": checkpoint,
        "fit_roles": list(spec["fit_roles"]),
        "selection_role": selection_role,
        "test_roles_used_for_selection": False,
        "selection_metrics": selection_metrics,
        "metrics_by_group": all_metrics,
        "eligible_trained_candidates": eligible,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
        "clean_guard": clean_guard,
        "family_winners": {
            "global_only": global_winner,
            "global_plus_latent": latent_winner,
        },
        "global_physics_features_adequate": global_adequate,
        "pooled_latent_stable_incremental_gain": latent_stable_gain,
        "paired_bootstrap": bootstraps,
    }


def _duplicate_groups(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field) or "")
        if value:
            groups[value].append(row)
    result: list[dict[str, Any]] = []
    for value, members in groups.items():
        roles = sorted({str(member["role"]) for member in members}, key=_role_key)
        if len(roles) > 1:
            result.append(
                {
                    "key": value,
                    "roles": roles,
                    "samples": [
                        {"sample_id": str(member["sample_id"]), "role": str(member["role"])}
                        for member in sorted(members, key=lambda item: str(item["sample_id"]))
                    ],
                }
            )
    return sorted(result, key=lambda item: (item["roles"], item["key"]))


def _duplicate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_groups = _duplicate_groups(rows, "input_fingerprint")
    full_groups = _duplicate_groups(rows, "full_fingerprint")
    provenance_groups = _duplicate_groups(rows, "provenance_source_id")
    ids = [str(row["sample_id"]) for row in rows]
    return {
        "unique_sample_ids": len(ids) == len(set(ids)),
        "cross_role_model_input_duplicate_groups": {"group_count": len(input_groups), "groups": input_groups},
        "cross_role_full_sample_duplicate_groups": {"group_count": len(full_groups), "groups": full_groups},
        "cross_role_provenance_duplicate_groups": {"group_count": len(provenance_groups), "groups": provenance_groups},
        "pass": len(ids) == len(set(ids)) and not (input_groups or full_groups or provenance_groups),
    }


def _build_reconstructed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles = sorted({str(row["role"]) for row in rows}, key=_role_key)
    results: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        results[protocol] = {checkpoint: _select_candidate(rows, protocol, checkpoint) for checkpoint in CHECKPOINTS}
        best = results[protocol]["best"]
        final = results[protocol]["final"]
        best_direction = {
            "selection_passed": best["selection_passed"],
            "global_physics_features_adequate": best["global_physics_features_adequate"],
            "pooled_latent_stable_incremental_gain": best["pooled_latent_stable_incremental_gain"],
        }
        final_direction = {
            "selection_passed": final["selection_passed"],
            "global_physics_features_adequate": final["global_physics_features_adequate"],
            "pooled_latent_stable_incremental_gain": final["pooled_latent_stable_incremental_gain"],
        }
        results[protocol]["best_final_direction_consistent"] = best_direction == final_direction
        results[protocol]["protocol_passed"] = bool(
            best["selection_passed"]
            and final["selection_passed"]
            and results[protocol]["best_final_direction_consistent"]
        )
    return {
        "row_count": len(rows),
        "role_counts": {role: int(sum(row["role"] == role for row in rows)) for role in roles},
        "protocol_results": results,
        "duplicate_leakage": _duplicate_summary(rows),
        "overall_gate4a_passed": bool(
            all(results[protocol]["protocol_passed"] for protocol in PROTOCOLS)
            and _duplicate_summary(rows)["pass"]
        ),
    }


def _assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AuditError(f"summary reconstruction keys differ at {path}")
        for key in actual:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AuditError(f"summary reconstruction list lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(actual, float) or isinstance(expected, float):
        if actual is None or expected is None or not math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise AuditError(f"summary reconstruction values differ at {path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AuditError(f"summary reconstruction values differ at {path}: {actual!r} != {expected!r}")


def _table_rows_with_latents(
    input_rows: Sequence[Mapping[str, Any]], latents: Mapping[str, Mapping[str, np.ndarray]], latent_dims: Mapping[str, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_row in input_rows:
        row = dict(input_row)
        sample_id = str(row["sample_id"])
        for checkpoint in CHECKPOINTS:
            value = latents[checkpoint][sample_id]
            if value.shape != (int(latent_dims[checkpoint]),):
                raise AuditError(f"{sample_id}: {checkpoint} latent shape drift")
            for index, component in enumerate(value):
                row[f"{checkpoint}_pooled_latent_{index:03d}"] = float(component)
        rows.append(row)
    return rows


def _matrix_from_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)


def _fit_and_apply(
    *,
    rows: list[dict[str, Any]],
    protocol: str,
    checkpoint: str,
    latent_dim: int,
    model_records: dict[str, Any],
) -> None:
    spec = _protocol_spec(protocol)
    fit_mask = np.asarray([row["role"] in spec["fit_roles"] for row in rows], dtype=bool)
    y = np.asarray([float(row["target_delta_s"]) for row in rows], dtype=np.float64)
    s_phys = np.asarray([float(row["s_phys_K"]) for row in rows], dtype=np.float64)
    contexts = {str(row["sample_id"]): row["_field_context"] for row in rows}
    v4_delta: np.ndarray | None = None
    v4_scale: np.ndarray | None = None
    for candidate in ALL_CANDIDATES:
        prefix = _candidate_prefix(protocol, checkpoint, candidate)
        if candidate == "physics_only":
            delta_hat = np.zeros(len(rows), dtype=np.float64)
        elif candidate == "v4_uncorrected_scale":
            if v4_delta is None:
                v4_scale = np.asarray(
                    [contexts[str(row["sample_id"])].frozen_scale_K[checkpoint] for row in rows], dtype=np.float64
                )
                v4_delta = np.log(v4_scale / s_phys)
            delta_hat = v4_delta
        else:
            feature_set = str(CANDIDATE_CATALOG[candidate]["feature_set"])
            columns = _feature_columns(checkpoint, latent_dim, feature_set)
            X = _matrix_from_rows(rows, columns)
            model = _fit_model(
                candidate=candidate,
                X=X,
                y=y,
                fit_mask=fit_mask,
                feature_names=columns,
            )
            model_record_key = f"{protocol}/{checkpoint}/{candidate}"
            model_records[model_record_key] = {
                "protocol": protocol,
                "checkpoint": checkpoint,
                "candidate": candidate,
                "fit_roles": list(spec["fit_roles"]),
                "fit_sample_ids": [str(row["sample_id"]) for row in rows if row["role"] in spec["fit_roles"]],
                "input_feature_columns": columns,
                "input_feature_source": "global physical/BC fields plus frozen input-only pooled latent only",
                **model,
            }
            delta_hat = _predict_model(model, X)
        scale_hat = s_phys * np.exp(delta_hat)
        if not np.all(np.isfinite(scale_hat)) or np.any(scale_hat <= 0.0):
            raise AuditError(f"{protocol}/{checkpoint}/{candidate}: invalid corrected scales")
        for index, row in enumerate(rows):
            context = contexts[str(row["sample_id"])]
            field_error = _field_cv_rmse(context, checkpoint, float(scale_hat[index]))
            row[f"{prefix}_delta_s_hat"] = float(delta_hat[index])
            row[f"{prefix}_scale_hat_K"] = float(scale_hat[index])
            row[f"{prefix}_scale_log_error"] = float(delta_hat[index] - y[index])
            row[f"{prefix}_frozen_shape_field_cv_rmse_K"] = field_error
            if candidate == "v4_uncorrected_scale":
                row[f"{checkpoint}_v4_uncorrected_delta_s"] = float(delta_hat[index])
                row[f"{checkpoint}_v4_uncorrected_scale_K"] = float(scale_hat[index])
                row[f"{checkpoint}_v4_uncorrected_scale_log_error"] = float(delta_hat[index] - y[index])
                row[f"{checkpoint}_v4_uncorrected_frozen_shape_field_cv_rmse_K"] = field_error


def _strip_internal_fields(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        clean = {column: row.get(column) for column in columns}
        missing = [column for column, value in clean.items() if value is None]
        if missing:
            raise AuditError(f"{row['sample_id']}: missing Gate 4A output columns: {missing[:8]}")
        result.append(clean)
    return result


def _validate_field_baselines(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        for protocol in PROTOCOLS:
            for checkpoint in CHECKPOINTS:
                physics_prefix = _candidate_prefix(protocol, checkpoint, "physics_only")
                v4_prefix = _candidate_prefix(protocol, checkpoint, "v4_uncorrected_scale")
                if abs(float(row[f"{physics_prefix}_delta_s_hat"])) > 1.0e-14:
                    raise AuditError("physics-only must keep delta_s_hat=0")
                if not math.isclose(
                    float(row[f"{v4_prefix}_scale_hat_K"]),
                    float(row[f"{checkpoint}_v4_uncorrected_scale_K"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise AuditError("V4 uncorrected baseline must be protocol invariant")


def _model_params_payload(
    *,
    contract: Mapping[str, Any],
    table_columns: Sequence[str],
    latent_provenance: Mapping[str, Any],
    model_records: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "heat3d_v5_gate4a_model_parameters_v1",
        "audit_id": AUDIT_ID,
        "contract_id": contract.get("contract_id"),
        "model_parameter_scope": "offline scalar delta_s models only; no RIGNO or shape parameter updates",
        "candidate_catalog": CANDIDATE_CATALOG,
        "global_feature_allowlist": list(GLOBAL_FEATURES),
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "latent_provenance": latent_provenance,
        "per_sample_table_columns": list(table_columns),
        "model_records": model_records,
        "selection_reconstructed_from_table": reconstructed["protocol_results"],
    }


def _output_paths(paths: Sequence[Path | None], overwrite: bool) -> tuple[Path, Path, Path, Path]:
    if len(paths) != 4 or any(path is None for path in paths):
        raise AuditError("audit requires --output-table, --output-model-params, --output-json, and --output-md")
    table, model_params, summary_json, summary_md = (Path(path) for path in paths if path is not None)
    if len({table.resolve(), model_params.resolve(), summary_json.resolve(), summary_md.resolve()}) != 4:
        raise AuditError("output paths must be distinct")
    if not overwrite:
        existing = [path for path in (table, model_params, summary_json, summary_md) if path.exists()]
        if existing:
            raise AuditError(f"refusing to overwrite existing output(s): {existing}")
    return table, model_params, summary_json, summary_md


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _bootstrap_text(payload: Mapping[str, Any] | None, metric: str) -> str:
    if payload is None:
        return "n/a"
    value = payload.get(metric)
    if not isinstance(value, Mapping):
        return "n/a"
    ci = _mapping(value.get("ci95"), "ci95")
    return f"[{_fmt(ci.get('low'))}, {_fmt(ci.get('high'))}]"


def render_markdown(payload: Mapping[str, Any]) -> str:
    reconstructed = _mapping(payload["reconstructed_from_table"], "reconstructed")
    protocols = _mapping(reconstructed["protocol_results"], "protocol_results")
    table = _mapping(payload["per_sample_table"], "table")
    lines = [
        "# V5 Gate 4A Offline Learned Scale-Correction Closeout",
        "",
        "## Scope",
        "",
        "- `s_phys` is the frozen uncalibrated Gate 1 `z_collapsed_1d_operator`; the learned target is `delta_s = log(s_true / s_phys)`.",
        "- V4 best/final shapes remain frozen. Corrected fields are reconstructed in raw temperature space and then projected only at prescribed Dirichlet nodes.",
        "- This is offline scalar-model feasibility only: no RIGNO, formal loss/configuration, data, label, split, or shape-model update occurred.",
        "",
    ]
    for protocol in PROTOCOLS:
        result = _mapping(protocols[protocol], protocol)
        lines.extend([f"## {protocol}", ""])
        lines.append(
            "| checkpoint | selected candidate | selection role | physics scale log-RMSE | selected scale log-RMSE | physics field RMSE K | selected field RMSE K | global adequate | latent stable gain |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for checkpoint in CHECKPOINTS:
            entry = _mapping(result[checkpoint], checkpoint)
            metrics = _mapping(entry["selection_metrics"], "selection_metrics")
            selected = entry["selected_candidate"]
            selected_metrics = metrics[selected] if selected is not None else {"scale_log_RMSE": None, "frozen_shape_field_CV_RMSE_K": None}
            physics = _mapping(metrics["physics_only"], "physics")
            lines.append(
                "| "
                + " | ".join(
                    (
                        checkpoint,
                        str(selected or "no eligible candidate"),
                        str(entry["selection_role"]),
                        _fmt(physics["scale_log_RMSE"]),
                        _fmt(selected_metrics["scale_log_RMSE"]),
                        _fmt(physics["frozen_shape_field_CV_RMSE_K"]),
                        _fmt(selected_metrics["frozen_shape_field_CV_RMSE_K"]),
                        str(entry["global_physics_features_adequate"]),
                        str(entry["pooled_latent_stable_incremental_gain"]),
                    )
                )
                + " |"
            )
            bootstrap = _mapping(entry["paired_bootstrap"], "bootstrap")
            lines.append(
                f"  - selected vs physics scale/field CI95: `{_bootstrap_text(bootstrap.get('selected_vs_physics_only'), 'scale_log_RMSE_delta')}` / `{_bootstrap_text(bootstrap.get('selected_vs_physics_only'), 'frozen_shape_field_CV_RMSE_delta')}`."
            )
            lines.append(
                f"  - selected vs uncorrected V4 scale/field CI95: `{_bootstrap_text(bootstrap.get('selected_vs_v4_uncorrected_scale'), 'scale_log_RMSE_delta')}` / `{_bootstrap_text(bootstrap.get('selected_vs_v4_uncorrected_scale'), 'frozen_shape_field_CV_RMSE_delta')}`."
            )
            lines.append(
                f"  - global+latent vs global-only scale/field CI95: `{_bootstrap_text(bootstrap.get('global_only_vs_global_plus_latent'), 'scale_log_RMSE_delta')}` / `{_bootstrap_text(bootstrap.get('global_only_vs_global_plus_latent'), 'frozen_shape_field_CV_RMSE_delta')}`."
            )
        lines.extend(
            (
                "",
                f"- Best/final direction consistent: `{result['best_final_direction_consistent']}`; protocol passed: `{result['protocol_passed']}`.",
            )
        )
        for checkpoint in CHECKPOINTS:
            guard = _mapping(result[checkpoint], checkpoint).get("clean_guard")
            if isinstance(guard, Mapping):
                lines.append(f"- `{checkpoint}` clean guard: `{dict(guard)}`.")
        lines.append("")
    leakage = _mapping(reconstructed["duplicate_leakage"], "leakage")
    lines.extend(
        (
            "## Interpretation And Integrity",
            "",
            "- Global-physics adequacy means its selection-role family winner improves both scale log-RMSE and frozen-shape field CV-RMSE versus physics-only.",
            "- A pooled-latent gain is called stable only when its paired-bootstrap CI95 upper bound is below zero for both metrics; a non-gain is a result, not an excuse to use test roles.",
            "- `test_iid` and `hard_challenge_test` are descriptive output rows only; no selection, standardization, threshold, or model fitting uses them.",
            f"- Per-sample CSV: `{table['row_count']}` rows; SHA256 `{table['sha256']}`.",
            f"- Cross-role input/full/provenance duplicate groups: `{_mapping(leakage['cross_role_model_input_duplicate_groups'], 'input')['group_count']}` / `{_mapping(leakage['cross_role_full_sample_duplicate_groups'], 'full')['group_count']}` / `{_mapping(leakage['cross_role_provenance_duplicate_groups'], 'provenance')['group_count']}`.",
            f"- Overall Gate 4A feasibility pass: `{reconstructed['overall_gate4a_passed']}`.",
            "- `--verify-summary` rebuilds outcomes from CSV only; `--verify-models` recomputes every trained scalar prediction from committed model parameters plus CSV input/latent columns.",
            "",
        )
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_outputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    model_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
    table: Path,
    model_params: Path,
    summary_json: Path,
    summary_md: Path,
) -> None:
    _write_table(rows, columns, table)
    _write_json(model_params, model_payload)
    _write_json(summary_json, summary_payload)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(render_markdown(summary_payload), encoding="utf-8")


def _verify_summary(table: Path, summary_path: Path) -> dict[str, Any]:
    payload = _read_json(summary_path)
    table_info = _mapping(payload.get("per_sample_table"), "per_sample_table")
    columns = table_info.get("columns")
    if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
        raise AuditError("summary lacks a valid per-sample table column schema")
    rows = _read_table(table, columns)
    reconstructed = _build_reconstructed(rows)
    expected = _mapping(payload.get("reconstructed_from_table"), "reconstructed_from_table")
    _assert_close(reconstructed, expected)
    if table_info.get("sha256") != _sha256(table) or int(table_info.get("row_count", -1)) != len(rows):
        raise AuditError("table checksum or row count differs from summary")
    return {"audit_id": payload.get("audit_id"), "row_count": len(rows), "table_sha256": _sha256(table), "verification": "passed"}


def _verify_models(table: Path, model_params_path: Path) -> dict[str, Any]:
    payload = _read_json(model_params_path)
    columns = payload.get("per_sample_table_columns")
    if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
        raise AuditError("model parameter JSON lacks table columns")
    rows = _read_table(table, columns)
    records = _mapping(payload.get("model_records"), "model_records")
    max_error = 0.0
    for key, record_raw in records.items():
        record = _mapping(record_raw, f"model_records.{key}")
        protocol = str(record["protocol"])
        checkpoint = str(record["checkpoint"])
        candidate = str(record["candidate"])
        feature_columns = record.get("input_feature_columns")
        if not isinstance(feature_columns, list) or not all(isinstance(value, str) for value in feature_columns):
            raise AuditError(f"{key}: invalid input_feature_columns")
        if any(any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS) for column in feature_columns):
            raise AuditError(f"{key}: label-derived feature leaked into stored model")
        X = _matrix_from_rows(rows, feature_columns)
        predicted = _predict_model(record, X)
        output_column = f"{_candidate_prefix(protocol, checkpoint, candidate)}_delta_s_hat"
        expected = np.asarray([float(row[output_column]) for row in rows], dtype=np.float64)
        error = float(np.max(np.abs(predicted - expected)))
        max_error = max(max_error, error)
        if error > 1.0e-10:
            raise AuditError(f"{key}: model prediction reconstruction drift {error:.3e}")
        fit_roles = tuple(str(value) for value in record.get("fit_roles", []))
        if any(role in {"test_iid", "hard_challenge_test"} for role in fit_roles):
            raise AuditError(f"{key}: test role appears in model fit roles")
        fit_ids = set(str(value) for value in record.get("fit_sample_ids", []))
        row_by_id = {str(row["sample_id"]): row for row in rows}
        if any(row_by_id[sample_id]["role"] not in fit_roles for sample_id in fit_ids):
            raise AuditError(f"{key}: fit sample ID violates fit-role provenance")
    return {"audit_id": payload.get("audit_id"), "model_count": len(records), "max_delta_s_abs_error": max_error, "verification": "passed"}


def _run_audit(
    *,
    dataset: Path,
    split_map: Path,
    contract_path: Path,
    gate1_table: Path,
    gate3_table: Path,
    best_latents: Path,
    final_latents: Path,
    best_latent_manifest: Path,
    final_latent_manifest: Path,
    best_prediction_paths: Sequence[Path],
    final_prediction_paths: Sequence[Path],
    output_table: Path,
    output_model_params: Path,
    output_json: Path,
    output_md: Path,
    table_label: str | None,
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    _assert_contract(contract)
    _validate_feature_allowlist()
    assignments, roles, split_payload = _load_split_map(split_map, contract)
    expected_ids = set(assignments)
    input_rows, predecessor_provenance = _build_input_rows(
        gate1_table=gate1_table,
        gate3_table=gate3_table,
        assignments=assignments,
    )
    frozen = _mapping(contract.get("frozen_predecessors"), "frozen_predecessors")
    expected_gate1 = _mapping(frozen.get("gate1_table"), "frozen_predecessors.gate1_table").get("sha256")
    expected_gate3 = _mapping(frozen.get("gate3_table"), "frozen_predecessors.gate3_table").get("sha256")
    if predecessor_provenance["gate1_table"]["sha256"] != expected_gate1:
        raise AuditError("Gate 1 final table hash drift")
    if predecessor_provenance["gate3_table"]["sha256"] != expected_gate3:
        raise AuditError("Gate 3 final table hash drift")
    latents: dict[str, dict[str, np.ndarray]] = {}
    latent_provenance: dict[str, Any] = {}
    for checkpoint, path, manifest_path in (
        ("best", best_latents, best_latent_manifest),
        ("final", final_latents, final_latent_manifest),
    ):
        latents[checkpoint], latent_provenance[checkpoint] = _load_latents(path, expected_ids, checkpoint)
        latent_provenance[checkpoint] = _attach_latent_manifest(latent_provenance[checkpoint], manifest_path, checkpoint)
    latent_dims = {checkpoint: int(latent_provenance[checkpoint]["dimension"]) for checkpoint in CHECKPOINTS}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_provenance: dict[str, Any] = {}
    for checkpoint, paths in (("best", best_prediction_paths), ("final", final_prediction_paths)):
        predictions[checkpoint], prediction_provenance[checkpoint] = gate3._load_prediction_archives(
            paths, expected_ids, checkpoint
        )
    contexts = _build_field_contexts(dataset=dataset, input_rows=input_rows, predictions=predictions)
    rows = _table_rows_with_latents(input_rows, latents, latent_dims)
    for row in rows:
        row["_field_context"] = contexts[str(row["sample_id"])]
    model_records: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        for checkpoint in CHECKPOINTS:
            _fit_and_apply(
                rows=rows,
                protocol=protocol,
                checkpoint=checkpoint,
                latent_dim=latent_dims[checkpoint],
                model_records=model_records,
            )
    _validate_field_baselines(rows)
    columns = _base_columns(latent_dims)
    table_rows = _strip_internal_fields(rows, columns)
    reconstructed = _build_reconstructed(table_rows)
    model_payload = _model_params_payload(
        contract=contract,
        table_columns=columns,
        latent_provenance=latent_provenance,
        model_records=model_records,
        reconstructed=reconstructed,
    )
    summary_payload: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "offline_scalar_models_with_frozen_v4_shapes",
        "contract_id": contract.get("contract_id"),
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "dataset_path": dataset.as_posix(),
            "split_map_path": split_map.as_posix(),
            "sample_count": len(table_rows),
            "roles": roles,
            "role_counts": {role: int(sum(row["role"] == role for row in table_rows)) for role in roles},
        },
        "frozen_definitions": {
            "s_phys": "raw_z_collapsed_1d_operator_K",
            "target_delta_s": "log(s_true_K / s_phys_K)",
            "corrected_scale": "s_phys_K * exp(delta_s_hat)",
            "field_reconstruction": "frozen V4 shape with corrected scale, followed by raw-space Dirichlet projection",
            "shape_model_training": False,
        },
        "input_leakage_guard": {
            "global_feature_allowlist": list(GLOBAL_FEATURES),
            "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
            "target_or_oracle_inputs_used": False,
            "test_roles_used_for_fit_or_selection": False,
        },
        "predecessor_provenance": predecessor_provenance,
        "pooled_latent_provenance": latent_provenance,
        "frozen_prediction_archives": prediction_provenance,
        "read_only_guardrails": {
            "RIGNO_parameter_changes": 0,
            "formal_training_runs": 0,
            "shape_model_training_runs": 0,
            "data_or_label_writes": 0,
            "permitted_writes": ["explicit Gate 4A CSV", "model parameters JSON", "summary JSON", "closeout Markdown"],
        },
        "per_sample_table": {
            "path": table_label or output_table.as_posix(),
            "sha256": None,
            "row_count": len(table_rows),
            "columns": columns,
        },
        "model_parameters": {
            "path": output_model_params.as_posix(),
            "sha256": None,
            "model_count": len(model_records),
        },
        "reconstructed_from_table": reconstructed,
    }
    # Write table and model parameters first, then hash both frozen artifacts in
    # the summary.  The model JSON is self-sufficient for scalar prediction
    # reconstruction from CSV global/latent inputs.
    _write_table(table_rows, columns, output_table)
    _write_json(output_model_params, model_payload)
    summary_payload["per_sample_table"]["sha256"] = _sha256(output_table)
    summary_payload["model_parameters"]["sha256"] = _sha256(output_model_params)
    _write_json(output_json, summary_payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary_payload), encoding="utf-8")
    return summary_payload


def _dry_run(
    *,
    split_map: Path,
    contract_path: Path,
    gate1_table: Path,
    gate3_table: Path,
    best_latents: Path,
    final_latents: Path,
    best_latent_manifest: Path,
    final_latent_manifest: Path,
    best_prediction_paths: Sequence[Path],
    final_prediction_paths: Sequence[Path],
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    _assert_contract(contract)
    _validate_feature_allowlist()
    assignments, roles, split_payload = _load_split_map(split_map, contract)
    expected_ids = set(assignments)
    _input_rows, predecessor_provenance = _build_input_rows(
        gate1_table=gate1_table, gate3_table=gate3_table, assignments=assignments
    )
    frozen = _mapping(contract.get("frozen_predecessors"), "frozen_predecessors")
    if predecessor_provenance["gate1_table"]["sha256"] != _mapping(frozen.get("gate1_table"), "gate1_table").get("sha256"):
        raise AuditError("Gate 1 final table hash drift")
    if predecessor_provenance["gate3_table"]["sha256"] != _mapping(frozen.get("gate3_table"), "gate3_table").get("sha256"):
        raise AuditError("Gate 3 final table hash drift")
    latent_report: dict[str, Any] = {}
    prediction_report: dict[str, Any] = {}
    for checkpoint, latent_path, manifest_path, prediction_paths in (
        ("best", best_latents, best_latent_manifest, best_prediction_paths),
        ("final", final_latents, final_latent_manifest, final_prediction_paths),
    ):
        _latents, latent_report[checkpoint] = _load_latents(latent_path, expected_ids, checkpoint)
        latent_report[checkpoint] = _attach_latent_manifest(latent_report[checkpoint], manifest_path, checkpoint)
        _predictions, prediction_report[checkpoint] = gate3._load_prediction_archives(
            prediction_paths, expected_ids, checkpoint
        )
    return {
        "audit_id": AUDIT_ID,
        "mode": "dry_run",
        "read_only": True,
        "dataset": {
            "dataset_id": split_payload.get("dataset_id"),
            "sample_count": len(assignments),
            "role_counts": {role: int(sum(value == role for value in assignments.values())) for role in roles},
        },
        "global_feature_allowlist": list(GLOBAL_FEATURES),
        "predecessor_sha256": {
            "gate1": predecessor_provenance["gate1_table"]["sha256"],
            "gate3": predecessor_provenance["gate3_table"]["sha256"],
        },
        "pooled_latents": latent_report,
        "frozen_prediction_archives": prediction_report,
        "planned_writes": [],
        "formal_training_runs": 0,
        "RIGNO_parameter_changes": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--gate1-table", type=Path)
    parser.add_argument("--gate3-table", type=Path)
    parser.add_argument("--best-latents", type=Path)
    parser.add_argument("--final-latents", type=Path)
    parser.add_argument("--best-latent-manifest", type=Path)
    parser.add_argument("--final-latent-manifest", type=Path)
    parser.add_argument("--best-predictions", type=Path, action="append", default=[])
    parser.add_argument("--final-predictions", type=Path, action="append", default=[])
    parser.add_argument("--output-table", type=Path)
    parser.add_argument("--output-model-params", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--table-label")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-summary", action="store_true")
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--model-params", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_summary:
            if args.table is None or args.summary_json is None:
                raise AuditError("--verify-summary requires --table and --summary-json")
            print(json.dumps(_verify_summary(args.table, args.summary_json), indent=2, sort_keys=True))
            return 0
        if args.verify_models:
            if args.table is None or args.model_params is None:
                raise AuditError("--verify-models requires --table and --model-params")
            print(json.dumps(_verify_models(args.table, args.model_params), indent=2, sort_keys=True))
            return 0
        required = (
            args.dataset,
            args.split_map,
            args.contract,
            args.gate1_table,
            args.gate3_table,
            args.best_latents,
            args.final_latents,
            args.best_latent_manifest,
            args.final_latent_manifest,
        )
        if any(value is None for value in required):
            raise AuditError("audit requires dataset/split/contract/predecessor tables/best and final latents")
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run(
                        split_map=args.split_map,
                        contract_path=args.contract,
                        gate1_table=args.gate1_table,
                        gate3_table=args.gate3_table,
                        best_latents=args.best_latents,
                        final_latents=args.final_latents,
                        best_latent_manifest=args.best_latent_manifest,
                        final_latent_manifest=args.final_latent_manifest,
                        best_prediction_paths=args.best_predictions,
                        final_prediction_paths=args.final_predictions,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        table, model_params, summary_json, summary_md = _output_paths(
            (args.output_table, args.output_model_params, args.output_json, args.output_md), args.overwrite
        )
        _run_audit(
            dataset=args.dataset,
            split_map=args.split_map,
            contract_path=args.contract,
            gate1_table=args.gate1_table,
            gate3_table=args.gate3_table,
            best_latents=args.best_latents,
            final_latents=args.final_latents,
            best_latent_manifest=args.best_latent_manifest,
            final_latent_manifest=args.final_latent_manifest,
            best_prediction_paths=args.best_predictions,
            final_prediction_paths=args.final_predictions,
            output_table=table,
            output_model_params=model_params,
            output_json=summary_json,
            output_md=summary_md,
            table_label=args.table_label,
        )
    except AuditError as exc:
        print(f"Gate 4A audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
