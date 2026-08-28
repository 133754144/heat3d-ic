"""Explicit temperature-representation contracts for V7 formal evaluation.

The frozen V6/P1i model and high-N scale path operate in absolute temperature
(``temperature_K``).  The frozen full-field metrics operate in temperature
rise (``deltaT_K``).  This module keeps the conversion explicit and
side-effect free: EvaluationCore never infers a representation from values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np


ABSOLUTE_TEMPERATURE_K = "absolute_temperature_K"
DELTA_T_K = "deltaT_K"
MODEL_RAW_STAGE = "model_raw"
HIGH_N_SCALED_STAGE = "high_n_scaled_output"
FORMAL_DIRECT_QUERY_STAGE = "formal_direct_query"
FORMAL_RECONSTRUCTED_FULL_FIELD_STAGE = "formal_reconstructed_full_field"


class TemperatureRepresentationError(ValueError):
    """Raised when a prediction has no explicit or compatible representation."""


def temperature_K_to_deltaT_K(
    temperature_K: Any,
    *,
    reference_temperature_K: float,
) -> np.ndarray:
    """Convert an absolute-temperature array to temperature rise explicitly."""

    reference = float(reference_temperature_K)
    if not math.isfinite(reference):
        raise TemperatureRepresentationError("reference_temperature_K must be finite")
    temperature = np.asarray(temperature_K, dtype=np.float64)
    if not np.all(np.isfinite(temperature)):
        raise TemperatureRepresentationError("temperature_K must be finite")
    return temperature - reference


def _contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash the canonical contract payload excluding its self-binding field."""

    payload = dict(value)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required(mapping: Mapping[str, Any], field: str, scope: str) -> Any:
    if field not in mapping or mapping[field] is None:
        raise TemperatureRepresentationError(f"{scope}.{field} is required")
    return mapping[field]


def validate_temperature_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a loaded machine-readable representation contract fail-closed."""

    value = dict(contract)
    if value.get("schema_version") != "heat3d_v7_temperature_representation_v1":
        raise TemperatureRepresentationError("unsupported temperature contract schema")
    reference = float(_required(value, "reference_temperature_K", "contract"))
    if not math.isfinite(reference):
        raise TemperatureRepresentationError("contract reference_temperature_K must be finite")
    stages = _required(value, "prediction_stages", "contract")
    if not isinstance(stages, Mapping):
        raise TemperatureRepresentationError("contract.prediction_stages must be a mapping")
    expected_stages = {
        MODEL_RAW_STAGE: ABSOLUTE_TEMPERATURE_K,
        HIGH_N_SCALED_STAGE: ABSOLUTE_TEMPERATURE_K,
        FORMAL_DIRECT_QUERY_STAGE: DELTA_T_K,
        FORMAL_RECONSTRUCTED_FULL_FIELD_STAGE: DELTA_T_K,
    }
    if dict(stages) != expected_stages:
        raise TemperatureRepresentationError("contract prediction stage semantics drifted")
    adapter = _required(value, "adapter", "contract")
    if not isinstance(adapter, Mapping) or dict(adapter) != {
        "adapter_id": "temperature_K_to_deltaT_K",
        "input_representation": ABSOLUTE_TEMPERATURE_K,
        "output_representation": DELTA_T_K,
        "formula": "deltaT_K = temperature_K - reference_temperature_K",
    }:
        raise TemperatureRepresentationError("contract adapter semantics drifted")
    formal = _required(value, "formal_evaluation_input", "contract")
    if not isinstance(formal, Mapping):
        raise TemperatureRepresentationError("contract.formal_evaluation_input must be a mapping")
    prediction_representation = _required(
        formal, "prediction_representation", "contract.formal_evaluation_input"
    )
    prediction_stage = _required(
        formal, "prediction_stage", "contract.formal_evaluation_input"
    )
    formal_reference = float(
        _required(formal, "reference_temperature_K", "contract.formal_evaluation_input")
    )
    if prediction_representation != DELTA_T_K:
        raise TemperatureRepresentationError("formal evaluation must receive deltaT_K")
    if prediction_stage not in {
        FORMAL_DIRECT_QUERY_STAGE,
        FORMAL_RECONSTRUCTED_FULL_FIELD_STAGE,
    }:
        raise TemperatureRepresentationError("unsupported formal prediction stage")
    if stages[prediction_stage] != DELTA_T_K or formal_reference != reference:
        raise TemperatureRepresentationError("formal temperature contract is inconsistent")
    sha256 = _required(value, "contract_sha256", "contract")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise TemperatureRepresentationError("contract_sha256 must be a SHA256 digest")
    return value


def load_temperature_contract(path: str | Path) -> dict[str, Any]:
    """Load a tracked contract and bind it to its exact source-file SHA256."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TemperatureRepresentationError("temperature contract must be an object")
    value = dict(payload)
    expected_sha256 = _contract_sha256(value)
    declared = value.get("contract_sha256")
    if declared != expected_sha256:
        raise TemperatureRepresentationError("temperature contract SHA binding drifted")
    return validate_temperature_contract(value)


def validate_formal_prediction_representation(
    *,
    prediction_representation: str,
    reference_temperature_K: float,
    prediction_stage: str,
    temperature_contract: Mapping[str, Any],
) -> None:
    """Require a formal record to match the frozen deltaT evaluation contract."""

    contract = validate_temperature_contract(temperature_contract)
    if prediction_representation != DELTA_T_K:
        raise TemperatureRepresentationError(
            "formal PredictionOnlyRecord requires prediction_representation='deltaT_K'"
        )
    if prediction_stage not in {
        FORMAL_DIRECT_QUERY_STAGE,
        FORMAL_RECONSTRUCTED_FULL_FIELD_STAGE,
    }:
        raise TemperatureRepresentationError("formal prediction_stage is not registered")
    if contract["prediction_stages"][prediction_stage] != prediction_representation:
        raise TemperatureRepresentationError("formal prediction representation/stage drifted")
    if float(reference_temperature_K) != float(contract["reference_temperature_K"]):
        raise TemperatureRepresentationError("formal prediction reference temperature drifted")
