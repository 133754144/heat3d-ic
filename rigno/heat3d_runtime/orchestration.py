"""Formal V7 prediction-to-evaluation orchestration.

The stable inference runtime produces prediction-only records.  This module
joins those records to an explicitly supplied frozen ``valid_iid`` truth
provider and delegates all accuracy work to :class:`EvaluationCore`.  It does
not load a checkpoint, choose a route, rebuild a reconstruction map, or read
labels from an inference object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_runtime.evaluation import (
    METRIC_SCHEMA_VERSION,
    EvaluationCore,
    EvaluationSample,
)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionOnlyRecord:
    """Prediction and label-independent evaluation metadata from inference."""

    sample_id: str
    prediction_deltaT_K: np.ndarray
    coords: np.ndarray
    control_volumes_m3: np.ndarray
    layer_id: np.ndarray
    q_W_m3: np.ndarray
    route_id: str
    anchor_context_resolution: int
    encoder_input_resolution: int
    output_query_resolution: int
    reconstruction_resolution: int
    direct_query: bool
    reconstruction_map_sha256: str | None = None
    reconstruction_contract_sha256: str | None = None
    prediction_artifact_sha256: str | None = None
    split: str = "valid_iid"

    def validated(self) -> "PredictionOnlyRecord":
        if self.split != "valid_iid":
            raise ValueError("formal evaluation accepts only valid_iid predictions")
        if not self.sample_id or not self.route_id:
            raise ValueError("sample_id and route_id are required")
        if not isinstance(self.direct_query, bool):
            raise ValueError("direct_query must be boolean")
        prediction = np.asarray(self.prediction_deltaT_K, dtype=np.float64).reshape(-1)
        coords = np.asarray(self.coords, dtype=np.float64)
        weights = np.asarray(self.control_volumes_m3, dtype=np.float64).reshape(-1)
        layer_id = np.asarray(self.layer_id, dtype=np.int32).reshape(-1)
        q = np.asarray(self.q_W_m3, dtype=np.float64).reshape(-1)
        if (
            prediction.size == 0
            or coords.shape != (prediction.size, 3)
            or weights.shape != (prediction.size,)
            or layer_id.shape != (prediction.size,)
            or q.shape != (prediction.size,)
            or not np.all(np.isfinite(prediction))
            or not np.all(np.isfinite(coords))
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.all(np.isfinite(q))
        ):
            raise ValueError("prediction-only record has invalid evaluation arrays")
        return PredictionOnlyRecord(
            sample_id=str(self.sample_id),
            prediction_deltaT_K=prediction,
            coords=coords,
            control_volumes_m3=weights,
            layer_id=layer_id,
            q_W_m3=q,
            route_id=str(self.route_id),
            anchor_context_resolution=int(self.anchor_context_resolution),
            encoder_input_resolution=int(self.encoder_input_resolution),
            output_query_resolution=int(self.output_query_resolution),
            reconstruction_resolution=int(self.reconstruction_resolution),
            direct_query=self.direct_query,
            reconstruction_map_sha256=self.reconstruction_map_sha256,
            reconstruction_contract_sha256=self.reconstruction_contract_sha256,
            prediction_artifact_sha256=(
                self.prediction_artifact_sha256 or _array_sha256(prediction)
            ),
            split="valid_iid",
        )

    def descriptor(self) -> dict[str, Any]:
        value = self.validated()
        return {
            "sample_id": value.sample_id,
            "route_id": value.route_id,
            "anchor_context_resolution": value.anchor_context_resolution,
            "encoder_input_resolution": value.encoder_input_resolution,
            "output_query_resolution": value.output_query_resolution,
            "reconstruction_resolution": value.reconstruction_resolution,
            "direct_query": value.direct_query,
            "prediction_artifact_sha256": value.prediction_artifact_sha256,
            "reconstruction_map_sha256": value.reconstruction_map_sha256,
            "reconstruction_contract_sha256": value.reconstruction_contract_sha256,
            "prediction_node_count": int(value.prediction_deltaT_K.size),
            "split": value.split,
        }


TruthLoader = Callable[[str], Mapping[str, Any]]


class FormalEvaluationOrchestrator:
    """The only V7 accuracy orchestration boundary."""

    execution_role = "production_inference"
    evaluation_split = "valid_iid"

    def __init__(self, evaluation_core: EvaluationCore | None = None) -> None:
        self.evaluation_core = EvaluationCore() if evaluation_core is None else evaluation_core

    def run(
        self,
        predictions: Sequence[PredictionOnlyRecord],
        *,
        truth_loader: TruthLoader,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Join prediction-only inference records to frozen truth after inference."""

        if not predictions:
            raise ValueError("formal evaluation requires at least one prediction")
        records = [row.validated() for row in predictions]
        sample_ids = [row.sample_id for row in records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("formal evaluation sample IDs must be unique")
        if any(row.split != self.evaluation_split for row in records):
            raise ValueError("formal evaluation prediction split drifted")
        route_ids = {row.route_id for row in records}
        if len(route_ids) != 1:
            raise ValueError("one formal evaluation receipt cannot mix route IDs")
        route_id = next(iter(route_ids))
        receipt_value = _validate_receipt_binding(
            receipt,
            records,
            route_id=route_id,
        )

        truth_rows = self.evaluation_core.load_valid_truth(sample_ids, truth_loader)
        samples: list[EvaluationSample] = []
        for record, truth in zip(records, truth_rows, strict=True):
            required = {"truth_deltaT_K"}
            missing = sorted(required - set(truth))
            if missing:
                raise ValueError(f"truth provider missing fields: {missing}")
            samples.append(
                EvaluationSample(
                    sample_id=record.sample_id,
                    prediction_deltaT_K=record.prediction_deltaT_K,
                    truth_deltaT_K=truth["truth_deltaT_K"],
                    control_volumes_m3=record.control_volumes_m3,
                    coords=record.coords,
                    layer_id=record.layer_id,
                    q_W_m3=record.q_W_m3,
                    split=self.evaluation_split,
                )
            )
        evaluation = self.evaluation_core.evaluate(samples, receipt=receipt_value)
        return {
            "schema_version": "heat3d_v7_formal_evaluation_orchestration_v1",
            "execution_role": self.execution_role,
            "evaluation_split": self.evaluation_split,
            "route_id": route_id,
            "prediction_only": True,
            "labels_read_by_inference": False,
            "truth_loaded_by": "EvaluationCore",
            "model_selection": False,
            "scientific_evidence_eligible": False,
            "prediction_records": [row.descriptor() for row in records],
            "evaluation": evaluation,
        }


def _validate_receipt_binding(
    receipt: Mapping[str, Any],
    records: Sequence[PredictionOnlyRecord],
    *,
    route_id: str,
) -> dict[str, Any]:
    """Require a formal receipt to bind every semantic route dimension.

    The inference side may not infer missing receipt fields.  In particular,
    a caller cannot silently turn a high-resolution direct query into a native
    or reconstruction-only evaluation by omitting a resolution field.
    """

    value = dict(receipt)
    required = (
        "route_id",
        "anchor_context_resolution",
        "encoder_input_resolution",
        "output_query_resolution",
        "reconstruction_resolution",
        "prediction_artifact_sha256_by_sample",
        "reconstruction_contract_sha256",
        "metric_schema_version",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"formal receipt is missing required bindings: {missing}")
    if value["route_id"] != route_id:
        raise ValueError("formal receipt route_id does not match predictions")
    expected = {
        field: getattr(records[0], field)
        for field in (
            "anchor_context_resolution",
            "encoder_input_resolution",
            "output_query_resolution",
            "reconstruction_resolution",
        )
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"formal receipt {field} does not match predictions")
        if any(getattr(row, field) != expected_value for row in records):
            raise ValueError(f"formal predictions have inconsistent {field}")
    expected_prediction_hashes = {
        row.sample_id: row.prediction_artifact_sha256 for row in records
    }
    if value["prediction_artifact_sha256_by_sample"] != expected_prediction_hashes:
        raise ValueError("formal receipt prediction artifact binding does not match")
    if value["metric_schema_version"] != METRIC_SCHEMA_VERSION:
        raise ValueError("formal receipt metric schema does not match EvaluationCore")
    return value
