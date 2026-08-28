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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


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
    checkpoint_sha256: str | None = None
    checkpoint_epoch: int | None = None
    dataset_manifest_sha256: str | None = None
    full_field_archive_sha256: str | None = None
    frozen_valid32_ids: tuple[str, ...] | None = None
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
        computed_prediction_sha256 = _array_sha256(prediction)
        if (
            self.prediction_artifact_sha256 is not None
            and self.prediction_artifact_sha256 != computed_prediction_sha256
        ):
            raise ValueError("prediction artifact SHA does not match prediction array")
        if self.checkpoint_sha256 is not None and not _is_sha256(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a SHA256 hex digest")
        if self.dataset_manifest_sha256 is not None and not _is_sha256(
            self.dataset_manifest_sha256
        ):
            raise ValueError("dataset_manifest_sha256 must be a SHA256 hex digest")
        if self.full_field_archive_sha256 is not None and not _is_sha256(
            self.full_field_archive_sha256
        ):
            raise ValueError("full_field_archive_sha256 must be a SHA256 hex digest")
        if self.checkpoint_epoch is not None and int(self.checkpoint_epoch) < 0:
            raise ValueError("checkpoint_epoch must be non-negative")
        valid32_ids = None
        if self.frozen_valid32_ids is not None:
            valid32_ids = tuple(map(str, self.frozen_valid32_ids))
            if not valid32_ids or len(set(valid32_ids)) != len(valid32_ids):
                raise ValueError("frozen_valid32_ids must be a non-empty unique sequence")
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
            prediction_artifact_sha256=computed_prediction_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_epoch=(None if self.checkpoint_epoch is None else int(self.checkpoint_epoch)),
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            full_field_archive_sha256=self.full_field_archive_sha256,
            frozen_valid32_ids=valid32_ids,
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
            "checkpoint_sha256": value.checkpoint_sha256,
            "checkpoint_epoch": value.checkpoint_epoch,
            "dataset_manifest_sha256": value.dataset_manifest_sha256,
            "full_field_archive_sha256": value.full_field_archive_sha256,
            "frozen_valid32_ids": list(value.frozen_valid32_ids or ()),
            "split": value.split,
        }


TruthLoader = Callable[[str], Mapping[str, Any]]


class FormalEvaluationOrchestrator:
    """The only V7 accuracy orchestration boundary."""

    execution_role = "compatibility_audit"
    inference_execution_role = "production_inference"
    evaluation_split = "valid_iid"

    def __init__(self, evaluation_core: EvaluationCore | None = None) -> None:
        self.evaluation_core = EvaluationCore() if evaluation_core is None else evaluation_core

    def run(
        self,
        predictions: Sequence[PredictionOnlyRecord],
        *,
        truth_loader: TruthLoader,
        receipt: Mapping[str, Any],
        route_contract: Mapping[str, Any],
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
            route_contract=route_contract,
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
            "inference_execution_role": self.inference_execution_role,
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
    route_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a formal receipt to bind every semantic route dimension.

    The inference side may not infer missing receipt fields.  In particular,
    a caller cannot silently turn a high-resolution direct query into a native
    or reconstruction-only evaluation by omitting a resolution field.
    """

    value = dict(receipt)
    contract_required = (
        "route_id",
        "strategy_name",
        "anchor_context_resolution",
        "encoder_input_resolution",
        "output_query_resolution",
        "reconstruction_resolution",
        "direct_query",
    )
    missing_contract = [field for field in contract_required if field not in route_contract]
    if missing_contract:
        raise ValueError(f"formal route contract is missing required bindings: {missing_contract}")
    required = (
        "execution_role",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "dataset_manifest_sha256",
        "full_field_archive_sha256",
        "frozen_valid32_ids",
        "route_id",
        "anchor_context_resolution",
        "encoder_input_resolution",
        "output_query_resolution",
        "reconstruction_resolution",
        "direct_query",
        "prediction_artifact_sha256_by_sample",
        "reconstruction_contract_sha256",
        "reconstruction_map_sha256_by_sample",
        "metric_schema_version",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"formal receipt is missing required bindings: {missing}")
    if value["execution_role"] != "compatibility_audit":
        raise ValueError("formal replay receipt must use compatibility_audit role")
    if value["route_id"] != route_id or route_contract["route_id"] != route_id:
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
        if route_contract[field] != expected_value:
            raise ValueError(f"registered route contract {field} does not match predictions")
        if any(getattr(row, field) != expected_value for row in records):
            raise ValueError(f"formal predictions have inconsistent {field}")
    if value["direct_query"] != route_contract["direct_query"]:
        raise ValueError("formal receipt direct_query does not match registered route")
    if any(row.direct_query != value["direct_query"] for row in records):
        raise ValueError("formal predictions have inconsistent direct_query")
    for field in (
        "checkpoint_sha256",
        "checkpoint_epoch",
        "dataset_manifest_sha256",
        "full_field_archive_sha256",
    ):
        observed = {getattr(row, field) for row in records}
        if len(observed) != 1 or None in observed or value[field] != next(iter(observed)):
            raise ValueError(f"formal receipt {field} does not match frozen predictions")
    frozen_id_sets = {tuple(row.frozen_valid32_ids or ()) for row in records}
    if len(frozen_id_sets) != 1 or not frozen_id_sets or () in frozen_id_sets:
        raise ValueError("formal predictions are missing the frozen valid32 ID binding")
    frozen_ids = next(iter(frozen_id_sets))
    if tuple(value["frozen_valid32_ids"]) != frozen_ids:
        raise ValueError("formal receipt frozen valid32 IDs do not match predictions")
    expected_prediction_hashes = {
        row.sample_id: row.prediction_artifact_sha256 for row in records
    }
    if value["prediction_artifact_sha256_by_sample"] != expected_prediction_hashes:
        raise ValueError("formal receipt prediction artifact binding does not match")
    expected_map_hashes = {
        row.sample_id: row.reconstruction_map_sha256 for row in records
    }
    if any(hash_value is None for hash_value in expected_map_hashes.values()):
        raise ValueError("formal predictions are missing reconstruction map SHA bindings")
    if value["reconstruction_map_sha256_by_sample"] != expected_map_hashes:
        raise ValueError("formal receipt reconstruction map binding does not match")
    contracts = {row.reconstruction_contract_sha256 for row in records}
    if len(contracts) != 1 or None in contracts:
        raise ValueError("formal predictions are missing reconstruction contract SHA")
    if value["reconstruction_contract_sha256"] != next(iter(contracts)):
        raise ValueError("formal receipt reconstruction contract binding does not match")
    if value["metric_schema_version"] != METRIC_SCHEMA_VERSION:
        raise ValueError("formal receipt metric schema does not match EvaluationCore")
    return value
