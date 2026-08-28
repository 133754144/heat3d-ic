"""Training-side Level-A validation adapter for the frozen V7 metric core.

This module is deliberately small: the trainer supplies prediction-only model
outputs and prepared valid batches, while :class:`EvaluationCore` remains the
single implementation of metric definitions and sufficient statistics.  It
never enumerates or loads ``test_iid``/sealed rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_runtime.evaluation import EvaluationCore, EvaluationSample

from .p1i import prediction_to_raw_delta


def _layer_ids(example: Any) -> np.ndarray:
    """Derive layer IDs from frozen geometry metadata, never from temperature."""

    layers = example.meta.get("layers_bottom_to_top")
    if layers is None:
        physics = example.meta.get("physics") or {}
        layers = physics.get("layers_bottom_to_top")
    if not layers:
        raise ValueError(f"{example.sample_id}: missing layer metadata for Level-A metrics")
    z = np.asarray(example.condition.coords, dtype=np.float64)[:, 2]
    origin = float(np.min(z))
    boundaries = origin + np.concatenate(
        [np.asarray([0.0]), np.cumsum([float(row["thickness_m"]) for row in layers])]
    )
    result = np.searchsorted(boundaries[1:-1], z, side="right").astype(np.int32)
    if result.shape != (len(z),) or np.any(result < 0) or np.any(result >= len(layers)):
        raise ValueError(f"{example.sample_id}: derived layer IDs are invalid")
    return result


def evaluate_level_a_validation(
    *,
    predictions: Sequence[Any],
    batches: Sequence[Any],
    examples: Sequence[Any],
    stats: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Evaluate one valid_iid population at training/native resolution."""

    by_id = {str(example.sample_id): example for example in examples}
    samples: list[EvaluationSample] = []
    if len(predictions) != len(batches):
        raise ValueError("prediction and validation-batch counts differ")
    for batch_predictions, batch in zip(predictions, batches, strict=True):
        if len(batch_predictions) != len(batch.groups):
            raise ValueError(f"{batch.batch_id}: prediction/group counts differ")
        for prediction, group in zip(batch_predictions, batch.groups, strict=True):
            raw_prediction = prediction_to_raw_delta(
                prediction, variant=variant, stats=stats
            )
            target = np.asarray(group["target_delta_raw"], dtype=np.float64)
            if raw_prediction.shape != target.shape:
                raise ValueError(
                    f"{batch.batch_id}: prediction shape {raw_prediction.shape} "
                    f"does not match target {target.shape}"
                )
            raw_prediction = raw_prediction[:, 0, :, 0]
            target = target[:, 0, :, 0]
            for row, sample_id in enumerate(batch.sample_ids):
                example = by_id[str(sample_id)]
                relative = example.get_relative_bc_feature_view()
                samples.append(
                    EvaluationSample(
                        sample_id=str(sample_id),
                        prediction_deltaT_K=raw_prediction[row],
                        truth_deltaT_K=target[row],
                        control_volumes_m3=example.v6_operator_point_weights(),
                        coords=np.asarray(example.condition.coords, dtype=np.float64),
                        layer_id=_layer_ids(example),
                        q_W_m3=np.asarray(relative.condition_features[:, 3], dtype=np.float64),
                    )
                )
    return EvaluationCore().evaluate(samples)


__all__ = ["evaluate_level_a_validation"]
