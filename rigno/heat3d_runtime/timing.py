"""Timing lifecycle contract for the V7 reference inference path.

This module defines stage boundaries only.  It deliberately does not load
truth, calculate metrics, or make a performance claim.  A future timing
collector can use the contract without changing the stable inference
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping


TIMING_STAGE_ORDER = (
    "preprocessing_feature",
    "graph_build",
    "compile_warmup",
    "model_forward",
    "reconstruction",
    "synchronized_result",
)
EXCLUDED_FROM_LATENCY = (
    "truth_loading",
    "metrics",
    "accuracy_audit",
)
LIFECYCLE_STATES = ("fresh", "resident", "Q1", "Q2", "throughput")
WORKLOAD_BOUNDARY = (
    "k/q/BC",
    "preprocessing_feature",
    "graph_build",
    "model_forward",
    "reconstruction",
    "synchronized_result_240825_field",
)


@dataclass
class TimingLifecycle:
    """One lifecycle record with an explicit stage order and state."""

    state: str
    query_count: int = 1
    stages: list[dict[str, Any]] = field(default_factory=list)
    _last_stage_index: int = -1

    def __post_init__(self) -> None:
        if self.state not in LIFECYCLE_STATES:
            raise ValueError(f"unsupported timing lifecycle state: {self.state!r}")
        if self.query_count < 1:
            raise ValueError("query_count must be positive")

    def record(self, stage: str, elapsed_seconds: float) -> None:
        if stage in EXCLUDED_FROM_LATENCY:
            raise ValueError(f"excluded operation cannot be a latency stage: {stage}")
        try:
            index = TIMING_STAGE_ORDER.index(stage)
        except ValueError as exc:
            raise ValueError(f"unknown timing stage: {stage!r}") from exc
        if index != self._last_stage_index + 1:
            raise ValueError(
                f"timing stages must be recorded in order; expected "
                f"{TIMING_STAGE_ORDER[self._last_stage_index + 1]!r}, got {stage!r}"
            )
        value = float(elapsed_seconds)
        if value < 0.0:
            raise ValueError("elapsed_seconds must be non-negative")
        self.stages.append({"stage": stage, "elapsed_seconds": value})
        self._last_stage_index = index

    def complete(self) -> dict[str, Any]:
        if self._last_stage_index != len(TIMING_STAGE_ORDER) - 1:
            missing = TIMING_STAGE_ORDER[self._last_stage_index + 1 :]
            raise ValueError(f"timing lifecycle is incomplete; missing={missing}")
        return {
            "state": self.state,
            "query_count": self.query_count,
            "stages": list(self.stages),
            "latency_boundary": list(WORKLOAD_BOUNDARY),
            "excluded_from_latency": list(EXCLUDED_FROM_LATENCY),
        }


class TimingCore:
    """Create and validate timing lifecycle records without running workloads."""

    @staticmethod
    def begin(*, state: str, query_count: int = 1) -> TimingLifecycle:
        return TimingLifecycle(state=state, query_count=query_count)

    @staticmethod
    def contract() -> dict[str, Any]:
        return {
            "stage_order": list(TIMING_STAGE_ORDER),
            "lifecycle_states": list(LIFECYCLE_STATES),
            "workload_boundary": list(WORKLOAD_BOUNDARY),
            "excluded_from_latency": list(EXCLUDED_FROM_LATENCY),
            "truth_loading_in_latency": False,
            "metrics_in_latency": False,
            "accuracy_audit_in_latency": False,
            "headline_speedup_claim": False,
            "formal_performance_experiment": False,
        }

    @staticmethod
    def validate_record(record: Mapping[str, Any]) -> None:
        if record.get("latency_boundary") != list(WORKLOAD_BOUNDARY):
            raise ValueError("timing record workload boundary drifted")
        if any(stage.get("stage") in EXCLUDED_FROM_LATENCY for stage in record.get("stages", ())):
            raise ValueError("timing record includes an excluded operation")
        observed = [stage.get("stage") for stage in record.get("stages", ())]
        if observed != list(TIMING_STAGE_ORDER):
            raise ValueError("timing record stage order drifted")


def monotonic_seconds() -> float:
    """Return the monotonic clock used by future collectors."""

    return time.perf_counter()
