"""Timing lifecycle contract for the V7 reference inference path.

This module defines stage boundaries only.  It deliberately does not load
truth, calculate metrics, or make a performance claim.  A future timing
collector can use the contract without changing the stable inference
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from collections.abc import Callable, Mapping
from typing import Any


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

    def measure(
        self,
        stage: str,
        operation: Callable[..., Any],
        *args: Any,
        synchronize: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one existing operation and record only its contract stage.

        ``synchronize`` is explicit so a caller cannot accidentally move the
        synchronization boundary into truth loading, metrics, or an earlier
        stage.  The operation itself is supplied by the stable inference
        caller; this helper does not cache, batch, compile, or alter it.
        """

        started = monotonic_seconds()
        result = operation(*args, **kwargs)
        if synchronize is not None:
            synchronize(result)
        self.record(stage, monotonic_seconds() - started)
        return result

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
    """Create, run, and validate the frozen timing lifecycle."""

    @staticmethod
    def begin(*, state: str, query_count: int = 1) -> TimingLifecycle:
        return TimingLifecycle(state=state, query_count=query_count)

    @staticmethod
    def run(
        *,
        state: str,
        steps: Mapping[str, Callable[[Mapping[str, Any]], Any]],
        query_count: int = 1,
        synchronizers: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute each real stage callback once under the frozen boundary.

        The callbacks are intentionally dependency-injected.  A production
        caller passes the already-defined feature, graph, model, and
        reconstruction operations; this method only supplies lifecycle
        accounting and forwards prior stage results.
        """

        expected = set(TIMING_STAGE_ORDER)
        observed = set(steps)
        if observed != expected:
            raise ValueError(
                "timing steps must match the frozen stage order; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        lifecycle = TimingCore.begin(state=state, query_count=query_count)
        results: dict[str, Any] = {}
        synchronizers = {} if synchronizers is None else dict(synchronizers)
        for stage in TIMING_STAGE_ORDER:
            operation = steps[stage]
            synchronizer = synchronizers.get(stage)
            results[stage] = lifecycle.measure(
                stage,
                operation,
                dict(results),
                synchronize=synchronizer,
            )
        record = lifecycle.complete()
        TimingCore.validate_record(record)
        return record, results

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
