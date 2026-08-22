#!/usr/bin/env python3
"""Shared fail-closed records for the frozen V6 publication runtime."""
from __future__ import annotations

from typing import Any


REQUIRED_FAILURE_FIELDS = (
    "original_exception_type", "original_exception", "sample_id",
    "order_position", "completed_count", "failure_stage",
    "residual_seconds", "residual_limit_seconds", "completed_rows",
)


def failure_record(
    exc: BaseException, *, sample_id: str, order_position: int,
    completed_rows: list[dict[str, Any]], failure_stage: str,
) -> dict[str, Any]:
    detail = dict(getattr(exc, "failure_observability", {}))
    detail.update({
        "original_exception_type": type(exc).__name__,
        "original_exception": str(exc),
        "sample_id": str(detail.get("sample_id", sample_id)),
        "order_position": int(detail.get("order_position", order_position)),
        "completed_count": len(completed_rows),
        "failure_stage": str(detail.get("failure_stage", failure_stage)),
        "residual_seconds": detail.get("residual_seconds"),
        "residual_limit_seconds": detail.get("residual_limit_seconds"),
        "completed_rows": list(completed_rows),
    })
    validate_failure_record(detail)
    return detail


def validate_failure_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FAILURE_FIELDS if field not in record]
    if missing:
        raise RuntimeError(f"runtime failure record missing fields: {missing}")
    if record["completed_count"] != len(record["completed_rows"]):
        raise RuntimeError("runtime failure completed-count drift")
    if not record["original_exception_type"] or not record["original_exception"]:
        raise RuntimeError("runtime failure original exception missing")
    if not record["sample_id"] or int(record["order_position"]) < 0:
        raise RuntimeError("runtime failure sample/order identity invalid")
    if not record["failure_stage"]:
        raise RuntimeError("runtime failure stage missing")
