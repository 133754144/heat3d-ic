"""Atomic PyTorch checkpoint helpers for the external G2 runners."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


TORCH_SCHEMA = "g2_external_torch_exact_resume_v1"


def atomic_torch_latest_checkpoint(
    path: str | Path,
    *,
    state: Mapping[str, Any],
    epoch: int,
    global_update_count: int,
    best_metric: float,
    best_epoch: int | None,
    runner_sha: str,
    config_sha: str,
    data_sha: str,
    batch_order_state: Mapping[str, Any],
    test_access: bool = False,
) -> dict[str, Any]:
    """Atomically save a complete external-runner epoch state.

    The state is intentionally supplied by the caller so the upstream model,
    optimizer, scheduler, and RNG representations remain unchanged.  The
    helper only adds the immutable execution metadata and uses ``os.replace``
    for the latest pointer.
    """

    if test_access:
        raise ValueError("exact-resume checkpoint cannot carry test access")
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TORCH_SCHEMA,
        **dict(state),
        "epoch": int(epoch),
        "global_update_count": int(global_update_count),
        "best_metric": float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "runner_sha": str(runner_sha),
        "config_sha": str(config_sha),
        "data_sha": str(data_sha),
        "batch_order_state": dict(batch_order_state),
        "test_access": False,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return {
        "schema_version": TORCH_SCHEMA,
        "path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "epoch": int(epoch),
        "global_update_count": int(global_update_count),
        "atomic_replace": True,
    }


def load_torch_latest_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on an incomplete external checkpoint."""

    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TORCH_SCHEMA:
        raise ValueError("external exact-resume checkpoint schema mismatch")
    if payload.get("test_access") is not False:
        raise ValueError("external checkpoint has test access")
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "global_update_count",
        "runner_sha",
        "config_sha",
        "data_sha",
        "batch_order_state",
    ):
        if key not in payload:
            raise ValueError(f"external checkpoint missing {key}")
    return payload


__all__ = ["TORCH_SCHEMA", "atomic_torch_latest_checkpoint", "load_torch_latest_checkpoint"]
