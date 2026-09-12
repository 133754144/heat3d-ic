"""Atomic, metadata-complete checkpoints for exact training resume.

The helpers in this module are deliberately framework-neutral.  They do not
choose an optimizer, schedule, batch order, or random-number policy; callers
must put those frozen contracts in the metadata payload.  A checkpoint is
written to a sibling temporary file and atomically replaced only after the
pickle has been flushed and closed, so a process observing ``latest.pkl`` sees
either the previous complete epoch or the new complete epoch.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping

from .core import TrainingState, tree


SCHEMA = "g2_exact_resume_checkpoint_v1"
REQUIRED_METADATA = (
    "epoch",
    "global_update_count",
    "best_metric",
    "best_epoch",
    "runner_sha",
    "config_sha",
    "data_sha",
)


def _tree_to_numpy(value: Any) -> Any:
    """Materialize array leaves without changing their tree structure."""

    try:
        import numpy as np

        return tree.tree_map(lambda leaf: np.asarray(leaf), value)
    except Exception:
        # Non-array toy states are still serializable and useful in tests.
        return value


def atomic_latest_checkpoint(
    path: str | Path,
    *,
    state: TrainingState,
    metadata: Mapping[str, Any],
    rng_state: Any,
    batch_state: Any,
    scheduler_state: Any = None,
) -> dict[str, Any]:
    """Write an exact-resume payload and return an auditable receipt.

    ``metadata`` must contain the epoch/update counters and immutable runner,
    config and data hashes.  RNG and batch state are required arguments even
    when a caller uses an explicit deterministic reconstruction token; this
    prevents accidental omission from a formal runner.
    """

    if not isinstance(state, TrainingState):
        raise TypeError("state must be TrainingState")
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"exact-resume metadata missing: {missing}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        **dict(metadata),
        "state": {
            "params": _tree_to_numpy(state.params),
            "optimizer_state": _tree_to_numpy(state.optimizer_state),
            "step": int(state.step),
        },
        "scheduler_state": _tree_to_numpy(scheduler_state),
        "rng_state": rng_state,
        "batch_state": batch_state,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": str(destination),
        "sha256": digest,
        "schema_version": SCHEMA,
        "epoch": int(metadata["epoch"]),
        "global_update_count": int(metadata["global_update_count"]),
        "atomic_replace": True,
    }


def load_latest_checkpoint(
    path: str | Path,
    *,
    expected_runner_sha: str | None = None,
    expected_config_sha: str | None = None,
    expected_data_sha: str | None = None,
) -> dict[str, Any]:
    """Load and validate an exact-resume checkpoint.

    Formal callers should provide all three expected hashes.  A mismatch is a
    hard failure before the state is handed to an optimizer, preventing a
    changed runner/config/dataset from silently continuing a run.
    """

    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported exact-resume checkpoint schema")
    if any(key not in payload for key in ("state", "scheduler_state", "rng_state", "batch_state")):
        raise ValueError("incomplete exact-resume checkpoint")
    expected = {
        "runner_sha": expected_runner_sha,
        "config_sha": expected_config_sha,
        "data_sha": expected_data_sha,
    }
    for key, value in expected.items():
        if value is not None and str(payload.get(key)) != str(value):
            raise ValueError(f"exact-resume checkpoint {key} mismatch")
    state_payload = payload["state"]
    state = TrainingState(
        params=state_payload["params"],
        optimizer_state=state_payload["optimizer_state"],
        step=int(state_payload["step"]),
    )
    return {**payload, "state_object": state}


__all__ = ["SCHEMA", "atomic_latest_checkpoint", "load_latest_checkpoint"]
