"""Geometry-independent, label-free V8 support selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .schema import SupportProvenance


SUPPORT_CLASSES = (
    "heat_source",
    "material_interface",
    "thermal_boundary_sink",
    "bulk_volume",
)


@dataclass(frozen=True)
class V8SupportSelection:
    indices: np.ndarray
    classes: np.ndarray
    class_counts: dict[str, int]
    candidate_counts: dict[str, int]
    coverage_fraction: dict[str, float]
    seed: int
    provenance: SupportProvenance
    selection_contract: str


def _seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"v8-support:{sample_id}:{seed}".encode()).hexdigest()
    return int(digest[:16], 16)


def select_v8_support(
    *,
    source_mask: np.ndarray,
    interface_mask: np.ndarray,
    boundary_sink_mask: np.ndarray,
    control_volume_m3: np.ndarray,
    sample_id: str,
    support_provenance: SupportProvenance,
    count: int = 1024,
    seed: int = 0,
) -> V8SupportSelection:
    """Select four physics strata using masks and geometry only.

    The selection algorithm deliberately has no temperature, gradient,
    hotspot, final refinement, power amplitude, conductivity amplitude or
    model-error input.  This does not make an ORACLE-derived candidate mask
    deployable; callers must state support provenance explicitly.
    Masks are made disjoint with boundary > interface > source priority, then
    the remainder is bulk.  Equal initial quotas are redistributed when a
    stratum is smaller than its quota.
    """

    volume = np.asarray(control_volume_m3, dtype=np.float64).reshape(-1)
    provenance = SupportProvenance(support_provenance)
    n = volume.size
    if count < 1 or count > n:
        raise ValueError("support count must be between 1 and cell count")
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("control volumes must be finite and positive")
    source = np.asarray(source_mask, dtype=bool).reshape(-1)
    interface = np.asarray(interface_mask, dtype=bool).reshape(-1)
    boundary = np.asarray(boundary_sink_mask, dtype=bool).reshape(-1)
    if source.size != n or interface.size != n or boundary.size != n:
        raise ValueError("support masks must align with control volumes")
    interface = interface & ~boundary
    source = source & ~boundary & ~interface
    bulk = ~(boundary | interface | source)
    masks = {
        "heat_source": source,
        "material_interface": interface,
        "thermal_boundary_sink": boundary,
        "bulk_volume": bulk,
    }
    candidate_counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    base = count // len(SUPPORT_CLASSES)
    quotas = {name: min(base, candidate_counts[name]) for name in SUPPORT_CLASSES}
    remaining = count - sum(quotas.values())
    # Deterministic redistribution to the strata with the largest unused pool.
    while remaining:
        available = {
            name: candidate_counts[name] - quotas[name] for name in SUPPORT_CLASSES
        }
        name = max(SUPPORT_CLASSES, key=lambda item: (available[item], -SUPPORT_CLASSES.index(item)))
        take = min(remaining, available[name])
        if take <= 0:
            raise ValueError("support candidate shortage after redistribution")
        quotas[name] += take
        remaining -= take

    rng = np.random.default_rng(_seed(sample_id, seed))
    selected: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for name in SUPPORT_CLASSES:
        candidates = np.flatnonzero(masks[name])
        quota = quotas[name]
        if quota == 0:
            continue
        weights = volume[candidates]
        probability = weights / np.sum(weights)
        choice = np.asarray(
            rng.choice(candidates, size=quota, replace=False, p=probability),
            dtype=np.int64,
        )
        selected.append(choice)
        classes.append(np.full(quota, name, dtype="U32"))
    indices = np.concatenate(selected)
    class_values = np.concatenate(classes)
    if len(indices) != count or len(np.unique(indices)) != count:
        raise ValueError("V8 support selection is not unique and complete")
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    class_values = class_values[order]
    class_counts = {name: int(np.count_nonzero(class_values == name)) for name in SUPPORT_CLASSES}
    coverage = {
        name: (
            class_counts[name] / candidate_counts[name]
            if candidate_counts[name]
            else 1.0
        )
        for name in SUPPORT_CLASSES
    }
    return V8SupportSelection(
        indices=indices,
        classes=class_values,
        class_counts=class_counts,
        candidate_counts=candidate_counts,
        coverage_fraction=coverage,
        seed=int(seed),
        provenance=provenance,
        selection_contract=(
            f"{provenance.value}; mask_and_control_volume_only; excludes direct "
            "T, gradT, hotspot, final refinement, field amplitudes and model error"
        ),
    )
