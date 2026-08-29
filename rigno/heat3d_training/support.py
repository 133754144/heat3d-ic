"""Label-independent support providers for the V7 G1 ablation contract.

The historical P1i support is a stored, sample-varying route and is consumed
by the Full parent unchanged.  The providers in this module are deliberately
geometry-only alternatives for the registered support-attribution variants.
They do not inspect q/k layout masks, numeric q values, temperature, labels,
solver output, or model error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np


SUPPORT_PROVIDER_SCHEMA_VERSION = "heat3d_v7_g1_support_provider_v3"
LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER = "layout_agnostic_stratified_v1"
CV_ONLY_PROVIDER = "cv_only_v1"
SUPPORTED_ALTERNATIVE_PROVIDERS = (
    LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER,
    CV_ONLY_PROVIDER,
)


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and bytes so order changes are observable."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _stable_seed(*, provider: str, sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{SUPPORT_PROVIDER_SCHEMA_VERSION}:{provider}:{sample_id}:{int(seed)}".encode(
            "utf-8"
        )
    ).hexdigest()
    return int(digest[:16], 16)


def _weighted_choice(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
    weights: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if candidates.size < int(count):
        raise ValueError(f"support candidate shortage: {candidates.size} < {count}")
    if weights.shape != candidates.shape or np.any(~np.isfinite(weights)):
        raise ValueError("support weights must be finite and aligned")
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    probability = None if total <= 0.0 else weights / total
    return np.asarray(
        rng.choice(candidates, size=int(count), replace=False, p=probability),
        dtype=np.int32,
    )


def _validate_geometry(
    coords: np.ndarray,
    control_volume: np.ndarray,
    boundaries: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(coords, dtype=np.float64)
    weights = np.asarray(control_volume, dtype=np.float64).reshape(-1)
    values = np.asarray(boundaries, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("coords must be finite [N,3]")
    if weights.shape != (points.shape[0],) or np.any(~np.isfinite(weights)):
        raise ValueError("control_volume must be finite and aligned with coords")
    if np.any(weights <= 0.0):
        raise ValueError("control_volume must be positive")
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("layer boundaries must contain finite endpoints")
    z = points[:, 2]
    top = np.isclose(z, float(np.max(z)), rtol=0.0, atol=1.0e-15)
    bottom = np.isclose(z, float(np.min(z)), rtol=0.0, atol=1.0e-15)
    interfaces = np.zeros(points.shape[0], dtype=bool)
    for boundary in values[1:-1]:
        interfaces |= np.isclose(z, float(boundary), rtol=0.0, atol=1.0e-15)
    # Strata are disjoint.  Boundary points are not double-counted.
    interfaces &= ~(top | bottom)
    volume = ~(top | bottom | interfaces)
    return points, weights, interfaces, top, bottom, volume


@dataclass(frozen=True)
class SupportSelection:
    """One deterministic, label-independent support selection."""

    provider_id: str
    sample_id: str
    seed: int
    indices: np.ndarray
    strata: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int32).reshape(-1)
        strata = np.asarray(self.strata).reshape(-1)
        if indices.size != 1024 or strata.size != indices.size:
            raise ValueError("V7 support selections must contain exactly 1024 nodes")
        if np.unique(indices).size != indices.size:
            raise ValueError("V7 support indices must be unique")
        if not np.all(np.isfinite(indices)):
            raise ValueError("V7 support indices must be finite")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "strata", strata.astype("U32"))

    @property
    def index_sha256(self) -> str:
        return array_sha256(self.indices)

    @property
    def strata_sha256(self) -> str:
        return array_sha256(self.strata)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SUPPORT_PROVIDER_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "sample_id": self.sample_id,
            "seed": int(self.seed),
            "count": int(self.indices.size),
            "index_sha256": self.index_sha256,
            "strata_sha256": self.strata_sha256,
            "strata_counts": {
                str(name): int(np.sum(self.strata == name))
                for name in sorted(set(self.strata.tolist()))
            },
            **dict(self.metadata),
        }


def select_layout_agnostic_stratified_support(
    *,
    coords: np.ndarray,
    control_volume: np.ndarray,
    boundaries: Sequence[float],
    sample_id: str,
    seed: int,
) -> SupportSelection:
    """Select the preregistered layout-agnostic stratified support.

    The exact quotas are interface=128, top=64, bottom=64, and interior
    volume=768.  The historical block quota is explicitly zero.  Every
    positive quota is sampled by control-volume-weighted choice.  No q/k
    layout mask or source amplitude is an input to this provider.
    """

    _points, weights, interfaces, top, bottom, volume = _validate_geometry(
        coords, control_volume, boundaries
    )
    quotas = {"block": 0, "interface": 128, "top": 64, "bottom": 64, "volume": 768}
    masks = {
        "interface": interfaces,
        "top": top,
        "bottom": bottom,
        "volume": volume,
    }
    rng = np.random.default_rng(
        _stable_seed(
            provider=LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER,
            sample_id=str(sample_id),
            seed=seed,
        )
    )
    selected: list[np.ndarray] = []
    selected_strata: list[np.ndarray] = []
    for name in ("interface", "top", "bottom", "volume"):
        candidates = np.flatnonzero(masks[name])
        choice = _weighted_choice(rng, candidates, quotas[name], weights[candidates])
        selected.append(choice)
        selected_strata.append(np.full(quotas[name], name, dtype="U32"))
    return SupportSelection(
        provider_id=LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER,
        sample_id=str(sample_id),
        seed=int(seed),
        indices=np.concatenate(selected),
        strata=np.concatenate(selected_strata),
        metadata={
            "algorithm": "layout_agnostic_cv_weighted_stratified_choice_v1",
            "selection_inputs": [
                "coords",
                "control_volume",
                "layer_boundaries",
                "sample_id",
                "run_seed",
            ],
            "excluded_inputs": [
                "q",
                "k",
                "q_block_layout_masks",
                "k_block_layout_masks",
                "temperature",
                "labels",
                "solver",
                "model_error",
            ],
            "quotas": quotas,
            "stratum_order": ["interface", "top", "bottom", "volume"],
            "volume_definition": "nodes outside top/bottom/internal-interface strata",
            "layout_agnostic": True,
            "label_independent": True,
        },
    )


def select_cv_only_support(
    *,
    coords: np.ndarray,
    control_volume: np.ndarray,
    boundaries: Sequence[float],
    sample_id: str,
    seed: int,
) -> SupportSelection:
    """Select 1024 control-volume-weighted interior volume nodes only."""

    _points, weights, _interfaces, _top, _bottom, volume = _validate_geometry(
        coords, control_volume, boundaries
    )
    candidates = np.flatnonzero(volume)
    rng = np.random.default_rng(
        _stable_seed(provider=CV_ONLY_PROVIDER, sample_id=str(sample_id), seed=seed)
    )
    indices = _weighted_choice(rng, candidates, 1024, weights[candidates])
    return SupportSelection(
        provider_id=CV_ONLY_PROVIDER,
        sample_id=str(sample_id),
        seed=int(seed),
        indices=indices,
        strata=np.full(1024, "volume", dtype="U32"),
        metadata={
            "algorithm": "interior_volume_cv_weighted_choice_v1",
            "selection_inputs": [
                "coords",
                "control_volume",
                "layer_boundaries",
                "sample_id",
                "run_seed",
            ],
            "excluded_inputs": [
                "q",
                "k",
                "q_block_layout_masks",
                "k_block_layout_masks",
                "temperature",
                "labels",
                "solver",
                "model_error",
            ],
            "quotas": {
                "block": 0,
                "interface": 0,
                "top": 0,
                "bottom": 0,
                "volume": 1024,
            },
            "volume_definition": "nodes outside top/bottom/internal-interface strata",
            "layout_agnostic": True,
            "label_independent": True,
        },
    )


def select_alternative_support(
    provider_id: str,
    *,
    coords: np.ndarray,
    control_volume: np.ndarray,
    boundaries: Sequence[float] | None,
    sample_id: str,
    seed: int,
) -> SupportSelection:
    """Resolve only explicitly registered alternative providers."""

    if boundaries is None:
        raise ValueError(f"{provider_id} requires layer boundaries")
    if provider_id == LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER:
        return select_layout_agnostic_stratified_support(
            coords=coords,
            control_volume=control_volume,
            boundaries=boundaries,
            sample_id=sample_id,
            seed=seed,
        )
    if provider_id == CV_ONLY_PROVIDER:
        return select_cv_only_support(
            coords=coords,
            control_volume=control_volume,
            boundaries=boundaries,
            sample_id=sample_id,
            seed=seed,
        )
    raise ValueError(f"unsupported V7 alternative support provider: {provider_id!r}")


def support_provider_contract() -> dict[str, Any]:
    """Return the machine-readable provider semantics used by the registry."""

    return {
        "schema_version": SUPPORT_PROVIDER_SCHEMA_VERSION,
        "providers": {
            LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER: {
                "canonical_name": "layout-agnostic stratified support",
                "semantic_delta": "fixed strata quotas with CV-weighted geometry-only selection and zero block quota",
                "quotas": {
                    "block": 0,
                    "interface": 128,
                    "top": 64,
                    "bottom": 64,
                    "volume": 768,
                },
                "input_features": [
                    "coords",
                    "control_volume",
                    "layer_boundaries",
                    "sample_id",
                    "run_seed",
                ],
                "forbidden_inputs": [
                    "q",
                    "k",
                    "q_block_layout_masks",
                    "k_block_layout_masks",
                    "temperature",
                    "labels",
                    "solver",
                    "model_error",
                ],
                "label_independent": True,
            },
            CV_ONLY_PROVIDER: {
                "canonical_name": "CV-only support",
                "semantic_delta": "1024 CV-weighted interior volume nodes with no boundary/interface/block quota",
                "quotas": {
                    "block": 0,
                    "interface": 0,
                    "top": 0,
                    "bottom": 0,
                    "volume": 1024,
                },
                "input_features": [
                    "coords",
                    "control_volume",
                    "layer_boundaries",
                    "sample_id",
                    "run_seed",
                ],
                "forbidden_inputs": [
                    "q",
                    "k",
                    "q_block_layout_masks",
                    "k_block_layout_masks",
                    "temperature",
                    "labels",
                    "solver",
                    "model_error",
                ],
                "label_independent": True,
            },
        },
        "determinism": "sha256(schema_version,provider,sample_id,run_seed) -> numpy Generator; no replacement",
        "training_support_default": "historical_v6_stored_support",
        "publication_evidence": False,
    }


__all__ = [
    "CV_ONLY_PROVIDER",
    "LAYOUT_AGNOSTIC_STRATIFIED_PROVIDER",
    "SUPPORTED_ALTERNATIVE_PROVIDERS",
    "SUPPORT_PROVIDER_SCHEMA_VERSION",
    "SupportSelection",
    "array_sha256",
    "select_alternative_support",
    "select_cv_only_support",
    "select_layout_agnostic_stratified_support",
    "support_provider_contract",
]
