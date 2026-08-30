"""Label-free P1i input adaptation for G2 external baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


P1I_FEATURE_NAMES = (
    "k_x",
    "k_y",
    "k_z",
    "q",
    "is_top",
    "is_bottom",
    "is_side",
    "is_interior",
    "top_h",
    "bottom_h",
    "top_T_inf_minus_T_ref",
)
ALLOWED_INPUT_SPLITS = frozenset({"train", "valid_iid"})


@dataclass(frozen=True)
class P1IInputBatch:
    """A batch of raw V6/P1i model inputs without labels.

    The adapter deliberately reads only ``condition`` and split metadata from
    V6 examples.  ``target`` is never touched here.  Truth loading remains an
    explicit EvaluationCore responsibility outside this package.
    """

    sample_ids: tuple[str, ...]
    coords: np.ndarray
    features: np.ndarray
    split: str
    feature_names: tuple[str, ...] = P1I_FEATURE_NAMES
    dataset_id: str = "heat3d_v6_p1i_continuous_physics1024_v1"

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be non-empty and unique")
        if self.split not in ALLOWED_INPUT_SPLITS:
            raise ValueError(
                "G2 P1i input adapter accepts only train/valid_iid; "
                f"got split={self.split!r}"
            )
        coords = np.asarray(self.coords, dtype=np.float32)
        features = np.asarray(self.features, dtype=np.float32)
        expected = (len(sample_ids),)
        if coords.ndim != 3 or coords.shape[:1] != expected or coords.shape[-1] != 3:
            raise ValueError(f"coords must be [B,N,3], got {coords.shape}")
        if features.ndim != 3 or features.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"features must be [B,N,C] aligned with coords, got {features.shape}"
            )
        if features.shape[-1] != len(self.feature_names):
            raise ValueError(
                f"feature width {features.shape[-1]} != schema width {len(self.feature_names)}"
            )
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(features)):
            raise ValueError("P1i inputs must be finite")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def batch_size(self) -> int:
        return int(self.coords.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.coords.shape[1])

    @classmethod
    def from_v6_examples(cls, examples: Sequence[Any]) -> "P1IInputBatch":
        """Adapt V6/P1i examples without accessing target/label attributes."""

        if not examples:
            raise ValueError("at least one V6/P1i example is required")
        sample_ids: list[str] = []
        coords: list[np.ndarray] = []
        features: list[np.ndarray] = []
        splits: list[str] = []
        feature_names: tuple[str, ...] | None = None
        for example in examples:
            sample_id = str(getattr(example, "sample_id", ""))
            condition = getattr(example, "condition", None)
            meta = getattr(example, "meta", {})
            adapter_meta = meta.get("v6_adapter", {}) if isinstance(meta, dict) else {}
            split = str(adapter_meta.get("manifest_split_role", ""))
            if not sample_id or condition is None:
                raise ValueError("V6 example is missing sample_id or condition")
            if split not in ALLOWED_INPUT_SPLITS:
                raise ValueError(f"{sample_id}: forbidden input split {split!r}")
            names = tuple(str(value) for value in condition.condition_feature_names)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("V6/P1i feature schemas differ within one batch")
            sample_ids.append(sample_id)
            coords.append(np.asarray(condition.coords, dtype=np.float32))
            features.append(np.asarray(condition.condition_features, dtype=np.float32))
            splits.append(split)
        if len(set(splits)) != 1:
            raise ValueError("do not mix train and valid_iid in one G2 input batch")
        if feature_names != P1I_FEATURE_NAMES:
            raise ValueError(f"unexpected frozen P1i feature schema: {feature_names}")
        return cls(
            sample_ids=tuple(sample_ids),
            coords=np.stack(coords, axis=0),
            features=np.stack(features, axis=0),
            split=splits[0],
            feature_names=feature_names,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        sample_ids: Sequence[str],
        coords: np.ndarray,
        features: np.ndarray,
        split: str,
        dataset_id: str = "synthetic_g2_native_smoke",
    ) -> "P1IInputBatch":
        """Construct a schema-checked smoke batch.

        This constructor permits a non-1024 point count only for explicitly
        named synthetic/native smoke data.  ``from_v6_examples`` remains the
        path for actual P1i data.
        """

        names = P1I_FEATURE_NAMES
        return cls(
            sample_ids=tuple(sample_ids),
            coords=coords,
            features=features,
            split=split,
            feature_names=names,
            dataset_id=dataset_id,
        )

    def single(self, index: int = 0) -> "P1IInputBatch":
        if not 0 <= int(index) < self.batch_size:
            raise IndexError(index)
        row = int(index)
        return P1IInputBatch(
            sample_ids=(self.sample_ids[row],),
            coords=self.coords[row : row + 1],
            features=self.features[row : row + 1],
            split=self.split,
            feature_names=self.feature_names,
            dataset_id=self.dataset_id,
        )

    def to_torch(self, *, device: str = "cpu") -> tuple[Any, Any]:
        """Materialize tensors explicitly at the adapter boundary."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for G2 model adapters") from exc
        return (
            torch.as_tensor(self.coords, dtype=torch.float32, device=device),
            torch.as_tensor(self.features, dtype=torch.float32, device=device),
        )


def unit_cube_latent_queries(resolution: int = 4) -> np.ndarray:
    """Return GINO's regular latent query grid in ``[0,1]^3``."""

    if int(resolution) < 2:
        raise ValueError("latent resolution must be >= 2")
    axis = np.linspace(0.0, 1.0, int(resolution), dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((x, y, z), axis=-1)[None, ...]
