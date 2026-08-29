"""Full-domain P1i adapters for the registered support ablations.

This module reads only explicitly selected train/valid_iid rows from the
frozen full-field archive. The shared mesh is label-independent. It is used
to construct the two support ablations and to retain a common 240825-node
evaluation domain; it is not a solver or a data generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rigno.heat3d_v1_native_supervised import (
    V1SteadyConditionInput,
    V1SteadyTarget,
)
from rigno.heat3d_v6_dataset import (
    V6_DUAL_ROBIN_CONDITION_FEATURES,
    V6DualRobinExample,
)
from rigno.heat3d_v6_global_context import global_context_from_v6_inputs

from .support import SupportSelection, select_alternative_support


@dataclass(frozen=True)
class FullFieldGeometry:
    coords: np.ndarray
    control_volume: np.ndarray
    layer_id: np.ndarray
    boundary_flags: np.ndarray
    layer_boundaries: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float64)
        cv = np.asarray(self.control_volume, dtype=np.float64).reshape(-1)
        layer_id = np.asarray(self.layer_id, dtype=np.int32).reshape(-1)
        flags = np.asarray(self.boundary_flags, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("full-field coords must be [N,3]")
        if cv.shape != (len(coords),) or np.any(~np.isfinite(cv)) or np.any(cv <= 0.0):
            raise ValueError("full-field control volumes must be positive [N]")
        if layer_id.shape != (len(coords),):
            raise ValueError("full-field layer_id must align with coords")
        if flags.shape != (len(coords), 4):
            raise ValueError("full-field boundary_flags must be [N,4]")
        boundaries = np.asarray(self.layer_boundaries, dtype=np.float64).reshape(-1)
        if boundaries.size < 2 or not np.all(np.isfinite(boundaries)):
            raise ValueError("full-field layer boundaries are invalid")
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "control_volume", cv)
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "boundary_flags", flags)
        object.__setattr__(self, "layer_boundaries", boundaries)


@dataclass(frozen=True)
class FullFieldP1IData:
    train_examples: tuple[V6DualRobinExample, ...]
    valid_examples: tuple[V6DualRobinExample, ...]
    full_truth_delta_by_id: dict[str, np.ndarray]
    full_q_by_id: dict[str, np.ndarray]
    context_by_id: dict[str, dict[str, float]]
    support_selection_by_id: dict[str, SupportSelection]
    geometry: FullFieldGeometry


def load_full_field_geometry(archive_path: str | Path) -> FullFieldGeometry:
    """Read the frozen shared mesh only; no sample labels are touched."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by CI contract
        raise RuntimeError("full-field support providers require h5py") from exc
    with h5py.File(Path(archive_path).resolve(), "r") as handle:
        shared = handle["shared"]
        coords = np.asarray(shared["coords_m"], dtype=np.float64)
        cv = np.asarray(shared["control_volume_m3"], dtype=np.float64)
        layer_id = np.asarray(shared["layer_id"], dtype=np.int32)
        flags = np.asarray(shared["boundary_flags"], dtype=np.float64)
    return FullFieldGeometry(
        coords=coords,
        control_volume=cv,
        layer_id=layer_id,
        boundary_flags=flags,
        layer_boundaries=np.asarray(
            [float(np.min(coords[:, 2])), float(np.max(coords[:, 2]))]
        ),
    )


def _layer_boundaries_from_meta(
    meta: Mapping[str, Any], geometry: FullFieldGeometry
) -> np.ndarray:
    physics = meta.get("physics") or {}
    layers = physics.get("layers_bottom_to_top") or meta.get("layers_bottom_to_top")
    if not layers:
        raise ValueError("P1i full-field metadata has no layer stack")
    z0 = float(np.min(geometry.coords[:, 2]))
    boundaries = z0 + np.concatenate(
        [
            np.asarray([0.0]),
            np.cumsum([float(row["thickness_m"]) for row in layers]),
        ]
    )
    if boundaries.size != len(layers) + 1:
        raise ValueError("P1i layer-boundary count drifted")
    if not np.allclose(
        boundaries[-1],
        np.max(geometry.coords[:, 2]),
        atol=1.0e-15,
        rtol=0.0,
    ):
        raise ValueError("P1i layer stack does not cover the full field")
    return boundaries


def _fraction_mask(
    coords: np.ndarray,
    bbox: Sequence[float],
    footprint: Sequence[float],
    layer_mask: np.ndarray,
) -> np.ndarray:
    if len(bbox) != 4:
        raise ValueError("P1i block bbox must contain x0,x1,y0,y1")
    x0, x1, y0, y1 = (float(value) for value in bbox)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("P1i block bbox is outside the frozen unit footprint")
    lx, ly = (float(value) for value in footprint[:2])
    x = coords[:, 0] / lx
    y = coords[:, 1] / ly
    return layer_mask & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def materialize_full_input_fields(
    *,
    meta: Mapping[str, Any],
    geometry: FullFieldGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize frozen k/q/feature flags from input metadata and mesh."""

    physics = meta.get("physics") or {}
    layers = physics.get("layers_bottom_to_top") or meta.get("layers_bottom_to_top")
    if not layers:
        raise ValueError("P1i metadata has no layer stack")
    footprint = physics.get("footprint_m") or [0.01, 0.01]
    layer_names = [str(row.get("id")) for row in layers]
    if np.any(geometry.layer_id < 0) or np.any(geometry.layer_id >= len(layers)):
        raise ValueError("full-field layer IDs are outside the metadata stack")
    background = np.asarray(
        [row["background_k_xyz_W_mK"] for row in layers], dtype=np.float64
    )
    k_field = background[geometry.layer_id].copy()
    k_values = [float(value) for value in meta.get("k_block_values_W_mK", ())]
    k_blocks = meta.get("k_blocks") or []
    if len(k_values) != len(k_blocks):
        raise ValueError("P1i k block/value length mismatch")
    for block, value in zip(k_blocks, k_values, strict=True):
        try:
            layer_index = layer_names.index(str(block["layer"]))
        except ValueError as exc:
            raise ValueError(
                f"unknown P1i k block layer: {block.get('layer')!r}"
            ) from exc
        mask = _fraction_mask(
            geometry.coords,
            block["bbox_fraction_xy"],
            footprint,
            geometry.layer_id == layer_index,
        )
        k_field[mask, :] = value
    if np.any(~np.isfinite(k_field)) or np.any(k_field <= 0.0):
        raise ValueError("materialized P1i k field is not positive finite")

    q_field = np.zeros(len(geometry.coords), dtype=np.float64)
    q_blocks = meta.get("q_blocks") or []
    fractions = [float(value) for value in meta.get("q_block_power_fractions", ())]
    if len(fractions) != len(q_blocks):
        raise ValueError("P1i q block/fraction length mismatch")
    package_power = float(meta["package_total_power_W"])
    for block, fraction in zip(q_blocks, fractions, strict=True):
        try:
            layer_index = layer_names.index(str(block["layer"]))
        except ValueError as exc:
            raise ValueError(
                f"unknown P1i q block layer: {block.get('layer')!r}"
            ) from exc
        mask = _fraction_mask(
            geometry.coords,
            block["bbox_fraction_xy"],
            footprint,
            geometry.layer_id == layer_index,
        )
        if not np.any(mask) or np.any(q_field[mask] != 0.0):
            raise ValueError("P1i q blocks overlap or have no full-field nodes")
        volume = float(np.sum(geometry.control_volume[mask]))
        q_field[mask] = package_power * fraction / volume
    if not np.isclose(
        np.sum(q_field * geometry.control_volume),
        package_power,
        rtol=2.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("materialized P1i q field violates frozen power conservation")
    return k_field, q_field, geometry.boundary_flags.copy()


def _example_from_full_fields(
    *,
    sample_id: str,
    split_role: str,
    meta: Mapping[str, Any],
    geometry: FullFieldGeometry,
    full_delta: np.ndarray,
    selection: SupportSelection,
    k_field: np.ndarray,
    q_field: np.ndarray,
    flags: np.ndarray,
    ) -> V6DualRobinExample:
    physics = meta.get("physics") or {}
    bottom_tinf = float(physics["ambient_K"])
    top_h = float(meta["top_h_W_m2K"])
    bottom_h = float(meta["bottom_h_W_m2K"])
    support = selection.indices
    support_coords = geometry.coords[support]
    support_features = np.column_stack(
        (
            k_field[support],
            q_field[support, None],
            flags[support],
            np.full((len(support), 1), top_h),
            np.full((len(support), 1), bottom_h),
            np.zeros((len(support), 1), dtype=np.float64),
        )
    )
    enriched_meta = dict(meta)
    enriched_meta["split_role"] = str(split_role)
    enriched_meta["v7_support_provider"] = selection.manifest()
    enriched_meta["v7_full_field_geometry"] = {
        "node_count": int(len(geometry.coords)),
        "layer_id_source": "frozen_full_field_shared_geometry",
        "label_independent": True,
    }
    enriched_meta["v6_adapter"] = {
        "dataset_id": str(meta["dataset_id"]),
        "manifest_split_role": str(split_role),
        "group_id": str(meta.get("group_id") or sample_id),
        "reference_temperature_K": bottom_tinf,
        "top_T_inf_K": bottom_tinf,
        "bottom_T_inf_K": bottom_tinf,
        "bottom_boundary_semantics": "robin_not_dirichlet",
        "operator_point_measure": "control_volume_frozen_v7_support_projection",
    }
    support_delta = np.asarray(full_delta, dtype=np.float64).reshape(-1)[support]
    return V6DualRobinExample(
        sample_id=sample_id,
        condition=V1SteadyConditionInput(
            coords=support_coords,
            condition_features=support_features,
            condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
            k_encoding_mode="diag3",
        ),
        target=V1SteadyTarget(
            target_u=support_delta + bottom_tinf,
            target_name="temperature",
            target_role="frozen full-field deltaT projected to registered support",
        ),
        meta=enriched_meta,
        operator_point_weights=geometry.control_volume[support],
    )


def _full_field_context(
    *,
    meta: Mapping[str, Any],
    geometry: FullFieldGeometry,
    k_field: np.ndarray,
    q_field: np.ndarray,
    flags: np.ndarray,
) -> dict[str, float]:
    """Build the frozen global context from the common full-field inputs.

    Alternative support providers may intentionally contain no source nodes
    (the CV-only provider).  The V6 global context is a global physical
    input, so its source projection must use the shared 240825-node input
    field rather than the selected support subset.  This does not read the
    full-field target.
    """

    physics = meta.get("physics") or {}
    ambient = float(physics["ambient_K"])
    top_h = float(meta["top_h_W_m2K"])
    bottom_h = float(meta["bottom_h_W_m2K"])
    condition = np.column_stack(
        (
            k_field,
            q_field[:, None],
            flags,
            np.full((len(k_field), 1), top_h),
            np.full((len(k_field), 1), bottom_h),
            np.zeros((len(k_field), 1), dtype=np.float64),
        )
    )
    layers = physics.get("layers_bottom_to_top") or meta.get("layers_bottom_to_top")
    footprint = physics.get("footprint_m") or [0.01, 0.01]
    return global_context_from_v6_inputs(
        coords=geometry.coords,
        raw_condition=condition,
        condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
        reference_temperature_K=ambient,
        top_T_inf_K=ambient,
        bottom_T_inf_K=ambient,
        operator_point_weights=geometry.control_volume,
        package_total_power_W=float(meta["package_total_power_W"]),
        package_extents_m=(
            float(footprint[0]),
            float(footprint[1]),
            float(sum(float(row["thickness_m"]) for row in layers)),
        ),
    )


def load_alternative_p1i_examples(
    *,
    subset: str | Path,
    manifest_path: str | Path,
    full_field_archive_path: str | Path,
    provider_id: str,
    seed: int,
) -> FullFieldP1IData:
    """Prepare one registered support provider from train/valid rows only."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("samples")
    if not isinstance(rows, list) or len(rows) != 1024:
        raise ValueError("P1i manifest must contain 1024 rows")
    selected_rows = {
        "train": [row for row in rows if str(row.get("split_role")) == "train"],
        "valid_iid": [
            row for row in rows if str(row.get("split_role")) == "valid_iid"
        ],
    }
    if {
        name: len(value) for name, value in selected_rows.items()
    } != {"train": 768, "valid_iid": 128}:
        raise ValueError("P1i train/valid population drifted")
    root = Path(subset).resolve()
    sample_root = root / "samples" if (root / "samples").is_dir() else root
    geometry = load_full_field_geometry(full_field_archive_path)
    first_relative = Path(
        str(
            selected_rows["train"][0].get("sample_dir")
            or selected_rows["train"][0].get("relative_path")
        )
    )
    if first_relative.parts[:1] == ("samples",):
        first_relative = Path(*first_relative.parts[1:])
    first_meta = json.loads(
        (sample_root / first_relative / "sample_meta.json").read_text(encoding="utf-8")
    )
    geometry = FullFieldGeometry(
        coords=geometry.coords,
        control_volume=geometry.control_volume,
        layer_id=geometry.layer_id,
        boundary_flags=geometry.boundary_flags,
        layer_boundaries=_layer_boundaries_from_meta(first_meta, geometry),
    )
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("full-field support providers require h5py") from exc
    with h5py.File(Path(full_field_archive_path).resolve(), "r") as handle:
        ids = np.asarray(handle["samples"]["sample_id"]).astype(str)
        id_to_row = {value: index for index, value in enumerate(ids.tolist())}
        full_delta_dataset = handle["samples"]["deltaT_K"]
        train_examples: list[V6DualRobinExample] = []
        valid_examples: list[V6DualRobinExample] = []
        full_truth: dict[str, np.ndarray] = {}
        full_q: dict[str, np.ndarray] = {}
        context_by_id: dict[str, dict[str, float]] = {}
        selections: dict[str, SupportSelection] = {}
        for role, role_rows in selected_rows.items():
            destination = train_examples if role == "train" else valid_examples
            for row in role_rows:
                sample_id = str(row["sample_id"])
                if sample_id not in id_to_row:
                    raise ValueError(
                        f"full-field archive is missing selected sample {sample_id}"
                    )
                relative = Path(
                    str(row.get("sample_dir") or row.get("relative_path") or sample_id)
                )
                if relative.parts[:1] == ("samples",):
                    relative = Path(*relative.parts[1:])
                meta = json.loads(
                    (sample_root / relative / "sample_meta.json").read_text(
                        encoding="utf-8"
                    )
                )
                boundaries = _layer_boundaries_from_meta(meta, geometry)
                selection = select_alternative_support(
                    provider_id,
                    coords=geometry.coords,
                    control_volume=geometry.control_volume,
                    boundaries=boundaries,
                    sample_id=sample_id,
                    seed=seed,
                )
                k_field, q_field, flags = materialize_full_input_fields(
                    meta=meta, geometry=geometry
                )
                context_by_id[sample_id] = _full_field_context(
                    meta=meta,
                    geometry=geometry,
                    k_field=k_field,
                    q_field=q_field,
                    flags=flags,
                )
                delta = np.asarray(
                    full_delta_dataset[id_to_row[sample_id]], dtype=np.float64
                )
                if delta.shape != (len(geometry.coords),) or not np.all(
                    np.isfinite(delta)
                ):
                    raise ValueError(f"{sample_id}: invalid selected full-field deltaT row")
                destination.append(
                    _example_from_full_fields(
                        sample_id=sample_id,
                        split_role=role,
                        meta=meta,
                        geometry=geometry,
                        full_delta=delta,
                        selection=selection,
                        k_field=k_field,
                        q_field=q_field,
                        flags=flags,
                    )
                )
                # Full-domain truth/q are needed only for common-domain
                # validation.  Training keeps only the projected support
                # target, so do not retain 768 redundant 240825-node rows.
                if role == "valid_iid":
                    full_truth[sample_id] = delta
                    full_q[sample_id] = q_field
                selections[sample_id] = selection
    return FullFieldP1IData(
        train_examples=tuple(train_examples),
        valid_examples=tuple(valid_examples),
        full_truth_delta_by_id=full_truth,
        full_q_by_id=full_q,
        context_by_id=context_by_id,
        support_selection_by_id=selections,
        geometry=geometry,
    )


__all__ = [
    "FullFieldGeometry",
    "FullFieldP1IData",
    "load_alternative_p1i_examples",
    "load_full_field_geometry",
    "materialize_full_input_fields",
]
