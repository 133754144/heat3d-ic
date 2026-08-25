"""V6 P1g/P1h dual-Robin dataset adapter.

The adapter deliberately does not reuse the V1 bottom-Dirichlet metadata
loader.  P1g has Robin conditions on both package faces; treating the bottom
ambient as a fixed surface temperature would change the physical problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v1_native_supervised import (
    V1RelativeBCFeatureView,
    V1SteadyConditionInput,
    V1SteadyTarget,
    V1TemperatureRiseLegacyBridge,
)
from rigno.models.operator import Inputs


P1G_GEOMETRY_ADAPTIVE_V6_DATASET_ID = (
    "heat3d_v6_p1g_geometry_deconfounded1024_v0"
)
SHARED_SUPPORT_V6_DATASET_ID = "heat3d_v6_p1h_shared_support1024_v0"
CONTINUOUS_PHYSICS_V6_DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
CANONICAL_V6_DATASET_ID = SHARED_SUPPORT_V6_DATASET_ID
SUPPORTED_V6_DATASET_IDS = {
    P1G_GEOMETRY_ADAPTIVE_V6_DATASET_ID,
    CANONICAL_V6_DATASET_ID,
    CONTINUOUS_PHYSICS_V6_DATASET_ID,
}
V6_DUAL_ROBIN_CONDITION_FEATURES = (
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
EXPECTED_SPLIT_COUNTS = {"train": 768, "valid_iid": 128, "test_iid": 128}


@dataclass(frozen=True)
class V6DualRobinExample:
    sample_id: str
    condition: V1SteadyConditionInput
    target: V1SteadyTarget
    meta: dict[str, Any]
    operator_point_weights: np.ndarray

    def get_relative_bc_feature_view(self) -> V1RelativeBCFeatureView:
        return V1RelativeBCFeatureView(
            condition_features=self.condition.condition_features,
            condition_feature_names=self.condition.condition_feature_names,
            t_ref_value=float(self.meta["v6_adapter"]["reference_temperature_K"]),
            t_ref_source="bottom_robin_ambient_temperature_K",
            view_name="v6_dual_robin_relative_bc_condition_features",
            view_role="canonical V6 inference feature view",
        )

    def build_temperature_rise_legacy_inputs_from_relative_features(
        self, bridge_policy: str = "zero_delta_u_bridge"
    ) -> V1TemperatureRiseLegacyBridge:
        if bridge_policy not in {"tref_u_bridge", "zero_delta_u_bridge"}:
            raise ValueError(f"unsupported bridge_policy={bridge_policy!r}")
        view = self.get_relative_bc_feature_view()
        n_points = self.condition.coords.shape[0]
        t_ref = jnp.full((1, 1, n_points, 1), view.t_ref_value, dtype=jnp.float32)
        target_temperature = jnp.asarray(
            self.target.target_u.reshape(1, 1, n_points, 1), dtype=jnp.float32
        )
        target_delta = target_temperature - t_ref
        legacy_u = t_ref if bridge_policy == "tref_u_bridge" else jnp.zeros_like(t_ref)
        c = jnp.asarray(
            view.condition_features.reshape(1, 1, n_points, -1), dtype=jnp.float32
        )
        coords = jnp.asarray(
            self.condition.coords.reshape(1, 1, n_points, 3), dtype=jnp.float32
        )
        return V1TemperatureRiseLegacyBridge(
            legacy_inputs=Inputs(
                u=legacy_u,
                c=c,
                x_inp=coords,
                x_out=coords,
                t=None,
                tau=None,
            ),
            target_delta_u=target_delta,
            t_ref=t_ref,
            t_ref_value=view.t_ref_value,
            t_ref_source=view.t_ref_source,
            condition_feature_names=view.condition_feature_names,
            bridge_policy=bridge_policy,
            bridge_name="v6_dual_robin_temperature_rise_bridge",
            bridge_role="runtime adapter; bottom Robin remains Robin",
        )

    def v6_operator_point_weights(self) -> np.ndarray:
        """Return the dataset-declared label-independent operator measure."""

        weights = np.asarray(self.operator_point_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (self.condition.coords.shape[0],) or np.any(weights <= 0.0):
            raise ValueError("V6 operator weights must be positive [N]")
        return weights / np.sum(weights)

    def v6_global_context_inputs(self) -> dict[str, Any]:
        view = self.get_relative_bc_feature_view()
        adapter = self.meta["v6_adapter"]
        sources = self.meta.get("sources") or []
        if sources:
            total_power = float(sum(float(row["source_power_W"]) for row in sources))
        else:
            total_power = float(self.meta["package_total_power_W"])
        physics = self.meta.get("physics") or {}
        layers = self.meta.get("layers_bottom_to_top") or physics.get(
            "layers_bottom_to_top"
        )
        if not layers:
            raise ValueError(f"{self.sample_id}: missing V6 layer stack metadata")
        total_thickness = float(
            sum(float(row["thickness_m"]) for row in layers)
        )
        footprint = physics.get("footprint_m", (0.01, 0.01))
        return {
            "coords": self.condition.coords,
            "raw_condition": view.condition_features,
            "condition_feature_names": view.condition_feature_names,
            "reference_temperature_K": view.t_ref_value,
            "top_T_inf_K": float(adapter["top_T_inf_K"]),
            "bottom_T_inf_K": float(adapter["bottom_T_inf_K"]),
            "operator_point_weights": self.v6_operator_point_weights(),
            "package_total_power_W": total_power,
            "package_extents_m": (
                float(footprint[0]),
                float(footprint[1]),
                total_thickness,
            ),
        }


class Heat3DV6DualRobinDataset:
    """Read P1g/P1h samples under the immutable manifest split contract."""

    def __init__(
        self,
        datadir: str | Path,
        manifest_path: str | Path,
        *,
        include_roles: set[str] | None = None,
    ) -> None:
        self.datadir = Path(datadir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()
        self.split_ids = self._split_ids_from_manifest()
        allowed = None if include_roles is None else set(include_roles)
        if allowed is not None and not allowed <= {"train", "valid_iid", "test_iid"}:
            raise ValueError(f"unsupported V6 materialized roles: {sorted(allowed)}")
        rows = self.manifest["samples"]
        self.materialized_roles = (
            {self._canonical_split_role(row) for row in rows}
            if allowed is None
            else allowed
        )
        self.samples = [
            self._load_sample(row)
            for row in rows
            if allowed is None or self._canonical_split_role(row) in allowed
        ]

    def _validate_manifest(self) -> None:
        if self.manifest.get("dataset_id") not in SUPPORTED_V6_DATASET_IDS:
            raise ValueError(
                "V6 training loader accepts only explicitly registered frozen datasets; "
                f"found {self.manifest.get('dataset_id')!r}"
            )
        rows = self.manifest.get("samples")
        if not isinstance(rows, list) or len(rows) != 1024:
            raise ValueError("canonical V6 manifest must contain exactly 1024 samples")
        ids = [str(row.get("sample_id") or "") for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("canonical V6 manifest sample IDs must be nonempty and unique")

    def _split_ids_from_manifest(self) -> dict[str, list[str]]:
        splits = {name: [] for name in EXPECTED_SPLIT_COUNTS}
        group_roles: dict[str, str] = {}
        for row in self.manifest["samples"]:
            role = self._canonical_split_role(row)
            group_id = str(row.get("group_id") or row.get("sample_id") or "")
            if not group_id:
                raise ValueError(f"{row.get('sample_id')}: missing group_id")
            previous = group_roles.setdefault(group_id, role)
            if previous != role:
                raise ValueError(f"V6 group leakage: {group_id} spans {previous}/{role}")
            splits[role].append(str(row["sample_id"]))
        counts = {name: len(values) for name, values in splits.items()}
        if counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"V6 manifest split counts drifted: {counts}")
        return splits

    def _canonical_split_role(self, row: dict[str, Any]) -> str:
        raw_role = str(row.get("split_role"))
        if self.manifest.get("dataset_id") == CONTINUOUS_PHYSICS_V6_DATASET_ID:
            role_map = {
                "train": "train",
                "valid_iid": "valid_iid",
                "test_iid": "test_iid",
            }
        else:
            role_map = {"train": "train", "valid": "valid_iid", "test": "test_iid"}
        role = role_map.get(raw_role)
        if role is None:
            raise ValueError(f"unsupported V6 manifest split_role={raw_role!r}")
        return role

    def _load_sample(self, row: dict[str, Any]) -> V6DualRobinExample:
        sample_id = str(row["sample_id"])
        relative = Path(str(row.get("sample_dir") or row.get("relative_path") or sample_id))
        if self.datadir.name == "samples" and relative.parts[:1] == ("samples",):
            relative = Path(*relative.parts[1:])
        sample_dir = self.datadir / relative
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        dataset_id = str(self.manifest["dataset_id"])
        if meta.get("dataset_id") != dataset_id:
            raise ValueError(f"{sample_id}: sample/manifest dataset_id mismatch")
        physics = meta.get("physics") or {}
        bc = meta.get("boundary_conditions") or physics.get("boundary_conditions") or {}
        top = bc.get("top") or {}
        bottom = bc.get("bottom") or {}
        if top.get("type") != "robin" or bottom.get("type") != "robin":
            raise ValueError(f"{sample_id}: both top and bottom must be Robin")

        coords = _load_matrix(sample_dir / "coords.npy", 3)
        k_field = _load_matrix(sample_dir / "k_field.npy", 3)
        q_field = _load_matrix(sample_dir / "q_field.npy", 1)
        stored_bc = np.asarray(np.load(sample_dir / "bc_features.npy"), dtype=np.float64)
        if stored_bc.ndim != 2 or stored_bc.shape[1] not in {4, 7}:
            raise ValueError(f"{sample_id}: expected BC features [N,4] or [N,7]")
        flags = stored_bc[:, :4]
        temperature = _load_column(sample_dir / "temperature.npy")
        count = coords.shape[0]
        if count != 1024 or any(
            array.shape[0] != count for array in (k_field, q_field, flags, temperature)
        ):
            raise ValueError(f"{sample_id}: canonical operator shape must be [1024,*]")
        if not np.allclose(np.sum(flags, axis=1), 1.0, atol=0.0, rtol=0.0):
            raise ValueError(f"{sample_id}: four BC flags must be one-hot")

        if dataset_id == CONTINUOUS_PHYSICS_V6_DATASET_ID:
            top_h = float(meta["top_h_W_m2K"])
            bottom_h = float(meta["bottom_h_W_m2K"])
            top_tinf = bottom_tinf = float(physics["ambient_K"])
        else:
            top_h = float(top["h_W_m2K"])
            bottom_h = float(bottom["h_W_m2K"])
            top_tinf = float(top["T_inf_K"])
            bottom_tinf = float(bottom["T_inf_K"])
        if min(top_h, bottom_h) <= 0.0:
            raise ValueError(f"{sample_id}: Robin h values must be positive")
        top_offset = top_tinf - bottom_tinf
        if stored_bc.shape[1] == 7:
            expected_bc = np.column_stack(
                (
                    flags,
                    np.full(count, top_h),
                    np.full(count, bottom_h),
                    np.full(count, top_offset),
                )
            )
            if not np.allclose(stored_bc, expected_bc, rtol=0.0, atol=1.0e-10):
                raise ValueError(f"{sample_id}: stored P1i BC broadcasts drifted")
        broadcast = np.column_stack(
            (
                np.full(count, top_h),
                np.full(count, bottom_h),
                np.full(count, top_offset),
            )
        )
        features = np.concatenate((k_field, q_field, flags, broadcast), axis=1)
        if features.shape != (1024, len(V6_DUAL_ROBIN_CONDITION_FEATURES)):
            raise AssertionError("V6 condition width invariant failed")
        enriched_meta = dict(meta)
        enriched_meta["v6_adapter"] = {
            "dataset_id": dataset_id,
            "manifest_split_role": self._canonical_split_role(row),
            "group_id": str(row.get("group_id") or sample_id),
            "reference_temperature_K": bottom_tinf,
            "top_T_inf_K": top_tinf,
            "bottom_T_inf_K": bottom_tinf,
            "bottom_boundary_semantics": "robin_not_dirichlet",
            "operator_point_measure": (
                "control_volume_frozen_irregular_1024"
                if dataset_id == CONTINUOUS_PHYSICS_V6_DATASET_ID
                else "equal_weight_frozen_irregular_1024"
            ),
        }
        if dataset_id == CONTINUOUS_PHYSICS_V6_DATASET_ID:
            operator_point_weights = np.asarray(
                np.load(sample_dir / "control_volume.npy"), dtype=np.float64
            ).reshape(-1)
        else:
            operator_point_weights = np.ones(count, dtype=np.float64)
        return V6DualRobinExample(
            sample_id=sample_id,
            condition=V1SteadyConditionInput(
                coords=coords,
                condition_features=features,
                condition_feature_names=V6_DUAL_ROBIN_CONDITION_FEATURES,
                k_encoding_mode="diag3",
            ),
            target=V1SteadyTarget(target_u=temperature),
            meta=enriched_meta,
            operator_point_weights=operator_point_weights,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> V6DualRobinExample:
        return self.samples[index]

    def sample_index_by_id(self) -> dict[str, int]:
        return {sample.sample_id: index for index, sample in enumerate(self.samples)}


def _load_matrix(path: Path, width: int) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != width or not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: expected finite [N,{width}], found {value.shape}")
    return value


def _load_column(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: expected finite [N] or [N,1], found {value.shape}")
    return value
