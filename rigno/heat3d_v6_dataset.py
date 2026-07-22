"""Canonical V6 P1g dual-Robin dataset adapter.

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


CANONICAL_V6_DATASET_ID = "heat3d_v6_p1g_geometry_deconfounded1024_v0"
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
        """Return the frozen P1g equal-weight operator-point measure."""

        count = int(self.condition.coords.shape[0])
        return np.full(count, 1.0 / count, dtype=np.float64)

    def v6_global_context_inputs(self) -> dict[str, Any]:
        view = self.get_relative_bc_feature_view()
        adapter = self.meta["v6_adapter"]
        sources = self.meta.get("sources") or []
        total_power = float(sum(float(row["source_power_W"]) for row in sources))
        total_thickness = float(
            sum(float(row["thickness_m"]) for row in self.meta["layers_bottom_to_top"])
        )
        return {
            "coords": self.condition.coords,
            "raw_condition": view.condition_features,
            "condition_feature_names": view.condition_feature_names,
            "reference_temperature_K": view.t_ref_value,
            "top_T_inf_K": float(adapter["top_T_inf_K"]),
            "bottom_T_inf_K": float(adapter["bottom_T_inf_K"]),
            "operator_point_weights": self.v6_operator_point_weights(),
            "package_total_power_W": total_power,
            "package_extents_m": (0.01, 0.01, total_thickness),
        }


class Heat3DV6DualRobinDataset:
    """Read canonical P1g samples and its manifest-locked split contract."""

    def __init__(self, datadir: str | Path, manifest_path: str | Path) -> None:
        self.datadir = Path(datadir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()
        self.split_ids = self._split_ids_from_manifest()
        self.samples = [self._load_sample(row) for row in self.manifest["samples"]]

    def _validate_manifest(self) -> None:
        if self.manifest.get("dataset_id") != CANONICAL_V6_DATASET_ID:
            raise ValueError(
                "V6 training loader accepts only canonical P1g-v0; found "
                f"{self.manifest.get('dataset_id')!r}"
            )
        rows = self.manifest.get("samples")
        if not isinstance(rows, list) or len(rows) != 1024:
            raise ValueError("canonical V6 manifest must contain exactly 1024 samples")
        ids = [str(row.get("sample_id") or "") for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("canonical V6 manifest sample IDs must be nonempty and unique")

    def _split_ids_from_manifest(self) -> dict[str, list[str]]:
        role_map = {"train": "train", "valid": "valid_iid", "test": "test_iid"}
        splits = {name: [] for name in EXPECTED_SPLIT_COUNTS}
        group_roles: dict[str, str] = {}
        for row in self.manifest["samples"]:
            role = role_map.get(str(row.get("split_role")))
            if role is None:
                raise ValueError(f"unsupported V6 manifest split_role={row.get('split_role')!r}")
            group_id = str(row.get("group_id") or "")
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

    def _load_sample(self, row: dict[str, Any]) -> V6DualRobinExample:
        sample_id = str(row["sample_id"])
        sample_dir = self.datadir / str(row.get("sample_dir") or sample_id)
        meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
        if meta.get("dataset_id") != CANONICAL_V6_DATASET_ID:
            raise ValueError(f"{sample_id}: sample dataset_id is not canonical P1g-v0")
        bc = meta.get("boundary_conditions") or {}
        top = bc.get("top") or {}
        bottom = bc.get("bottom") or {}
        if top.get("type") != "robin" or bottom.get("type") != "robin":
            raise ValueError(f"{sample_id}: both top and bottom must be Robin")

        coords = _load_matrix(sample_dir / "coords.npy", 3)
        k_field = _load_matrix(sample_dir / "k_field.npy", 3)
        q_field = _load_matrix(sample_dir / "q_field.npy", 1)
        flags = _load_matrix(sample_dir / "bc_features.npy", 4)
        temperature = _load_matrix(sample_dir / "temperature.npy", 1)
        count = coords.shape[0]
        if count != 1024 or any(
            array.shape[0] != count for array in (k_field, q_field, flags, temperature)
        ):
            raise ValueError(f"{sample_id}: canonical operator shape must be [1024,*]")
        if not np.allclose(np.sum(flags, axis=1), 1.0, atol=0.0, rtol=0.0):
            raise ValueError(f"{sample_id}: four BC flags must be one-hot")

        top_h = float(top["h_W_m2K"])
        bottom_h = float(bottom["h_W_m2K"])
        top_tinf = float(top["T_inf_K"])
        bottom_tinf = float(bottom["T_inf_K"])
        if min(top_h, bottom_h) <= 0.0:
            raise ValueError(f"{sample_id}: Robin h values must be positive")
        top_offset = top_tinf - bottom_tinf
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
            "dataset_id": CANONICAL_V6_DATASET_ID,
            "manifest_split_role": str(row["split_role"]),
            "group_id": str(row["group_id"]),
            "reference_temperature_K": bottom_tinf,
            "top_T_inf_K": top_tinf,
            "bottom_T_inf_K": bottom_tinf,
            "bottom_boundary_semantics": "robin_not_dirichlet",
            "operator_point_measure": "equal_weight_frozen_irregular_1024",
        }
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
