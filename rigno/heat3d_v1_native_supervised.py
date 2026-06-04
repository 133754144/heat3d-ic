"""Native v1 supervised data structures for steady temperature prediction.

This module keeps v1 problem semantics separate from the legacy
`Inputs(u, c, ...)` model API. In the native contract, the target field is the
steady temperature field, and the condition features are known physical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir
from rigno.models.operator import Inputs


@dataclass(frozen=True)
class V1SteadyConditionInput:
  coords: np.ndarray
  condition_features: np.ndarray
  condition_feature_names: tuple[str, ...]
  k_encoding_mode: str


@dataclass(frozen=True)
class V1SteadyTarget:
  target_u: np.ndarray
  target_name: str = "temperature"
  target_role: str = "steady temperature supervised target"


@dataclass(frozen=True)
class V1RelativeBCFeatureView:
  condition_features: np.ndarray
  condition_feature_names: tuple[str, ...]
  t_ref_value: float
  t_ref_source: str
  view_name: str = "relative_bc_condition_features"
  view_role: str = (
    "optional feature view for temperature-rise diagnostics; not the default loader contract"
  )


@dataclass(frozen=True)
class V1TemperatureRiseLegacyBridge:
  legacy_inputs: Inputs
  target_delta_u: jnp.ndarray
  t_ref: jnp.ndarray
  t_ref_value: float
  t_ref_source: str
  condition_feature_names: tuple[str, ...] = ()
  bridge_policy: str = "tref_u_bridge"
  bridge_name: str = "temperature_rise_legacy_bridge"
  target_delta_name: str = "delta_temperature"
  bridge_role: str = "internal legacy bridge; not v1 canonical semantics"


@dataclass(frozen=True)
class V1SteadySupervisedExampleNative:
  sample_id: str
  condition: V1SteadyConditionInput
  target: V1SteadyTarget
  meta: dict[str, Any]

  def to_legacy_inputs(self, split_u_channels: int = 1) -> Inputs:
    """Packs condition features into the legacy RIGNO input API.

    This is not the canonical v1 semantics. It exists only as a bridge to the
    current `Inputs(u, c, ...)` model interface.
    """

    features = self.condition.condition_features
    if split_u_channels < 1:
      raise ValueError("split_u_channels must be at least 1")
    if split_u_channels > features.shape[1]:
      raise ValueError(
        f"split_u_channels={split_u_channels} exceeds feature dimension {features.shape[1]}"
      )

    n_points = self.condition.coords.shape[0]
    u = jnp.asarray(features[:, :split_u_channels].reshape(1, 1, n_points, split_u_channels))
    c = None
    if features.shape[1] > split_u_channels:
      c = jnp.asarray(features[:, split_u_channels:].reshape(1, 1, n_points, -1))

    coords = jnp.asarray(self.condition.coords.reshape(1, 1, n_points, 3))
    return Inputs(
      u=u,
      c=c,
      x_inp=coords,
      x_out=coords,
      t=None,
      tau=None,
    )

  def build_temperature_rise_legacy_inputs(self) -> V1TemperatureRiseLegacyBridge:
    """Builds a non-leaking temperature-rise bridge for the legacy RIGNO API.

    Native semantics remain `condition_features -> target_temperature`. This
    bridge uses a metadata-derived constant baseline temperature as legacy
    `Inputs.u`, places all condition features in `Inputs.c`, and supervises the
    model against `target_delta_u = target_temperature - T_ref`.
    """

    t_ref_value, t_ref_source = _resolve_t_ref_value(self.meta)
    return self._build_temperature_rise_bridge(
      condition_features=self.condition.condition_features,
      condition_feature_names=self.condition.condition_feature_names,
      t_ref_value=t_ref_value,
      t_ref_source=t_ref_source,
      bridge_policy="tref_u_bridge",
    )

  def get_relative_bc_feature_view(self) -> V1RelativeBCFeatureView:
    """Returns an optional feature view with BC temperatures relative to T_ref.

    This does not modify the default condition feature contract. It replaces
    raw absolute `top_T_inf` and `bottom_T_fixed` columns with
    `top_T_inf_minus_T_ref` and `bottom_T_fixed_minus_T_ref` for temperature-rise
    diagnostics.
    """

    names = self.condition.condition_feature_names
    features = np.asarray(self.condition.condition_features, dtype=np.float64)
    top_index = _feature_index(names, "top_T_inf")
    bottom_index = _feature_index(names, "bottom_T_fixed")
    t_ref_value, t_ref_source = _resolve_t_ref_value(self.meta)

    relative_features = features.copy()
    relative_features[:, top_index] = features[:, top_index] - t_ref_value
    relative_features[:, bottom_index] = features[:, bottom_index] - t_ref_value

    relative_names = list(names)
    relative_names[top_index] = "top_T_inf_minus_T_ref"
    relative_names[bottom_index] = "bottom_T_fixed_minus_T_ref"

    return V1RelativeBCFeatureView(
      condition_features=relative_features,
      condition_feature_names=tuple(relative_names),
      t_ref_value=t_ref_value,
      t_ref_source=t_ref_source,
    )

  def build_temperature_rise_legacy_inputs_from_relative_features(
    self,
    bridge_policy: str = "tref_u_bridge",
  ) -> V1TemperatureRiseLegacyBridge:
    """Builds a temperature-rise bridge from the optional relative BC view.

    Supported policies:
    - `tref_u_bridge`: legacy `u` is the constant non-leaking `T_ref` field.
    - `zero_delta_u_bridge`: legacy `u` is a zero delta-temperature field; `T_ref`
      is kept only for target construction and final temperature recovery.
    """

    relative_view = self.get_relative_bc_feature_view()
    return self._build_temperature_rise_bridge(
      condition_features=relative_view.condition_features,
      condition_feature_names=relative_view.condition_feature_names,
      t_ref_value=relative_view.t_ref_value,
      t_ref_source=relative_view.t_ref_source,
      bridge_policy=bridge_policy,
    )

  def _build_temperature_rise_bridge(
    self,
    condition_features: np.ndarray,
    condition_feature_names: tuple[str, ...],
    t_ref_value: float,
    t_ref_source: str,
    bridge_policy: str,
  ) -> V1TemperatureRiseLegacyBridge:
    if bridge_policy not in {"tref_u_bridge", "zero_delta_u_bridge"}:
      raise ValueError(
        "bridge_policy must be one of {'tref_u_bridge', 'zero_delta_u_bridge'}, "
        f"found {bridge_policy!r}"
      )

    n_points = self.condition.coords.shape[0]
    t_ref = jnp.full((1, 1, n_points, 1), t_ref_value, dtype=jnp.float32)
    target_temperature = jnp.asarray(self.target.target_u.reshape(1, 1, n_points, 1))
    target_delta_u = target_temperature - t_ref
    legacy_u = t_ref if bridge_policy == "tref_u_bridge" else jnp.zeros_like(t_ref)
    condition_features_jnp = jnp.asarray(condition_features.reshape(1, 1, n_points, -1))
    coords = jnp.asarray(self.condition.coords.reshape(1, 1, n_points, 3))

    legacy_inputs = Inputs(
      u=legacy_u,
      c=condition_features_jnp,
      x_inp=coords,
      x_out=coords,
      t=None,
      tau=None,
    )
    return V1TemperatureRiseLegacyBridge(
      legacy_inputs=legacy_inputs,
      target_delta_u=target_delta_u,
      t_ref=t_ref,
      t_ref_value=t_ref_value,
      t_ref_source=t_ref_source,
      condition_feature_names=condition_feature_names,
      bridge_policy=bridge_policy,
    )


class Heat3DV1NativeSupervisedDataset:
  """Builds v1-native supervised examples from the tiny supervised smoke subset."""

  def __init__(
    self,
    datadir: str | Path | None = None,
    repo_dir: str | Path | None = None,
    k_encoding_mode: str = "diag3",
    boundary_mask_fallback: bool = True,
  ) -> None:
    self._legacy_dataset = Heat3DV1SupervisedDataset(
      datadir=(default_v1_supervised_samples_dir(repo_dir) if datadir is None else datadir),
      repo_dir=repo_dir,
      k_encoding_mode=k_encoding_mode,
      boundary_mask_fallback=boundary_mask_fallback,
    )
    self.k_encoding_mode = k_encoding_mode
    self.boundary_mask_fallback = bool(boundary_mask_fallback)
    self.samples = [self._to_native(sample) for sample in self._legacy_dataset.samples]

  def _to_native(self, sample: dict[str, Any]) -> V1SteadySupervisedExampleNative:
    condition = V1SteadyConditionInput(
      coords=sample["coords"],
      condition_features=sample["physics_input"].features,
      condition_feature_names=sample["physics_input"].feature_names,
      k_encoding_mode=self.k_encoding_mode,
    )
    target = V1SteadyTarget(
      target_u=sample["temperature"],
    )
    return V1SteadySupervisedExampleNative(
      sample_id=sample["sample_id"],
      condition=condition,
      target=target,
      meta=sample["meta"],
    )

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> V1SteadySupervisedExampleNative:
    return self.samples[index]

  def sample_index_by_id(self) -> dict[str, int]:
    return {sample.sample_id: index for index, sample in enumerate(self.samples)}


def _resolve_t_ref_value(meta: dict[str, Any]) -> tuple[float, str]:
  boundary_params = meta.get("boundary_params", {})
  if isinstance(boundary_params, dict):
    bottom = boundary_params.get("bottom", {})
    if isinstance(bottom, dict) and "fixed_temperature_K" in bottom:
      return float(bottom["fixed_temperature_K"]), "bottom_dirichlet_fixed_temperature_K"

    top = boundary_params.get("top", {})
    if isinstance(top, dict) and "ambient_temperature_K" in top:
      return float(top["ambient_temperature_K"]), "top_robin_ambient_temperature_K"

  return 300.0, "fallback_300K"


def _feature_index(feature_names: tuple[str, ...], name: str) -> int:
  try:
    return feature_names.index(name)
  except ValueError as exc:
    raise ValueError(f"Required feature {name!r} not found in {feature_names}") from exc
