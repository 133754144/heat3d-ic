"""Metadata-first Heat3D v1 dataset loader skeleton.

This module does not integrate with the v0 training pipeline. It defines how the
v1 metadata-first subset can be read and converted into default pure-physics
model inputs for later experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rigno.models.operator import Inputs
from rigno.heat3d_v1_schema import default_v1_samples_dir, find_sample_dirs, load_sample_meta


BC_FEATURE_NAMES = (
  "is_top",
  "is_bottom",
  "is_side",
  "is_interior",
  "top_h",
  "top_T_inf",
  "bottom_T_fixed",
)

K_TENSOR6_FEATURE_NAMES = (
  "k_xx",
  "k_yy",
  "k_zz",
  "k_xy",
  "k_xz",
  "k_yz",
)

SUPPORTED_K_SHAPES = {
  "native": {"(N,1)": True, "(N,3)": True, "(N,6)": True},
  "diag3": {"(N,1)": True, "(N,3)": True, "(N,6)": False},
}


PURE_PHYSICS_BASE_FEATURE_NAMES = (
  "q",
  "is_top",
  "is_bottom",
  "is_side",
  "is_interior",
  "top_h",
  "top_T_inf",
  "bottom_T_fixed",
)


@dataclass(frozen=True)
class V1PhysicsInput:
  coords: np.ndarray
  raw_k_field: np.ndarray
  encoded_k_field: np.ndarray
  q_field: np.ndarray
  bc_encoding: np.ndarray
  features: np.ndarray
  feature_names: tuple[str, ...]


@dataclass(frozen=True)
class V1OperatorInterfaceInputs:
  inputs: Inputs
  full_feature_names: tuple[str, ...]
  u_feature_names: tuple[str, ...]
  c_feature_names: tuple[str, ...]
  adapter_note: str


class Heat3DV1MetadataDataset:
  """Loads metadata-first v1 samples without solver or training integration.

  Default input mode is pure-physics:

  - coords
  - k_field
  - q_field
  - BC encoding

  Semantic arrays `layer_id`, `region_id`, and `material_id` are retained as
  optional auxiliary metadata but are not required by the default input path.
  """

  def __init__(
    self,
    datadir: str | Path | None = None,
    repo_dir: str | Path | None = None,
    input_mode: str = "pure_physics",
    k_encoding_mode: str = "native",
    allowed_stages: tuple[str, ...] = ("metadata_only",),
  ) -> None:
    if input_mode != "pure_physics":
      raise ValueError("Heat3DV1MetadataDataset currently supports only input_mode='pure_physics'")
    if k_encoding_mode not in SUPPORTED_K_SHAPES:
      raise ValueError(
        f"Unsupported k_encoding_mode={k_encoding_mode!r}; expected one of {sorted(SUPPORTED_K_SHAPES)}"
      )

    self.datadir = (
      Path(datadir).resolve()
      if datadir is not None
      else default_v1_samples_dir(repo_dir)
    )
    self.sample_dirs = find_sample_dirs(self.datadir)
    if not self.sample_dirs:
      raise FileNotFoundError(f"No v1 sample directories found under {self.datadir}")

    self.input_mode = input_mode
    self.k_encoding_mode = k_encoding_mode
    self.allowed_stages = tuple(allowed_stages)
    self.samples = [self._load_sample(sample_dir) for sample_dir in self.sample_dirs]

  @staticmethod
  def _encode_k_field(k_field: np.ndarray, k_encoding_mode: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if k_field.ndim != 2:
      raise ValueError(f"k_field must be 2D with shape (N, C), found {k_field.shape}")

    channels = k_field.shape[1]
    if k_encoding_mode == "native":
      if channels == 1:
        return k_field, ("k_iso",)
      if channels == 3:
        return k_field, ("k_x", "k_y", "k_z")
      if channels == 6:
        return k_field, K_TENSOR6_FEATURE_NAMES
      raise ValueError(f"native k encoding expects C in {{1,3,6}}, found {channels}")

    if k_encoding_mode == "diag3":
      if channels == 1:
        encoded = np.repeat(k_field, repeats=3, axis=1)
        return encoded, ("k_x", "k_y", "k_z")
      if channels == 3:
        return k_field, ("k_x", "k_y", "k_z")
      if channels == 6:
        raise NotImplementedError(
          "k_encoding_mode='diag3' for symmetric tensor k_field with shape (N,6) "
          "is not implemented yet"
        )
      raise ValueError(f"diag3 k encoding expects C in {{1,3,6}}, found {channels}")

    raise ValueError(f"Unsupported k_encoding_mode={k_encoding_mode!r}")

  def _load_sample(self, sample_dir: Path) -> dict[str, Any]:
    meta = load_sample_meta(sample_dir)
    stage = meta.get("stage")
    if stage not in self.allowed_stages:
      raise ValueError(
        f"Heat3DV1MetadataDataset expects stage in {self.allowed_stages}, found {stage!r}"
      )

    coords = np.load(sample_dir / "coords.npy")
    layer_id = np.load(sample_dir / "layer_id.npy")
    region_id = np.load(sample_dir / "region_id.npy")
    material_id = np.load(sample_dir / "material_id.npy")
    k_field = np.load(sample_dir / "k_field.npy")
    q_field = np.load(sample_dir / "q_field.npy")

    encoded_k_field, k_feature_names = self._encode_k_field(k_field, self.k_encoding_mode)
    bc_encoding = self._encode_boundary_conditions(coords, meta)
    feature_names = k_feature_names + PURE_PHYSICS_BASE_FEATURE_NAMES
    features = np.concatenate([encoded_k_field, q_field, bc_encoding], axis=-1)
    if features.shape[1] != len(feature_names):
      raise ValueError(
        f"feature_names length {len(feature_names)} does not match feature dimension {features.shape[1]}"
      )

    return {
      "sample_dir": sample_dir,
      "sample_id": meta.get("sample_id"),
      "meta": meta,
      "coords": coords,
      "layer_id": layer_id,
      "region_id": region_id,
      "material_id": material_id,
      "k_field": k_field,
      "encoded_k_field": encoded_k_field,
      "k_feature_names": k_feature_names,
      "q_field": q_field,
      "bc_encoding": bc_encoding,
      "physics_input": V1PhysicsInput(
        coords=coords,
        raw_k_field=k_field,
        encoded_k_field=encoded_k_field,
        q_field=q_field,
        bc_encoding=bc_encoding,
        features=features,
        feature_names=feature_names,
      ),
    }

  @staticmethod
  def _boundary_masks(n_points: int, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = np.zeros((n_points, 1), dtype=np.float64)
    bottom = np.zeros((n_points, 1), dtype=np.float64)
    side = np.zeros((n_points, 1), dtype=np.float64)

    boundary_regions = meta.get("boundary_regions", [])
    if not isinstance(boundary_regions, list):
      return top, bottom, side

    for region in boundary_regions:
      if not isinstance(region, dict):
        continue
      name = region.get("name")
      indices = region.get("point_indices", [])
      if not isinstance(indices, list):
        continue
      valid = [index for index in indices if isinstance(index, int) and 0 <= index < n_points]
      if name == "top":
        top[valid, 0] = 1.0
      elif name == "bottom":
        bottom[valid, 0] = 1.0
      elif name == "sides":
        side[valid, 0] = 1.0

    return top, bottom, side

  @classmethod
  def _encode_boundary_conditions(cls, coords: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    n_points = coords.shape[0]
    top, bottom, side = cls._boundary_masks(n_points, meta)
    interior = ((top + bottom + side) == 0.0).astype(np.float64)

    params = meta.get("boundary_params", {})
    top_params = params.get("top", {}) if isinstance(params, dict) else {}
    bottom_params = params.get("bottom", {}) if isinstance(params, dict) else {}

    top_h = np.full((n_points, 1), float(top_params.get("h_W_m2K", 0.0)), dtype=np.float64)
    top_t_inf = np.full(
      (n_points, 1),
      float(top_params.get("ambient_temperature_K", 0.0)),
      dtype=np.float64,
    )
    bottom_t_fixed = np.full(
      (n_points, 1),
      float(bottom_params.get("fixed_temperature_K", 0.0)),
      dtype=np.float64,
    )

    return np.concatenate(
      [top, bottom, side, interior, top_h, top_t_inf, bottom_t_fixed],
      axis=-1,
    )

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> dict[str, Any]:
    return self.samples[index]

  def get_model_input(
    self,
    index: int,
    include_auxiliary_metadata: bool = False,
  ) -> dict[str, Any]:
    sample = self.samples[index]
    result = {
      "sample_id": sample["sample_id"],
      "input_mode": self.input_mode,
      "k_encoding_mode": self.k_encoding_mode,
      "coords": sample["physics_input"].coords,
      "raw_k_field": sample["physics_input"].raw_k_field,
      "encoded_k_field": sample["physics_input"].encoded_k_field,
      "q_field": sample["physics_input"].q_field,
      "bc_encoding": sample["physics_input"].bc_encoding,
      "features": sample["physics_input"].features,
      "feature_names": sample["physics_input"].feature_names,
    }

    if include_auxiliary_metadata:
      result["auxiliary_metadata"] = {
        "layer_id": sample["layer_id"],
        "region_id": sample["region_id"],
        "material_id": sample["material_id"],
      }

    return result

  def get_operator_interface_inputs(
    self,
    index: int,
    split_u_channels: int = 1,
  ) -> V1OperatorInterfaceInputs:
    """Packs v1 pure-physics features into the current operator input contract.

    This is a minimal compatibility adapter for the existing RIGNO interface:

    - `u` carries the first channel only, because the current operator assumes
      `inputs.u.shape[-1] == num_outputs`
    - `c` carries the remaining pure-physics feature channels

    The canonical v1 model-facing mode should therefore be established before
    this split, for example with `k_encoding_mode="diag3"`.
    """

    sample = self.samples[index]
    features = sample["physics_input"].features
    feature_names = sample["physics_input"].feature_names

    if split_u_channels < 1:
      raise ValueError("split_u_channels must be at least 1")
    if split_u_channels > features.shape[1]:
      raise ValueError(
        f"split_u_channels={split_u_channels} exceeds feature dimension {features.shape[1]}"
      )

    n_points = sample["coords"].shape[0]
    u = jnp.asarray(features[:, :split_u_channels].reshape(1, 1, n_points, split_u_channels))

    c = None
    c_feature_names: tuple[str, ...] = ()
    if features.shape[1] > split_u_channels:
      c_np = features[:, split_u_channels:]
      c = jnp.asarray(c_np.reshape(1, 1, n_points, c_np.shape[1]))
      c_feature_names = feature_names[split_u_channels:]

    x = jnp.asarray(sample["coords"].reshape(1, 1, n_points, sample["coords"].shape[1]))
    inputs = Inputs(
      u=u,
      c=c,
      x_inp=x,
      x_out=x,
      t=None,
      tau=None,
    )
    return V1OperatorInterfaceInputs(
      inputs=inputs,
      full_feature_names=feature_names,
      u_feature_names=feature_names[:split_u_channels],
      c_feature_names=c_feature_names,
      adapter_note=(
        "Current operator interface requires one u-channel because num_outputs=1. "
        "The v1 adapter therefore routes the first canonical feature channel into u "
        "and the remaining pure-physics channels into c."
      ),
    )

  def describe(self) -> dict[str, Any]:
    return {
      "datadir": str(self.datadir),
      "sample_count": len(self.samples),
      "input_mode": self.input_mode,
      "k_encoding_mode": self.k_encoding_mode,
      "supported_k_shapes": SUPPORTED_K_SHAPES[self.k_encoding_mode],
      "feature_names": list(self.samples[0]["physics_input"].feature_names) if self.samples else [],
      "stage_requirement": list(self.allowed_stages),
      "temperature_supported": False,
      "solver_supported": False,
      "training_pipeline_integration": False,
    }
