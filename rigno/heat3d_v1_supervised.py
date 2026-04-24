"""Supervised v1 smoke dataset and adapter for steady temperature prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rigno.dataset_Heat3D_v1 import Heat3DV1MetadataDataset, V1OperatorInterfaceInputs
from rigno.models.operator import Inputs


SUPERVISED_SUBSET_NAME = "v1_multilayer_bc_eq_supervised_smoke"


def default_v1_supervised_samples_dir(repo_dir: str | Path | None = None) -> Path:
  root = Path(repo_dir).resolve() if repo_dir is not None else Path(__file__).resolve().parents[1]
  return root / "data" / "heat3d-thermal-simulation" / "subsets" / SUPERVISED_SUBSET_NAME / "samples"


@dataclass(frozen=True)
class V1SupervisedExample:
  sample_id: str
  inputs: Inputs
  target_temperature: jnp.ndarray
  full_feature_names: tuple[str, ...]
  u_feature_names: tuple[str, ...]
  c_feature_names: tuple[str, ...]
  target_role: str


class Heat3DV1SupervisedDataset(Heat3DV1MetadataDataset):
  """Very small supervised smoke dataset for steady temperature prediction."""

  def __init__(
    self,
    datadir: str | Path | None = None,
    repo_dir: str | Path | None = None,
    input_mode: str = "pure_physics",
    k_encoding_mode: str = "diag3",
  ) -> None:
    super().__init__(
      datadir=(default_v1_supervised_samples_dir(repo_dir) if datadir is None else datadir),
      repo_dir=repo_dir,
      input_mode=input_mode,
      k_encoding_mode=k_encoding_mode,
      allowed_stages=("supervised_smoke", "solver_smoke"),
    )

  def _load_sample(self, sample_dir: Path) -> dict[str, Any]:
    sample = super()._load_sample(sample_dir)
    temperature = np.load(sample_dir / "temperature.npy")
    if temperature.ndim != 2 or temperature.shape[1] != 1:
      raise ValueError(f"temperature.npy must have shape (N,1), found {temperature.shape}")
    if temperature.shape[0] != sample["coords"].shape[0]:
      raise ValueError("temperature.npy and coords.npy must share the same N")
    sample["temperature"] = temperature
    return sample

  def get_supervised_example(self, index: int) -> V1SupervisedExample:
    operator_inputs: V1OperatorInterfaceInputs = self.get_operator_interface_inputs(index)
    sample = self.samples[index]
    target = jnp.asarray(sample["temperature"].reshape(1, 1, sample["temperature"].shape[0], 1))
    return V1SupervisedExample(
      sample_id=sample["sample_id"],
      inputs=operator_inputs.inputs,
      target_temperature=target,
      full_feature_names=operator_inputs.full_feature_names,
      u_feature_names=operator_inputs.u_feature_names,
      c_feature_names=operator_inputs.c_feature_names,
      target_role="supervised target / steady temperature label",
    )

  def describe(self) -> dict[str, Any]:
    info = super().describe()
    info["temperature_supported"] = True
    info["supervised_target"] = "temperature.npy"
    info["training_pipeline_integration"] = False
    return info
