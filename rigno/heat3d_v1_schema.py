"""Schema helpers for the Heat3D v1 metadata-first subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "v1.0"
SUBSET_NAME = "v1_multilayer_bc_eq_demo"
VALID_SUBSET_NAMES = {
  "v1_multilayer_bc_eq_demo",
  "v1_multilayer_bc_eq_supervised_smoke",
  "v1_multilayer_bc_eq_supervised_small",
}

VALID_STAGES = {"metadata_only", "solver_smoke", "supervised_smoke"}
VALID_SPLITS = {
  "train",
  "valid",
  "test_id",
  "test_smoke",
  "test_ood_stack",
  "test_ood_bc",
  "test_ood_material",
}
VALID_BOUNDARY_TYPES = {"Dirichlet", "Neumann", "Robin", "adiabatic"}
VALID_INTERFACE_TYPES = {"perfect_contact", "contact_resistance"}

REQUIRED_ARRAYS = (
  "coords.npy",
  "layer_id.npy",
  "region_id.npy",
  "material_id.npy",
  "k_field.npy",
  "q_field.npy",
)
REQUIRED_META_FIELDS = (
  "schema_version",
  "subset_name",
  "sample_id",
  "stage",
  "split",
  "domain",
  "layers",
  "regions",
  "materials",
  "boundary_regions",
  "boundary_types",
  "boundary_params",
  "interfaces",
  "generation_config",
  "units",
  "validation",
  "parameter_sources",
)
REQUIRED_UNITS = {
  "coords": "m",
  "k_field": "W/(m*K)",
  "q_field": "W/m^3",
  "temperature": "K",
  "thickness": "m",
  "htc": "W/(m^2*K)",
}
REQUIRED_PARAMETER_SOURCE_KEYS = (
  "literature_backed",
  "provisional_engineering_assumption",
  "requires_user_confirmation",
)


def default_v1_samples_dir(repo_dir: str | Path | None = None) -> Path:
  """Returns the default local v1 metadata-smoke sample directory."""

  root = Path(repo_dir).resolve() if repo_dir is not None else Path(__file__).resolve().parents[1]
  return root / "data" / "heat3d-thermal-simulation" / "subsets" / SUBSET_NAME / "samples"


def find_sample_dirs(path: str | Path) -> list[Path]:
  """Returns sample directories below a sample or subset path."""

  root = Path(path)
  if root.name.startswith("sample_") and root.is_dir():
    return [root]

  if (root / "samples").is_dir():
    root = root / "samples"

  if not root.is_dir():
    return []

  return sorted(
    child for child in root.iterdir()
    if child.is_dir() and child.name.startswith("sample_")
  )


def load_sample_meta(sample_dir: str | Path) -> dict[str, Any]:
  """Loads sample_meta.json from a v1 sample directory."""

  meta_path = Path(sample_dir) / "sample_meta.json"
  with meta_path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _load_array(sample_dir: Path, name: str, errors: list[str]) -> np.ndarray | None:
  path = sample_dir / name
  if not path.exists():
    errors.append(f"missing required array: {name}")
    return None
  try:
    return np.load(path)
  except Exception as exc:  # pragma: no cover - defensive error reporting
    errors.append(f"failed to load {name}: {exc}")
    return None


def _load_required_arrays(sample_dir: Path, errors: list[str]) -> dict[str, np.ndarray | None]:
  return {name: _load_array(sample_dir, name, errors) for name in REQUIRED_ARRAYS}


def _check_required_meta(meta: dict[str, Any], errors: list[str]) -> None:
  for field in REQUIRED_META_FIELDS:
    if field not in meta:
      errors.append(f"sample_meta.json missing required field: {field}")


def _check_shapes(arrays: dict[str, np.ndarray | None], errors: list[str]) -> dict[str, tuple[int, ...]]:
  if any(value is None for value in arrays.values()):
    return {}

  coords = arrays["coords.npy"]
  layer_id = arrays["layer_id.npy"]
  region_id = arrays["region_id.npy"]
  material_id = arrays["material_id.npy"]
  k_field = arrays["k_field.npy"]
  q_field = arrays["q_field.npy"]

  assert coords is not None
  assert layer_id is not None
  assert region_id is not None
  assert material_id is not None
  assert k_field is not None
  assert q_field is not None

  if coords.ndim != 2 or coords.shape[1] != 3:
    errors.append(f"coords.npy must have shape (N, 3), found {coords.shape}")
    return {name: tuple(value.shape) for name, value in arrays.items() if value is not None}

  n_points = coords.shape[0]
  for name, array in (
    ("layer_id.npy", layer_id),
    ("region_id.npy", region_id),
    ("material_id.npy", material_id),
  ):
    if array.ndim != 1 or array.shape[0] != n_points:
      errors.append(f"{name} must have shape (N,), found {array.shape} for N={n_points}")

  if k_field.ndim != 2 or k_field.shape[0] != n_points or k_field.shape[1] not in (1, 3, 6):
    errors.append(
      f"k_field.npy must have shape (N, 1), (N, 3), or (N, 6); found {k_field.shape}"
    )

  if q_field.ndim != 2 or q_field.shape != (n_points, 1):
    errors.append(f"q_field.npy must have shape (N, 1), found {q_field.shape}")

  return {name: tuple(value.shape) for name, value in arrays.items() if value is not None}


def _collect_ids(meta_items: list[Any], key: str) -> set[int]:
  values = set()
  for item in meta_items:
    if isinstance(item, dict) and key in item:
      values.add(item[key])
  return values


def _check_array_metadata_consistency(
  meta: dict[str, Any],
  arrays: dict[str, np.ndarray | None],
  errors: list[str],
) -> None:
  coords = arrays.get("coords.npy")
  layer_id = arrays.get("layer_id.npy")
  region_id = arrays.get("region_id.npy")
  material_id = arrays.get("material_id.npy")

  if coords is None or layer_id is None or region_id is None or material_id is None:
    return

  layers = meta.get("layers", [])
  regions = meta.get("regions", [])
  materials = meta.get("materials", [])

  if not isinstance(layers, list):
    errors.append("layers must be a list")
    return
  if not isinstance(regions, list):
    errors.append("regions must be a list")
    return
  if not isinstance(materials, list):
    errors.append("materials must be a list")
    return

  valid_layer_ids = _collect_ids(layers, "id")
  valid_region_ids = _collect_ids(regions, "id")
  valid_material_ids = _collect_ids(materials, "id")

  array_layer_ids = {int(value) for value in np.unique(layer_id)}
  array_region_ids = {int(value) for value in np.unique(region_id)}
  array_material_ids = {int(value) for value in np.unique(material_id)}

  missing_layers = sorted(array_layer_ids - valid_layer_ids)
  missing_regions = sorted(array_region_ids - valid_region_ids)
  missing_materials = sorted(array_material_ids - valid_material_ids)
  if missing_layers:
    errors.append(f"layer_id.npy contains ids missing from metadata.layers: {missing_layers}")
  if missing_regions:
    errors.append(f"region_id.npy contains ids missing from metadata.regions: {missing_regions}")
  if missing_materials:
    errors.append(f"material_id.npy contains ids missing from metadata.materials: {missing_materials}")

  for region in regions:
    if not isinstance(region, dict):
      errors.append("each regions entry must be a dictionary")
      continue
    layer_ref = region.get("layer_id")
    material_ref = region.get("material_id")
    if layer_ref not in valid_layer_ids:
      errors.append(f"region {region.get('id')} references invalid layer_id: {layer_ref}")
    if material_ref not in valid_material_ids:
      errors.append(f"region {region.get('id')} references invalid material_id: {material_ref}")


def _check_boundary_indices(
  meta: dict[str, Any],
  arrays: dict[str, np.ndarray | None],
  errors: list[str],
) -> None:
  coords = arrays.get("coords.npy")
  if coords is None:
    return

  n_points = coords.shape[0]
  boundary_regions = meta.get("boundary_regions", [])
  if not isinstance(boundary_regions, list):
    return

  for region in boundary_regions:
    if not isinstance(region, dict):
      continue
    point_indices = region.get("point_indices")
    if point_indices is None:
      continue
    if not isinstance(point_indices, list):
      errors.append(f"boundary region {region.get('name')} point_indices must be a list")
      continue
    for index in point_indices:
      if not isinstance(index, int):
        errors.append(f"boundary region {region.get('name')} has non-integer point index: {index!r}")
        continue
      if index < 0 or index >= n_points:
        errors.append(
          f"boundary region {region.get('name')} point index out of range: {index} for N={n_points}"
        )


def _check_heat_layer_consistency(
  meta: dict[str, Any],
  arrays: dict[str, np.ndarray | None],
  errors: list[str],
) -> None:
  layer_id = arrays.get("layer_id.npy")
  q_field = arrays.get("q_field.npy")
  if layer_id is None or q_field is None:
    return

  generation_config = meta.get("generation_config", {})
  if not isinstance(generation_config, dict):
    errors.append("generation_config must be a dictionary")
    return

  heat_layers = generation_config.get("heat_layers", [])
  if not heat_layers:
    return
  if not isinstance(heat_layers, list):
    errors.append("generation_config.heat_layers must be a list")
    return

  layers = meta.get("layers", [])
  if not isinstance(layers, list):
    errors.append("layers must be a list")
    return
  layer_name_to_id = {
    layer.get("name"): layer.get("id")
    for layer in layers
    if isinstance(layer, dict) and "name" in layer and "id" in layer
  }

  q_nonzero = np.abs(q_field[:, 0]) > 0.0
  for layer_name in heat_layers:
    if layer_name not in layer_name_to_id:
      errors.append(f"generation_config.heat_layers references unknown layer: {layer_name!r}")
      continue
    target_layer_id = layer_name_to_id[layer_name]
    target_mask = layer_id == target_layer_id
    if not np.any(q_nonzero & target_mask):
      errors.append(
        f"q_field has no nonzero values on declared heat layer {layer_name!r} (layer_id={target_layer_id})"
      )


def _check_units(meta: dict[str, Any], errors: list[str]) -> None:
  units = meta.get("units", {})
  if not isinstance(units, dict):
    errors.append("units must be a dictionary")
    return

  for key, expected in REQUIRED_UNITS.items():
    actual = units.get(key)
    if actual is None:
      errors.append(f"units missing required key: {key}")
    elif actual != expected:
      errors.append(f"units.{key} must be {expected!r}, found {actual!r}")


def _check_parameter_sources(meta: dict[str, Any], errors: list[str]) -> None:
  sources = meta.get("parameter_sources", {})
  if not isinstance(sources, dict):
    errors.append("parameter_sources must be a dictionary")
    return

  for key in REQUIRED_PARAMETER_SOURCE_KEYS:
    if key not in sources:
      errors.append(f"parameter_sources missing required key: {key}")
    elif not isinstance(sources[key], list):
      errors.append(f"parameter_sources.{key} must be a list")


def _check_boundaries(meta: dict[str, Any], errors: list[str]) -> None:
  boundary_regions = meta.get("boundary_regions", [])
  boundary_types = meta.get("boundary_types", {})
  boundary_params = meta.get("boundary_params", {})

  if not isinstance(boundary_regions, list):
    errors.append("boundary_regions must be a list")
    return
  if not isinstance(boundary_types, dict):
    errors.append("boundary_types must be a dictionary")
    return
  if not isinstance(boundary_params, dict):
    errors.append("boundary_params must be a dictionary")
    return

  names = []
  for region in boundary_regions:
    if not isinstance(region, dict) or "name" not in region:
      errors.append("each boundary_regions entry must be a dictionary with a name")
      continue
    names.append(region["name"])

  for name in names:
    if name not in boundary_types:
      errors.append(f"boundary_types missing entry for boundary region: {name}")
    elif boundary_types[name] not in VALID_BOUNDARY_TYPES:
      errors.append(f"boundary_types.{name} has invalid type: {boundary_types[name]!r}")

    if name not in boundary_params:
      errors.append(f"boundary_params missing entry for boundary region: {name}")

  extra_type_names = sorted(set(boundary_types) - set(names))
  extra_param_names = sorted(set(boundary_params) - set(names))
  if extra_type_names:
    errors.append(f"boundary_types has entries without boundary_regions: {extra_type_names}")
  if extra_param_names:
    errors.append(f"boundary_params has entries without boundary_regions: {extra_param_names}")

  expected_first_stage = {
    "top": "Robin",
    "bottom": "Dirichlet",
    "sides": "adiabatic",
  }
  for name, expected_type in expected_first_stage.items():
    if boundary_types.get(name) != expected_type:
      errors.append(
        f"first-stage BC expects {name}={expected_type}, found {boundary_types.get(name)!r}"
      )


def _check_interfaces(meta: dict[str, Any], errors: list[str]) -> None:
  layers = meta.get("layers", [])
  interfaces = meta.get("interfaces", [])

  if not isinstance(layers, list):
    errors.append("layers must be a list")
    return
  if not isinstance(interfaces, list):
    errors.append("interfaces must be a list")
    return

  layer_ids = {layer.get("id") for layer in layers if isinstance(layer, dict)}
  for interface in interfaces:
    if not isinstance(interface, dict):
      errors.append("each interface entry must be a dictionary")
      continue

    interface_type = interface.get("type")
    if interface_type not in VALID_INTERFACE_TYPES:
      errors.append(f"interface has invalid type: {interface_type!r}")

    lower = interface.get("lower_layer_id")
    upper = interface.get("upper_layer_id")
    if lower not in layer_ids or upper not in layer_ids:
      errors.append(f"interface references invalid layer pair: {lower}, {upper}")
    if lower == upper:
      errors.append(f"interface references identical layer ids: {lower}")


def validate_sample(sample_dir: str | Path, require_metadata_only: bool = True) -> dict[str, Any]:
  """Validates one v1 sample directory.

  Returns a dictionary with errors, warnings, array shapes, and selected metadata.
  """

  sample_path = Path(sample_dir)
  errors: list[str] = []
  warnings: list[str] = []

  meta_path = sample_path / "sample_meta.json"
  if not meta_path.exists():
    return {
      "sample_dir": str(sample_path),
      "errors": ["missing sample_meta.json"],
      "warnings": warnings,
      "shapes": {},
      "meta": {},
    }

  try:
    meta = load_sample_meta(sample_path)
  except Exception as exc:
    return {
      "sample_dir": str(sample_path),
      "errors": [f"failed to load sample_meta.json: {exc}"],
      "warnings": warnings,
      "shapes": {},
      "meta": {},
    }

  _check_required_meta(meta, errors)
  arrays = _load_required_arrays(sample_path, errors)
  shapes = _check_shapes(arrays, errors)

  if meta.get("schema_version") != SCHEMA_VERSION:
    errors.append(f"schema_version must be {SCHEMA_VERSION!r}, found {meta.get('schema_version')!r}")
  if meta.get("subset_name") not in VALID_SUBSET_NAMES:
    errors.append(
      f"subset_name must be one of {sorted(VALID_SUBSET_NAMES)!r}, found {meta.get('subset_name')!r}"
    )
  if meta.get("sample_id") != sample_path.name:
    errors.append(f"sample_id must match directory name {sample_path.name!r}")

  stage = meta.get("stage")
  if stage not in VALID_STAGES:
    errors.append(f"stage must be one of {sorted(VALID_STAGES)}, found {stage!r}")
  if require_metadata_only and stage != "metadata_only":
    errors.append(f"metadata smoke validation requires stage='metadata_only', found {stage!r}")
  if stage == "metadata_only" and (sample_path / "temperature.npy").exists():
    errors.append("metadata_only sample must not include temperature.npy")
  if stage in {"solver_smoke", "supervised_smoke"} and not (sample_path / "temperature.npy").exists():
    errors.append(f"{stage} sample must include temperature.npy")

  split = meta.get("split")
  if split not in VALID_SPLITS:
    errors.append(f"split must be one of {sorted(VALID_SPLITS)}, found {split!r}")

  _check_boundaries(meta, errors)
  _check_interfaces(meta, errors)
  _check_units(meta, errors)
  _check_parameter_sources(meta, errors)
  _check_array_metadata_consistency(meta, arrays, errors)
  _check_boundary_indices(meta, arrays, errors)
  _check_heat_layer_consistency(meta, arrays, errors)

  return {
    "sample_dir": str(sample_path),
    "errors": errors,
    "warnings": warnings,
    "shapes": shapes,
    "meta": {
      "sample_id": meta.get("sample_id"),
      "stage": meta.get("stage"),
      "split": meta.get("split"),
      "layer_count": len(meta.get("layers", [])) if isinstance(meta.get("layers"), list) else None,
      "region_count": len(meta.get("regions", [])) if isinstance(meta.get("regions"), list) else None,
      "material_count": len(meta.get("materials", [])) if isinstance(meta.get("materials"), list) else None,
      "interface_count": len(meta.get("interfaces", [])) if isinstance(meta.get("interfaces"), list) else None,
    },
  }


def validate_path(path: str | Path, require_metadata_only: bool = True) -> dict[str, Any]:
  """Validates one sample or every sample below a subset path."""

  sample_dirs = find_sample_dirs(path)
  results = [validate_sample(sample_dir, require_metadata_only) for sample_dir in sample_dirs]
  return {
    "root": str(Path(path)),
    "sample_count": len(sample_dirs),
    "results": results,
    "error_count": sum(len(result["errors"]) for result in results),
    "warning_count": sum(len(result["warnings"]) for result in results),
  }


def summarize_sample(sample_dir: str | Path) -> dict[str, Any]:
  """Returns a compact summary for inspection output."""

  sample_path = Path(sample_dir)
  meta = load_sample_meta(sample_path)
  shapes = {}
  for name in REQUIRED_ARRAYS + ("temperature.npy",):
    path = sample_path / name
    if path.exists():
      shapes[name] = tuple(np.load(path).shape)

  q_nonzero_layer_ids: list[int] = []
  q_nonzero_layer_names: list[str] = []
  if (sample_path / "layer_id.npy").exists() and (sample_path / "q_field.npy").exists():
    layer_ids = np.load(sample_path / "layer_id.npy")
    q_field = np.load(sample_path / "q_field.npy")
    q_nonzero_layer_ids = sorted({int(value) for value in np.unique(layer_ids[np.abs(q_field[:, 0]) > 0.0])})
    layer_lookup = {
      layer.get("id"): layer.get("name")
      for layer in meta.get("layers", [])
      if isinstance(layer, dict)
    }
    q_nonzero_layer_names = [layer_lookup.get(layer_id, f"layer_{layer_id}") for layer_id in q_nonzero_layer_ids]

  parameter_sources = meta.get("parameter_sources") or {}
  parameter_source_counts = {
    key: len(parameter_sources.get(key, []))
    for key in REQUIRED_PARAMETER_SOURCE_KEYS
  }

  return {
    "sample_dir": str(sample_path),
    "sample_id": meta.get("sample_id"),
    "stage": meta.get("stage"),
    "split": meta.get("split"),
    "layers": [layer.get("name") for layer in meta.get("layers", [])],
    "boundary_types": meta.get("boundary_types"),
    "interfaces": [
      {
        "name": interface.get("name"),
        "type": interface.get("type"),
        "lower_layer_id": interface.get("lower_layer_id"),
        "upper_layer_id": interface.get("upper_layer_id"),
      }
      for interface in meta.get("interfaces", [])
    ],
    "shapes": shapes,
    "parameter_sources": parameter_sources,
    "parameter_source_counts": parameter_source_counts,
    "k_field_shape": shapes.get("k_field.npy"),
    "q_nonzero_layer_ids": q_nonzero_layer_ids,
    "q_nonzero_layer_names": q_nonzero_layer_names,
  }
