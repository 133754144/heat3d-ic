"""Manifest resolver helpers for Heat3D v1 supervised-small dry-runs.

This module resolves a supervised-small manifest into generator-readable
planning records. It does not generate arrays, write data, run the reference
solver, or start training.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


NOMINAL_Q_W_M3 = 1.0e8
NOMINAL_Q_SOURCE = "tools/generate_heat3d_v1_metadata_smoke.py::_q_field"
NOMINAL_TOP_H_W_M2K = 2000.0
NOMINAL_TOP_H_SOURCE = "tools/generate_heat3d_v1_metadata_smoke.py::_meta.boundary_params.top.h_W_m2K"

Q_SCALE_MULTIPLIERS = {
  "low": 0.5,
  "nominal": 1.0,
  "high": 1.5,
}
TOP_H_MULTIPLIERS = {
  "low": 0.5,
  "nominal": 1.0,
  "high": 1.5,
  "held_out_top_h": 2.0,
}
SUPPORTED_K_FIELD_SHAPES = {
  "(N,1)",
  "(N,3)",
}

STACK_TEMPLATES = {
  "baseline_4_layer": {
    "template_name": "baseline_4_layer",
    "role": "main same-family smoke stack",
    "variant": "baseline_train_family",
    "layer_names": ["substrate_equiv", "active_die_0", "tim_equiv", "heatsink_equiv"],
    "geometry": "regular_layered_rectangular_stack",
    "microstructure": "equivalent_layers_only",
    "smoke_level_abstraction": True,
  },
  "compact_3_layer": {
    "template_name": "compact_3_layer",
    "role": "simplified 3-layer smoke stack",
    "variant": "compact_train_family",
    "layer_names": ["substrate_equiv", "active_die_0", "heatsink_equiv"],
    "geometry": "regular_layered_rectangular_stack",
    "microstructure": "equivalent_layers_only",
    "smoke_level_abstraction": True,
  },
  "dual_active_4_layer": {
    "template_name": "dual_active_4_layer",
    "role": "multi-active-layer smoke stack",
    "variant": "dual_active_train_family",
    "layer_names": ["substrate_equiv", "active_die_0", "active_die_1", "heatsink_equiv"],
    "geometry": "regular_layered_rectangular_stack",
    "microstructure": "equivalent_layers_only",
    "smoke_level_abstraction": True,
  },
  "interposer_like_4_layer": {
    "template_name": "interposer_like_4_layer",
    "role": "train-side interposer-like equivalent stack variation",
    "variant": "train_interposer_like_variation",
    "layer_names": ["substrate_equiv", "interposer_equiv", "active_die_0", "heatsink_equiv"],
    "geometry": "regular_layered_rectangular_stack",
    "microstructure": "equivalent_layers_only",
    "smoke_level_abstraction": True,
    "distinct_from": "heldout_interposer_4_layer",
  },
  "heldout_interposer_4_layer": {
    "template_name": "heldout_interposer_4_layer",
    "role": "held-out stack smoke candidate",
    "variant": "heldout_interposer_like_candidate",
    "layer_names": ["substrate_equiv", "interposer_equiv", "active_die_0", "heatsink_equiv"],
    "geometry": "regular_layered_rectangular_stack",
    "microstructure": "equivalent_layers_only",
    "smoke_level_abstraction": True,
    "distinct_from": "interposer_like_4_layer",
  },
}

HEAT_SOURCE_PATTERNS = {
  "single_centered_active_die_0": {
    "pattern_name": "single_centered_active_die_0",
    "role": "centered single block source",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.50, 0.50],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
  },
  "single_left_shifted_active_die_0": {
    "pattern_name": "single_left_shifted_active_die_0",
    "role": "left-shifted single block source",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.35, 0.50],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
  },
  "single_right_shifted_active_die_0": {
    "pattern_name": "single_right_shifted_active_die_0",
    "role": "right-shifted single block source",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.65, 0.50],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
  },
  "single_offset_active_die_0": {
    "pattern_name": "single_offset_active_die_0",
    "role": "off-center single block source",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.35, 0.65],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
  },
  "two_spots_active_die_0": {
    "pattern_name": "two_spots_active_die_0",
    "role": "two block sources in one active layer",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.35, 0.50],
        "size_xy_fraction": [0.20, 0.20],
      },
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.65, 0.50],
        "size_xy_fraction": [0.20, 0.20],
      },
    ],
    "smoke_level_block_source": True,
  },
  "heldout_source_location": {
    "pattern_name": "heldout_source_location",
    "role": "held-out source-location smoke candidate",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.75, 0.75],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
    "not_ood_evidence": True,
  },
  "dual_active_layers": {
    "pattern_name": "dual_active_layers",
    "role": "one block source in each active layer",
    "source_blocks": [
      {
        "layer": "active_die_0",
        "center_xy_fraction": [0.50, 0.50],
        "size_xy_fraction": [0.25, 0.25],
      },
      {
        "layer": "active_die_1",
        "center_xy_fraction": [0.50, 0.50],
        "size_xy_fraction": [0.25, 0.25],
      },
    ],
    "smoke_level_block_source": True,
  },
}


def load_manifest(path: str | Path) -> dict[str, Any]:
  """Loads a supervised-small manifest JSON file."""

  with Path(path).open("r", encoding="utf-8") as f:
    return json.load(f)


def _parameter_entry(manifest: dict[str, Any], map_name: str, category: str) -> dict[str, Any]:
  parameter_maps = manifest.get("parameter_maps", {})
  if not isinstance(parameter_maps, dict):
    raise ValueError("manifest.parameter_maps must be a dictionary")
  category_map = parameter_maps.get(map_name, {})
  if not isinstance(category_map, dict):
    raise ValueError(f"manifest.parameter_maps.{map_name} must be a dictionary")
  entry = category_map.get(category)
  if not isinstance(entry, dict):
    raise ValueError(f"Unknown {map_name} category: {category!r}")
  return entry


def resolve_bc_baseline(category: str, manifest: dict[str, Any]) -> dict[str, Any]:
  """Resolves a BC baseline category from manifest parameter maps."""

  entry = _parameter_entry(manifest, "bc_baseline", category)
  bottom = entry.get("bottom_fixed_temperature_K")
  top = entry.get("top_ambient_temperature_K")
  if bottom is None or top is None:
    raise ValueError(f"bc_baseline {category!r} requires bottom and top temperatures")
  return {
    "category": category,
    "bottom_fixed_temperature_K": float(bottom),
    "top_ambient_temperature_K": float(top),
    "parameter_status": entry.get("status", "requires_user_confirmation"),
    "source": "manifest.parameter_maps.bc_baseline",
  }


def resolve_q_scale(category: str, manifest: dict[str, Any]) -> dict[str, Any]:
  """Resolves a q-scale category using relative smoke multipliers."""

  entry = _parameter_entry(manifest, "q_scale_category", category)
  multiplier = entry.get("multiplier_to_current_smoke_nominal")
  if multiplier is None:
    multiplier = Q_SCALE_MULTIPLIERS.get(category)
  if multiplier is None:
    raise ValueError(f"Unknown q_scale_category: {category!r}")
  absolute_value = entry.get("volumetric_heat_generation_W_m3")
  if absolute_value is None:
    absolute_value = NOMINAL_Q_W_M3 * float(multiplier)
    absolute_source = NOMINAL_Q_SOURCE
    absolute_resolution = "resolved_from_current_smoke_nominal_for_dry_run_only"
  else:
    absolute_value = float(absolute_value)
    absolute_source = "manifest.parameter_maps.q_scale_category"
    absolute_resolution = "resolved_from_manifest_absolute_value"
  return {
    "category": category,
    "multiplier_to_current_smoke_nominal": float(multiplier),
    "current_nominal_value": NOMINAL_Q_W_M3,
    "resolved_value_W_m3": absolute_value,
    "resolved_value_source": absolute_source,
    "resolution_mode": absolute_resolution,
    "parameter_status": entry.get("status", "requires_user_confirmation"),
    "literature_backed": False,
    "writes_manifest": False,
  }


def resolve_top_h(category: str, manifest: dict[str, Any]) -> dict[str, Any]:
  """Resolves a top-HTC category using relative smoke multipliers."""

  entry = _parameter_entry(manifest, "top_h_category", category)
  multiplier = entry.get("multiplier_to_current_smoke_nominal")
  if multiplier is None:
    multiplier = TOP_H_MULTIPLIERS.get(category)
  if multiplier is None:
    raise ValueError(f"Unknown top_h_category: {category!r}")
  absolute_value = entry.get("h_W_m2K")
  if absolute_value is None:
    absolute_value = NOMINAL_TOP_H_W_M2K * float(multiplier)
    absolute_source = NOMINAL_TOP_H_SOURCE
    absolute_resolution = "resolved_from_current_smoke_nominal_for_dry_run_only"
  else:
    absolute_value = float(absolute_value)
    absolute_source = "manifest.parameter_maps.top_h_category"
    absolute_resolution = "resolved_from_manifest_absolute_value"
  return {
    "category": category,
    "multiplier_to_current_smoke_nominal": float(multiplier),
    "current_nominal_value": NOMINAL_TOP_H_W_M2K,
    "resolved_value_W_m2K": absolute_value,
    "resolved_value_source": absolute_source,
    "resolution_mode": absolute_resolution,
    "parameter_status": entry.get("status", "requires_user_confirmation"),
    "literature_backed": False,
    "writes_manifest": False,
  }


def resolve_stack_template(template_name: str) -> dict[str, Any]:
  """Resolves a stack template name into a smoke-level stack descriptor."""

  if template_name not in STACK_TEMPLATES:
    raise ValueError(f"Unsupported stack_template: {template_name!r}")
  return deepcopy(STACK_TEMPLATES[template_name])


def resolve_heat_source_pattern(pattern_name: str) -> dict[str, Any]:
  """Resolves a heat-source pattern into normalized source-block descriptors."""

  if pattern_name not in HEAT_SOURCE_PATTERNS:
    raise ValueError(f"Unsupported heat_source_pattern: {pattern_name!r}")
  return deepcopy(HEAT_SOURCE_PATTERNS[pattern_name])


def resolve_sample(sample: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
  """Resolves one manifest sample into a no-write generator plan."""

  k_field_shape = sample.get("k_field_shape")
  if k_field_shape not in SUPPORTED_K_FIELD_SHAPES:
    raise ValueError(f"Unsupported k_field_shape: {k_field_shape!r}")

  stack = resolve_stack_template(str(sample.get("stack_template")))
  source = resolve_heat_source_pattern(str(sample.get("heat_source_pattern")))
  q_scale = resolve_q_scale(str(sample.get("q_scale_category")), manifest)
  top_h = resolve_top_h(str(sample.get("top_h_category")), manifest)
  bc_baseline = resolve_bc_baseline(str(sample.get("bc_baseline_category")), manifest)
  return {
    "sample_id": sample.get("sample_id"),
    "split": sample.get("split"),
    "seed": sample.get("seed"),
    "stack_template": stack,
    "heat_source_pattern": source,
    "q_scale": q_scale,
    "top_h": top_h,
    "bc_baseline": bc_baseline,
    "k_field_shape": k_field_shape,
    "anisotropy_type": sample.get("anisotropy_type"),
    "parameter_status": sample.get("parameter_status"),
    "ood_role": sample.get("ood_role"),
  }


def resolve_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
  """Resolves every sample in a manifest for no-write dry-run checks."""

  resolved_samples = []
  errors = []
  for sample in manifest.get("samples", []):
    try:
      resolved_samples.append(resolve_sample(sample, manifest))
    except Exception as exc:
      errors.append({
        "sample_id": sample.get("sample_id"),
        "error": str(exc),
      })
  return {
    "resolved_samples": resolved_samples,
    "errors": errors,
    "supported_stack_templates": sorted(STACK_TEMPLATES),
    "supported_heat_source_patterns": sorted(HEAT_SOURCE_PATTERNS),
    "supported_k_field_shapes": sorted(SUPPORTED_K_FIELD_SHAPES),
    "nominal_q_W_m3": NOMINAL_Q_W_M3,
    "nominal_q_source": NOMINAL_Q_SOURCE,
    "nominal_top_h_W_m2K": NOMINAL_TOP_H_W_M2K,
    "nominal_top_h_source": NOMINAL_TOP_H_SOURCE,
  }
