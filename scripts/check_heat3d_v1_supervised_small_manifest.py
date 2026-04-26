"""Dry-run validator for the Heat3D v1 supervised-small manifest.

This script checks whether the 16-sample manifest can be resolved into a
future generator-readable plan. It does not generate samples, does not write to
data/, does not run a solver, and does not train a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_manifest_resolver import (  # noqa: E402
  HEAT_SOURCE_PATTERNS,
  STACK_TEMPLATES,
  SUPPORTED_K_FIELD_SHAPES,
  load_manifest,
  resolve_manifest,
)


EXPECTED_DATASET_NAME = "v1_multilayer_bc_eq_supervised_small"
EXPECTED_SCAFFOLD_COMMIT = "1e00a15"
EXPECTED_SPLIT_COUNTS = {
  "train": 10,
  "valid": 3,
  "test_smoke": 1,
  "test_ood_bc": 1,
  "test_ood_stack": 1,
}
EXPECTED_SAMPLE_IDS = [f"sample_{idx:03d}" for idx in range(16)]
REQUIRED_SAMPLE_FIELDS = (
  "sample_id",
  "split",
  "seed",
  "stack_template",
  "k_field_shape",
  "anisotropy_type",
  "heat_source_pattern",
  "q_scale_category",
  "bc_baseline_category",
  "top_h_category",
  "expected_purpose",
  "parameter_status",
  "ood_role",
)
PROTECTED_SUBSET_NAMES = {
  "v1_multilayer_bc_eq_demo",
  "v1_multilayer_bc_eq_supervised_smoke",
}
STRUCTURED_EXCLUDED_CHECK_FIELDS = (
  "k_field_shape",
  "stack_template",
  "heat_source_pattern",
  "anisotropy_type",
)
EXCLUDED_FIRST_STEP_MARKERS = (
  "(N,6)",
  "irregular",
  "explicit_TSV",
  "BEOL",
  "bump",
  "transient",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Validate and dry-run the Heat3D v1 supervised-small manifest."
  )
  parser.add_argument(
    "manifest",
    type=Path,
    help="Path to configs/heat3d_v1_supervised_small_manifest.json.",
  )
  return parser.parse_args()


def _sorted_unique(values: list[str]) -> list[str]:
  return sorted(set(values))


def _check_default_route(route: dict[str, Any]) -> list[str]:
  errors = []
  expected = {
    "condition_feature_view": "relative_bc",
    "bridge": "zero_delta_u_bridge",
    "target": "DeltaT = T - T_ref",
  }
  for key, value in expected.items():
    if route.get(key) != value:
      errors.append(f"default_route.{key} must be {value!r}, found {route.get(key)!r}")
  return errors


def _check_required_sample_fields(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
  missing: dict[str, list[str]] = {}
  for sample in samples:
    sample_id = sample.get("sample_id", "<missing_sample_id>")
    absent = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
    if absent:
      missing[sample_id] = absent
  return missing


def _find_excluded_first_step_features(
  samples: list[dict[str, Any]],
  parameter_maps: dict[str, Any],
) -> list[str]:
  """Checks only structured fields, not free-text notes or descriptions."""

  found = []
  declared_exclusions = parameter_maps.get("excluded_first_step_features", [])
  if not isinstance(declared_exclusions, list):
    declared_exclusions = []

  for sample in samples:
    for field in STRUCTURED_EXCLUDED_CHECK_FIELDS:
      value = str(sample.get(field, ""))
      for marker in EXCLUDED_FIRST_STEP_MARKERS:
        if marker in value:
          found.append(f"{sample.get('sample_id')}.{field}: {marker}")
      for declared in declared_exclusions:
        declared_text = str(declared)
        if declared_text and declared_text == value:
          found.append(f"{sample.get('sample_id')}.{field}: {declared_text}")
  return found


def _output_subset_safety(repo_root: Path, dataset_name: str) -> dict[str, Any]:
  output_path = repo_root / "data" / "heat3d-thermal-simulation" / "subsets" / dataset_name
  under_data = "data" in output_path.relative_to(repo_root).parts
  protected_collision = dataset_name in PROTECTED_SUBSET_NAMES
  return {
    "target_path": str(output_path),
    "under_ignored_data_dir_expected": under_data,
    "protected_subset_collision": protected_collision,
    "protected_subset_names": sorted(PROTECTED_SUBSET_NAMES),
    "safe_target_name": under_data and not protected_collision,
  }


def _status_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
  return dict(Counter(sample.get("parameter_status") for sample in samples))


def _category_summary(
  resolved_samples: list[dict[str, Any]],
  key: str,
  value_key: str,
) -> list[str]:
  seen = {}
  for sample in resolved_samples:
    resolved = sample[key]
    category = resolved["category"]
    seen[category] = resolved
  lines = []
  for category, resolved in sorted(seen.items()):
    lines.append(
      f"{category}: multiplier={resolved['multiplier_to_current_smoke_nominal']} "
      f"resolved_{value_key}={resolved[f'resolved_value_{value_key}']} "
      f"source={resolved['resolved_value_source']} "
      f"status={resolved['parameter_status']} "
      f"writes_manifest={resolved['writes_manifest']}"
    )
  return lines


def main() -> int:
  args = parse_args()
  manifest_path = args.manifest.resolve()
  manifest = load_manifest(manifest_path)
  samples = manifest.get("samples", [])
  if not isinstance(samples, list):
    raise ValueError("manifest.samples must be a list")

  dataset_name = manifest.get("dataset_name")
  sample_ids = [sample.get("sample_id") for sample in samples]
  split_counts = dict(Counter(sample.get("split") for sample in samples))
  parameter_maps = manifest.get("parameter_maps", {})
  if not isinstance(parameter_maps, dict):
    parameter_maps = {}

  manifest_contract_errors = []
  if dataset_name != EXPECTED_DATASET_NAME:
    manifest_contract_errors.append(
      f"dataset_name must be {EXPECTED_DATASET_NAME!r}, found {dataset_name!r}"
    )
  if manifest.get("scaffold_base_commit") != EXPECTED_SCAFFOLD_COMMIT:
    manifest_contract_errors.append(
      "scaffold_base_commit must be "
      f"{EXPECTED_SCAFFOLD_COMMIT!r}, found {manifest.get('scaffold_base_commit')!r}"
    )
  manifest_contract_errors.extend(_check_default_route(manifest.get("default_route", {})))
  if split_counts != EXPECTED_SPLIT_COUNTS:
    manifest_contract_errors.append(
      f"split counts must be {EXPECTED_SPLIT_COUNTS}, found {split_counts}"
    )
  if sample_ids != EXPECTED_SAMPLE_IDS:
    manifest_contract_errors.append("sample ids must be sample_000 through sample_015 in order")

  missing_fields = _check_required_sample_fields(samples)
  if missing_fields:
    manifest_contract_errors.append(f"samples missing required fields: {missing_fields}")

  excluded_features = _find_excluded_first_step_features(samples, parameter_maps)
  resolved = resolve_manifest(manifest)
  resolved_samples = resolved["resolved_samples"]
  resolver_errors = resolved["errors"]

  stack_values = [sample.get("stack_template", "") for sample in samples]
  heat_values = [sample.get("heat_source_pattern", "") for sample in samples]
  k_shape_values = [sample.get("k_field_shape", "") for sample in samples]
  supported_stacks = [value for value in _sorted_unique(stack_values) if value in STACK_TEMPLATES]
  unsupported_stacks = [value for value in _sorted_unique(stack_values) if value not in STACK_TEMPLATES]
  supported_heat = [value for value in _sorted_unique(heat_values) if value in HEAT_SOURCE_PATTERNS]
  unsupported_heat = [value for value in _sorted_unique(heat_values) if value not in HEAT_SOURCE_PATTERNS]
  supported_k_shapes = [value for value in _sorted_unique(k_shape_values) if value in SUPPORTED_K_FIELD_SHAPES]
  unsupported_k_shapes = [value for value in _sorted_unique(k_shape_values) if value not in SUPPORTED_K_FIELD_SHAPES]

  output_safety = _output_subset_safety(REPO_DIR, str(dataset_name))

  manifest_contract_ok = not manifest_contract_errors
  excluded_feature_check_ok = not excluded_features
  resolver_plan_ok = not resolver_errors
  generator_support_ready = not (unsupported_stacks or unsupported_heat or unsupported_k_shapes or resolver_errors)
  numeric_resolution_ready = resolver_plan_ok and len(resolved_samples) == len(samples)
  output_subset_safe = output_safety["safe_target_name"]
  safe_execution_ready = (
    manifest_contract_ok
    and excluded_feature_check_ok
    and resolver_plan_ok
    and generator_support_ready
    and numeric_resolution_ready
    and output_subset_safe
  )
  data_generation_ready = False

  direct_user_confirmation_samples = [
    sample["sample_id"]
    for sample in samples
    if sample.get("parameter_status") == "requires_user_confirmation"
  ]
  category_user_confirmation_samples = [
    sample["sample_id"]
    for sample in samples
    if sample.get("q_scale_category") in {"low", "nominal", "high"}
    or sample.get("top_h_category") in {"low", "nominal", "high", "held_out_top_h"}
  ]

  print("Heat3D v1 supervised-small manifest dry-run")
  print(f"manifest path: {manifest_path}")
  print(f"dataset_name: {dataset_name}")
  print(f"sample_count: {len(samples)}")
  print(f"split_counts: {split_counts}")
  print(f"sample_ids: {sample_ids}")
  print()

  print("default route")
  route = manifest.get("default_route", {})
  print(f"  condition_feature_view: {route.get('condition_feature_view')}")
  print(f"  bridge: {route.get('bridge')}")
  print(f"  target: {route.get('target')}")
  print()

  print("stack_template resolver support")
  print(f"  supported: {supported_stacks}")
  print(f"  unsupported: {unsupported_stacks}")
  for name in ("interposer_like_4_layer", "heldout_interposer_4_layer"):
    if name in STACK_TEMPLATES:
      item = STACK_TEMPLATES[name]
      print(f"  {name}: role={item['role']}, variant={item['variant']}")
  print()

  print("heat_source_pattern resolver support")
  print(f"  supported: {supported_heat}")
  print(f"  unsupported: {unsupported_heat}")
  print("  source patterns are smoke-level rectangular source blocks, not industrial power maps")
  print()

  print("k_field_shape support")
  print(f"  supported: {supported_k_shapes}")
  print(f"  unsupported: {unsupported_k_shapes}")
  print("  (N,6) remains schema-reserved and is not generated in this phase")
  print()

  print("q_scale resolver summary")
  for line in _category_summary(resolved_samples, "q_scale", "W_m3"):
    print(f"  {line}")
  print()

  print("top_h resolver summary")
  for line in _category_summary(resolved_samples, "top_h", "W_m2K"):
    print(f"  {line}")
  print()

  print("bc_baseline resolver summary")
  seen_bc = {}
  for sample in resolved_samples:
    baseline = sample["bc_baseline"]
    seen_bc[baseline["category"]] = baseline
  for category, baseline in sorted(seen_bc.items()):
    print(
      f"  {category}: bottom={baseline['bottom_fixed_temperature_K']} K, "
      f"top={baseline['top_ambient_temperature_K']} K, "
      f"status={baseline['parameter_status']}"
    )
  print()

  print("sample-level resolved plan summary")
  for sample in resolved_samples:
    print(
      f"  {sample['sample_id']}: split={sample['split']}, "
      f"stack={sample['stack_template']['template_name']}[{sample['stack_template']['variant']}], "
      f"source={sample['heat_source_pattern']['pattern_name']}, "
      f"q={sample['q_scale']['category']}({sample['q_scale']['multiplier_to_current_smoke_nominal']}x), "
      f"top_h={sample['top_h']['category']}({sample['top_h']['multiplier_to_current_smoke_nominal']}x), "
      f"bc={sample['bc_baseline']['category']}, k={sample['k_field_shape']}"
    )
  print()

  print("parameter_status summary")
  print(f"  {_status_counts(samples)}")
  print(f"  direct_samples_requiring_user_confirmation: {direct_user_confirmation_samples}")
  print(
    "  samples_using_relative_q_or_top_h_categories: "
    f"{sorted(set(category_user_confirmation_samples))}"
  )
  print()

  print("output subset safety check")
  print(f"  target_path: {output_safety['target_path']}")
  print(f"  under_ignored_data_dir_expected: {output_safety['under_ignored_data_dir_expected']}")
  print(f"  protected_subset_collision: {output_safety['protected_subset_collision']}")
  print(f"  protected_subset_names: {output_safety['protected_subset_names']}")
  print(f"  safe_target_name: {output_safety['safe_target_name']}")
  print(f"  data_generation_ready: {data_generation_ready}")
  print()

  print("manifest_contract_errors")
  if manifest_contract_errors:
    for error in manifest_contract_errors:
      print(f"  {error}")
  else:
    print("  none")
  print()

  print("excluded_feature_violations")
  if excluded_features:
    for item in excluded_features:
      print(f"  {item}")
  else:
    print("  none")
  print()

  print("resolver_errors")
  if resolver_errors:
    for error in resolver_errors:
      print(f"  {error}")
  else:
    print("  none")
  print()

  print(f"manifest_contract_ok: {manifest_contract_ok}")
  print(f"excluded_feature_check_ok: {excluded_feature_check_ok}")
  print(f"resolver_plan_ok: {resolver_plan_ok}")
  print(f"generator_support_ready: {generator_support_ready}")
  print(f"numeric_resolution_ready: {numeric_resolution_ready}")
  print(f"safe_execution_ready: {safe_execution_ready}")
  print(f"data_generation_ready: {data_generation_ready}")
  print(f"dry_run_pass: {safe_execution_ready}")

  return 0 if safe_execution_ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
