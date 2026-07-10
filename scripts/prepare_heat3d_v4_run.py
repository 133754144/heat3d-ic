#!/usr/bin/env python3
"""Prepare Heat3D V4 inherited YAML and dry-run command plans.

Preparation is registry-driven. This script writes mirror/generated files from
the authoritative JSON registry, then calls the independent registry checker
before printing any dry-run command plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_heat3d_v4_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    build_inherited_yaml,
    check_registry,
    load_registry,
    registry_rows,
    resolve_inherited_yaml,
    write_csv_mirror,
    write_generated_yaml,
)
from rigno.heat3d_v2_runner_command import (  # noqa: E402
    build_v2_command_plan,
    summarize_command_plan,
)


def main() -> int:
    args = _parse_args()
    registry_path = _repo_path(args.registry)
    registry = load_registry(registry_path)
    rows = _filter_rows(registry_rows(registry), args.config_id)

    if args.write_csv_mirror:
        mirror_path = _repo_path(registry["csv_mirror_path"])
        write_csv_mirror(mirror_path, registry_rows(registry))
        print(f"wrote CSV audit mirror: {_relative_to_repo(mirror_path)}")

    if args.write_yaml:
        for row in rows:
            path = write_generated_yaml(row)
            print(f"wrote inherited YAML: {_relative_to_repo(path)}")

    # Required gate: prepare must not bypass the independent checker.
    checked_rows = check_registry(registry_path, emit_warnings=True)
    print("registry checker passed")
    rows = _filter_rows(checked_rows, args.config_id)

    if args.dry_run:
        for row in rows:
            inherited = build_inherited_yaml(row)
            generated_path = _repo_path(row["generated_yaml"])
            resolved = resolve_inherited_yaml(inherited, generated_path)
            plan = build_v2_command_plan(
                resolved, python_executable=args.python_executable
            )
            print(f"config_id: {row['config_id']}")
            print(f"generated_yaml: {row['generated_yaml']}")
            print(f"runner_family: {row['runner_family']}")
            print(f"dataset_name: {row['dataset_name']}")
            print(f"subset_path: {row['subset_path']}")
            print(f"manifest_path: {row['manifest_path']}")
            print(f"split_map_path: {row['split_map_path']}")
            print(f"target_mode: {row['target_mode']}")
            print(f"bridge_policy: {row['bridge_policy']}")
            print(f"normalization_profile: {row['normalization_profile']}")
            print(f"input_feature_schema: {row['input_feature_schema']}")
            print(f"coord_policy: {row['coord_policy']}")
            print(f"extent_feature_policy: {row['extent_feature_policy']}")
            print(
                "condition_feature_transform: "
                f"{row['condition_feature_transform']}"
            )
            print(f"node_coordinate_encoding: {row['node_coordinate_encoding']}")
            print(f"node_coordinate_freqs: {row['node_coordinate_freqs']}")
            print(f"decoder_bypass_mode: {row['decoder_bypass_mode']}")
            print(f"decoder_bypass_features: {row['decoder_bypass_features']}")
            print(
                "decoder_bypass_feature_source: "
                f"{row['decoder_bypass_feature_source']}"
            )
            print(f"target_recovery_policy: {row['target_recovery_policy']}")
            print(f"feature_manifest_hash: {row['feature_manifest_hash']}")
            print(f"lr_init: {row['lr_init']}")
            print(f"lr_peak: {row['lr_peak']}")
            print(f"lr_base: {row['lr_base']}")
            print(f"lr_lowr: {row['lr_lowr']}")
            print(f"pct_start: {row['pct_start']}")
            print(f"pct_final: {row['pct_final']}")
            print(f"sample_weight_policy: {row['sample_weight_policy']}")
            print(f"sample_weight_json: {row['sample_weight_json']}")
            print(f"sample_weight_default: {row['sample_weight_default']}")
            print(f"sample_weight_normalize: {row['sample_weight_normalize']}")
            print(f"metrics_profile: {row['metrics_profile']}")
            print(f"metrics_contract: {row['metrics_contract']}")
            print(f"selection_metric: {row['selection_metric']}")
            print(f"prediction_split: {row['prediction_split']}")
            print(summarize_command_plan(plan))

    print(f"prepared registry rows: {len(rows)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--config-id", action="append")
    parser.add_argument("--write-yaml", action="store_true")
    parser.add_argument("--write-csv-mirror", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default="python3")
    return parser.parse_args()


def _filter_rows(
    rows: list[dict[str, str]], config_ids: list[str] | None
) -> list[dict[str, str]]:
    if not config_ids:
        return rows
    wanted = set(config_ids)
    selected = [row for row in rows if row["config_id"] in wanted]
    missing = sorted(wanted - {row["config_id"] for row in selected})
    if missing:
        raise ValueError(f"missing config_id(s): {', '.join(missing)}")
    return selected


def _repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
