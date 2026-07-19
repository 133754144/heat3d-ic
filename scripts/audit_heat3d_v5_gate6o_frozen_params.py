#!/usr/bin/env python3
"""Verify that Gate 6O Stage 2 changed only global scale MLP leaves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_heat3d_v1_medium_controlled_training_export import (  # noqa: E402
    _load_params_checkpoint,
)


CHECKPOINTS = (
    "params_best_valid_point_global.pkl",
    "params_best_valid_sample_first.pkl",
    "params_best_valid_base_mse.pkl",
    "params_final.pkl",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _path_text(path: Any) -> str:
    return "/".join(
        str(getattr(item, "key", getattr(item, "name", item)))
        for item in path
    )


def _is_trainable(path: Any) -> bool:
    return any(
        str(getattr(item, "key", getattr(item, "name", item))).startswith(
            "global_scale_"
        )
        for item in path
    )


def main() -> int:
    args = _args()
    initial = _load_params_checkpoint(args.init_checkpoint)["params"]
    initial_leaves = jax.tree_util.tree_flatten_with_path(initial)[0]
    reports = {}
    for filename in CHECKPOINTS:
        checkpoint_path = args.run_dir / filename
        payload = _load_params_checkpoint(checkpoint_path)
        leaves = jax.tree_util.tree_flatten_with_path(payload["params"])[0]
        if len(leaves) != len(initial_leaves):
            raise ValueError(f"{filename}: parameter leaf count changed")
        frozen_max = 0.0
        trainable_max = 0.0
        trainable_changed = 0
        trainable_names = []
        for (left_path, left), (right_path, right) in zip(
            initial_leaves, leaves, strict=True
        ):
            if _path_text(left_path) != _path_text(right_path):
                raise ValueError(f"{filename}: parameter path order changed")
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(left, dtype=np.float64)
                        - np.asarray(right, dtype=np.float64)
                    )
                )
            )
            if _is_trainable(left_path):
                trainable_names.append(_path_text(left_path))
                trainable_max = max(trainable_max, difference)
                trainable_changed += int(difference > 0.0)
            else:
                frozen_max = max(frozen_max, difference)
        report = {
            "checkpoint": filename,
            "epoch": int(payload["epoch"]),
            "parameter_leaf_count": len(leaves),
            "trainable_leaf_count": len(trainable_names),
            "trainable_leaf_names": sorted(trainable_names),
            "trainable_changed_leaf_count": trainable_changed,
            "trainable_max_abs_difference": trainable_max,
            "frozen_max_abs_difference": frozen_max,
            "passed": bool(
                frozen_max == 0.0
                and trainable_changed > 0
                and set(trainable_names)
                == {
                    "global_scale_hidden/bias",
                    "global_scale_hidden/kernel",
                    "global_scale_output/bias",
                    "global_scale_output/kernel",
                }
            ),
        }
        if not report["passed"]:
            raise ValueError(f"{filename}: frozen-parameter audit failed: {report}")
        reports[filename] = report
    result = {
        "schema_version": "heat3d_v5_gate6o_frozen_parameter_audit_v1",
        "status": "passed",
        "trainable_scope": "global_scale_mlp_only",
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "output": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
