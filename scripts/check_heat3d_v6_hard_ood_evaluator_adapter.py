#!/usr/bin/env python3
"""Validate the population-count-only hard-stress evaluator adapter."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT / "configs/heat3d_v6/v6_hard_ood_evaluator_adapter.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def main() -> int:
    adapter = json.loads(ADAPTER.read_text(encoding="utf-8"))
    base_commit = adapter["base_evaluator"]["preregistration_commit"]
    current_evaluator_path = ROOT / adapter["new_evaluator"]["path"]
    current_helper_path = ROOT / adapter["new_production_helper"]["path"]
    current_evaluator = current_evaluator_path.read_bytes()
    current_helper = current_helper_path.read_bytes()
    base_evaluator = subprocess.run(
        [
            "git",
            "show",
            f"{base_commit}:{adapter['base_evaluator']['path']}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    base_helper = subprocess.run(
        [
            "git",
            "show",
            f"{base_commit}:{adapter['base_production_helper']['path']}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert _sha256_bytes(base_evaluator) == adapter["base_evaluator"]["sha256"]
    assert _sha256_bytes(base_helper) == adapter["base_production_helper"]["sha256"]
    assert _sha256_bytes(current_evaluator) == adapter["new_evaluator"]["sha256"]
    assert _sha256_bytes(current_helper) == adapter["new_production_helper"]["sha256"]
    assert _function_source(
        base_evaluator.decode(), "_evaluate"
    ) == _function_source(current_evaluator.decode(), "_evaluate")
    helper_text = current_helper.decode()
    assert "expected_sample_count: int = 128" in helper_text
    assert "len(predictions) != expected_sample_count" in helper_text
    evaluator_text = current_evaluator.decode()
    assert evaluator_text.count("expected_sample_count=len(") == 2
    assert adapter["metrics_formulas_changed"] is False
    assert adapter["frozen_scientific_behavior_changed"] is False
    assert (
        adapter["model_checkpoint_sampling_graph_reconstruction_changed"]
        is False
    )
    assert adapter["failure"]["hard_labels_read"] is False
    assert adapter["failure"]["formal_result_written"] is False
    print(
        json.dumps(
            {
                "status": "passed",
                "adapter_scope": "prediction_count_only",
                "historical_default": 128,
                "hard_expected_count": 16,
                "metric_function_exact": True,
                "hard_labels_read_before_adapter": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
