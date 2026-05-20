#!/usr/bin/env python3
"""Smoke-check the planned Heat3D v1 medium1024 manifest.

This test only validates the dry-run manifest schema. It does not touch data or
output directories and does not generate samples.
"""

from __future__ import annotations

from check_heat3d_v1_physics_label_medium1024_manifest import (
    DEFAULT_MANIFEST,
    EXPECTED_TOTAL_COUNT,
    _read_json,
    validate_manifest,
)


def main() -> int:
    manifest = _read_json(DEFAULT_MANIFEST)
    errors, summary = validate_manifest(manifest)
    if errors:
        raise AssertionError(f"medium1024 manifest smoke failed: {errors}")

    if summary["split_total"] != EXPECTED_TOTAL_COUNT:
        raise AssertionError(f"unexpected split_total: {summary['split_total']}")
    if summary["coverage_totals"].get("source_pattern_tag") != EXPECTED_TOTAL_COUNT:
        raise AssertionError("source_pattern_tag coverage does not sum to 1024")
    if summary["coverage_totals"].get("k_field_mode") != EXPECTED_TOTAL_COUNT:
        raise AssertionError("k_field_mode coverage does not sum to 1024")
    if summary["diag3_count"] != 320:
        raise AssertionError(f"unexpected diag3 count: {summary['diag3_count']}")
    if abs(summary["diag3_fraction"] - 0.3125) > 1e-12:
        raise AssertionError(f"unexpected diag3 fraction: {summary['diag3_fraction']}")
    if summary["held_out_bc_count"] != 40:
        raise AssertionError(f"unexpected held-out BC count: {summary['held_out_bc_count']}")
    if summary["held_out_stack_count"] != 40:
        raise AssertionError(f"unexpected held-out stack count: {summary['held_out_stack_count']}")
    if manifest.get("samples") != []:
        raise AssertionError("medium1024 planning manifest must not include generated samples")
    if manifest.get("full_generation_ready") is True:
        raise AssertionError("medium1024 planning manifest must not be marked full-generation ready")

    print("medium1024_manifest_smoke_ok: True")
    print(f"manifest: {DEFAULT_MANIFEST}")
    print(f"split_total: {summary['split_total']}")
    print(f"diag3_fraction: {summary['diag3_fraction']:.6f}")
    print(f"held_out_bc_count: {summary['held_out_bc_count']}")
    print(f"held_out_stack_count: {summary['held_out_stack_count']}")
    print("scope: dry-run manifest only; no generated data required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
