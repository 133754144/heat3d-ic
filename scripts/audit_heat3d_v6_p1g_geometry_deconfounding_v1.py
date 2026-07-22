#!/usr/bin/env python3
"""Run the common P1g audit against the immutable P1g-v1 artifacts."""

from __future__ import annotations

from pathlib import Path

import audit_heat3d_v6_p1g_geometry_deconfounding as audit


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs/heat3d_v6"


def main() -> int:
    audit.STEM = "v6_p1g_geometry_deconfounded1024_v1"
    audit.CONFIG = CONFIG_DIR / f"{audit.STEM}.yaml"
    audit.DATASET = ROOT / "data/heat3d_v6_p1g_geometry_deconfounded1024_v1"
    audit.AUDIT_JSON = CONFIG_DIR / f"{audit.STEM}_geometry_audit.json"
    audit.QUALIFICATION_JSON = CONFIG_DIR / f"{audit.STEM}_qualification.json"
    audit.CONTINGENCY_CSV = CONFIG_DIR / f"{audit.STEM}_joint_contingency.csv"
    audit.PROJECTION_CSV = CONFIG_DIR / f"{audit.STEM}_projection_diagnostics.csv"
    audit.LAYER_CSV = CONFIG_DIR / f"{audit.STEM}_layer_projection_errors.csv"
    audit.REPORT = ROOT / "docs/v6_p1g_geometry_deconfounding_v1_audit.md"
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
