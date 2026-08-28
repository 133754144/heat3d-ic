"""Fail-closed static audit for the V7 stable runtime import boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (
    ROOT / "rigno" / "heat3d_runtime",
    ROOT / "rigno" / "heat3d_training",
)
ENTRYPOINTS = (
    ROOT / "scripts" / "run_heat3d_v7_formal_training.py",
    ROOT / "scripts" / "run_heat3d_v7_formal_p1i_training.py",
)
FORBIDDEN_TERMS = ("smoke", "development")


def main() -> None:
    violations: list[str] = []
    paths = [path for root in PRODUCTION_ROOTS for path in sorted(root.glob("*.py"))]
    paths.extend(ENTRYPOINTS)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        if "sys.path" in source or "runner_module" in source:
            violations.append(f"{path}: runtime mutation or runner bridge marker")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module == "scripts" or module.startswith("scripts."):
                    violations.append(f"{path}: forbidden scripts import {module!r}")
                if any(term in module.lower() for term in FORBIDDEN_TERMS):
                    violations.append(f"{path}: forbidden legacy import {module!r}")
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(
                            f"{path}: private cross-module import {node.module}.{alias.name}"
                        )
        lowered = source.lower()
        if "monkey patch" in lowered or "monkey_patch" in lowered:
            violations.append(f"{path}: monkey patch marker in production source")
    if violations:
        raise SystemExit("\n".join(violations))
    print("V7 production import audit: PASS")


if __name__ == "__main__":
    main()
