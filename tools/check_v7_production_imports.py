"""Fail-closed static audit for the V7 stable runtime import boundary."""

from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "rigno" / "heat3d_runtime"
FORBIDDEN_TERMS = ("smoke", "development")


def main() -> None:
    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.glob("*.py")):
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
    if violations:
        raise SystemExit("\n".join(violations))
    print("V7 production import audit: PASS")


if __name__ == "__main__":
    main()
