#!/usr/bin/env python3
"""Run the byte-frozen Gate-5 evaluator for the two Gate-6C scratch runs.

This adapter changes only the accepted config IDs and the recorded source
registry commit.  Metric, checkpoint, split, normalization, context, and
reporting code remains the file committed at ``FROZEN_EVALUATOR_COMMIT``.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys


FROZEN_EVALUATOR_COMMIT = "639872abcb0f7afd3b6c2d319a7d395bde75c9a4"
FROZEN_EVALUATOR_SHA256 = "aed63bbfa0e23aa69b944960f222feac05dc3682783ab601a3e90ae54581911d"
GATE6C_CONFIGS = {
    "V4P5_11_gate6c_scratch_l1_tail_balanced",
    "V4P5_12_gate6c_scratch_l2_shape_balanced",
}
N3_CONFIG = "V4P5_07_native_pooled_latent_global_film"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evaluator_path = root / "scripts/evaluate_heat3d_v5_gate5_checkpoints.py"
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != FROZEN_EVALUATOR_COMMIT:
        raise SystemExit(
            f"Gate 6D evaluator must run at {FROZEN_EVALUATOR_COMMIT}, got {head}"
        )
    source_hash = _sha256(evaluator_path)
    if source_hash != FROZEN_EVALUATOR_SHA256:
        raise SystemExit(
            "frozen evaluator source differs from the committed Gate-5 engine: "
            f"{source_hash}"
        )
    registry_commit = os.environ.get("GATE6D_REGISTRY_COMMIT", "").strip()
    if not registry_commit:
        raise SystemExit("GATE6D_REGISTRY_COMMIT is required")
    registry_commit = subprocess.check_output(
        ["git", "rev-parse", registry_commit], cwd=root, text=True
    ).strip()

    csv.field_size_limit(sys.maxsize)
    import evaluate_heat3d_v5_gate5_checkpoints as evaluator

    valid_only = os.environ.get("GATE6D_VALID_ONLY", "0") == "1"
    if valid_only:
        evaluator.ALLOWED_CONFIGS = {N3_CONFIG}
        evaluator.ROLES = ("valid_iid",)
    else:
        evaluator.ALLOWED_CONFIGS = set(GATE6C_CONFIGS)
    evaluator._path_commit = lambda _path: registry_commit
    return int(evaluator.main())


if __name__ == "__main__":
    raise SystemExit(main())
