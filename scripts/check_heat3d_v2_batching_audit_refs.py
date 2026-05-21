#!/usr/bin/env python3
"""Print local source references relevant to Heat3D v2 batching audit.

This script is intentionally read-only: it does not import JAX, read datasets,
run training, or create outputs.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    Path("scripts/run_heat3d_v1_medium_controlled_training_export.py"),
    Path("scripts/check_heat3d_v1_small_train_valid_smoke.py"),
    Path("rigno/heat3d_v2_config.py"),
    Path("rigno/heat3d_v2_runner_command.py"),
    Path("scripts/train_heat3d_operator.py"),
    Path("rigno/heat3d_pipeline.py"),
]

KEYWORDS = (
    "batch_size",
    "full-batch",
    "full_batch",
    "train_groups",
    "valid_groups",
    "all_groups",
    "_make_batch_group",
    "_loss_components",
    "_global_norm",
    "value_and_grad",
    "gradient_clip_norm",
    "save_best_predictions",
    "iterate_batch_indices",
)


def main() -> int:
    print("Heat3D v2 local batching audit references")
    for relative_path in TARGETS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            print(f"\n{relative_path}: not found")
            continue

        print(f"\n{relative_path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = []
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(keyword.lower() in lowered for keyword in KEYWORDS):
                matches.append((line_number, line.strip()))

        if not matches:
            print("  no keyword matches")
            continue

        for line_number, line in matches[:80]:
            print(f"  {line_number}: {line}")
        if len(matches) > 80:
            print(f"  ... {len(matches) - 80} additional matches omitted")

    print("\nBatching audit reference scan complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
