#!/usr/bin/env python3
"""Smoke-check Heat3D v2 draft configs without touching training code."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v2_config import (  # noqa: E402
    load_v2_config,
    resolve_baseline_reference,
    summarize_v2_config,
)


CONFIG_PATHS = (
    Path("configs/heat3d_v2/smoke_minimal.yaml"),
    Path("configs/heat3d_v2/medium1024_gapA_controlled.yaml"),
    Path("configs/heat3d_v2/frozen_v1_reference.yaml"),
)


def main() -> int:
    for relative_path in CONFIG_PATHS:
        config_path = REPO_ROOT / relative_path
        config = load_v2_config(config_path)
        summary = summarize_v2_config(config)
        _print_summary(relative_path, summary)

        if summary.get("config_role") == "controlled":
            reference = resolve_baseline_reference(config, base_dir=REPO_ROOT)
            if reference is None:
                raise ValueError(f"{relative_path}: missing baseline reference")
            reference_summary = summarize_v2_config(reference)
            print(
                "  baseline reference: "
                f"{reference_summary.get('baseline_reference_name')} "
                f"best_epoch={reference_summary.get('baseline_reference_best_epoch')}"
            )

    print("Heat3D v2 config smoke passed.")
    return 0


def _print_summary(relative_path: Path, summary: dict[str, Any]) -> None:
    print(f"config: {relative_path}")
    print(f"  role: {summary.get('config_role')}")
    print(f"  dataset: {summary.get('dataset_name')}")
    if summary.get("config_role") == "baseline_reference":
        print(
            "  optimizer: "
            f"{summary.get('optimizer_name')} lr={summary.get('optimizer_lr')}"
        )
        print(f"  loss: {summary.get('loss_mode')}")
        print(
            "  baseline reference: "
            f"{summary.get('baseline_reference_name')} "
            f"best_epoch={summary.get('baseline_reference_best_epoch')}"
        )
        return

    print(
        "  model: "
        f"{summary.get('model_architecture')} "
        f"latent={summary.get('model_node_latent_size')} "
        f"edge={summary.get('model_edge_latent_size')} "
        f"steps={summary.get('model_processor_steps')}"
    )
    print(
        "  optimizer: "
        f"{summary.get('optimizer_name')} lr={summary.get('optimizer_lr')}"
    )
    print(f"  loss: {summary.get('loss_mode')}")
    print(
        "  run: "
        f"{summary.get('run_mode')} epochs={summary.get('run_epochs')}"
    )
    print(f"  export: {summary.get('export_output_dir')}")
    print(f"  diagnostics: {', '.join(summary.get('diagnostics_enabled', []))}")


if __name__ == "__main__":
    raise SystemExit(main())
