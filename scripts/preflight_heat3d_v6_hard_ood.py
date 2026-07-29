#!/usr/bin/env python3
"""Run the label-free V6 hard/OOD preflight after preregistration is pushed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/heat3d_v6"
PREREG = CONFIG / "v6_hard_ood_preregistration.json"
ROLE = CONFIG / "v6_hard_input_stress_role.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    role = json.loads(ROLE.read_text(encoding="utf-8"))
    committed = subprocess.run(
        [
            "git",
            "show",
            f"{args.preregistration_commit}:"
            "configs/heat3d_v6/v6_hard_ood_preregistration.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    committed_prereg_sha256 = hashlib.sha256(committed).hexdigest()
    if committed_prereg_sha256 != _sha256(PREREG):
        raise RuntimeError("working preregistration differs from pushed gate")
    current_evaluator = ROOT / prereg["workflow"]["evaluator_path"]
    current_evaluator_sha256 = _sha256(current_evaluator)
    adapter_path = CONFIG / "v6_hard_ood_evaluator_adapter.json"
    adapter_binding = None
    if current_evaluator_sha256 != prereg["workflow"]["evaluator_sha256"]:
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        if (
            adapter["base_evaluator"]["sha256"]
            != prereg["workflow"]["evaluator_sha256"]
            or adapter["new_evaluator"]["sha256"]
            != current_evaluator_sha256
            or adapter["metrics_formulas_changed"] is not False
        ):
            raise RuntimeError("hard evaluator adapter binding drifted")
        adapter_binding = {
            "path": "configs/heat3d_v6/v6_hard_ood_evaluator_adapter.json",
            "sha256": _sha256(adapter_path),
            "scope": "prediction_count_only",
        }

    checkpoint = args.run_dir / "params_best_valid_point_global.pkl"
    if _sha256(checkpoint) != prereg["checkpoint"]["sha256"]:
        raise RuntimeError("reference checkpoint hash drifted")
    manifest = json.loads(
        (ROOT / prereg["dataset"]["manifest_path"]).read_text(encoding="utf-8")
    )
    manifest_rows = {
        str(row["sample_id"]): row for row in manifest["samples"]
    }
    for sample_id in role["sample_ids"]:
        row = manifest_rows[sample_id]
        if row["split_role"] != "test":
            raise RuntimeError("hard stress role escaped the test holdout")
        sample_dir = args.dataset / row["sample_dir"]
        for input_name in (
            "coords.npy",
            "k_field.npy",
            "q_field.npy",
            "bc_features.npy",
            "bc_parameters.npy",
        ):
            if not (sample_dir / input_name).is_file():
                raise RuntimeError(f"missing label-free input {sample_id}/{input_name}")

    commands = prereg["command_plan"]
    command_audit = []
    for resolution, command in zip((4096, 8192, 16384), commands):
        passed = all(
            (
                f"--resolution {resolution}" in command,
                "--role hard_input_stress" in command,
                "--role-manifest configs/heat3d_v6/v6_hard_input_stress_role.json"
                in command,
                "v6_source_aware_resolution_ladder.json" in command,
                "32768" not in command,
                "run_heat3d_v4_config.py" not in command,
                "controlled_training" not in command,
            )
        )
        if not passed:
            raise RuntimeError(f"command audit failed for {resolution}")
        command_audit.append(
            {
                "resolution": resolution,
                "passed": True,
                "training_command": False,
                "formal_execution_performed": False,
            }
        )

    payload = {
        "schema_version": "heat3d_v6_hard_ood_preflight_v2",
        "status": "passed",
        "preregistration_commit": args.preregistration_commit,
        "preregistration_sha256": _sha256(PREREG),
        "role_manifest_sha256": _sha256(ROLE),
        "evaluator_sha256": current_evaluator_sha256,
        "evaluator_adapter": adapter_binding,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_modified": False,
        "dataset_id": prereg["dataset"]["dataset_id"],
        "dataset_manifest_sha256": prereg["dataset"]["manifest_sha256"],
        "hard_input_stress_sample_count": len(role["sample_ids"]),
        "role_input_files_checked_per_sample": 5,
        "temperature_deltaT_or_full_field_labels_opened": False,
        "canonical_ood_status": "not_available",
        "ood_labels_opened": False,
        "command_audit": command_audit,
        "training_executed": False,
        "model_checkpoint_sampling_graph_reconstruction_modified": False,
        "local_absolute_paths_persisted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "hard_samples": len(role["sample_ids"]),
                "label_access": False,
                "commands": len(command_audit),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
