#!/usr/bin/env python3
"""Regression fixture for V5 plan-to-runner adaptation (no training)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v5_config import LOCAL_BYPASS_FEATURES, build_v5_runner_plan  # noqa: E402
from rigno.models.rigno import RIGNO  # noqa: E402


CONFIG = REPO_ROOT / "configs/heat3d_v5/v5_clean_first_bypass_ablation_plan.yaml"
EXPECTED = {
    "frozen_v4_full_condition_reference": ("full_condition", "none"),
    "local_only_bypass_control": ("explicit_local_condition", "none"),
    "bypass_disabled_control": ("none", "none"),
    "local_only_bypass_plus_global_film_smoke": ("explicit_local_condition", "film"),
}


def _option(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def main() -> int:
    # The V5 plan deliberately preserves audited source names through command
    # resolution for provenance.  Verify that the actual model accepts that
    # metadata too; a runner --dry-run alone cannot catch constructor kwargs.
    local_metadata_probe = RIGNO(
        num_outputs=1,
        decoder_bypass_local_feature_names=LOCAL_BYPASS_FEATURES,
    )
    if tuple(local_metadata_probe.decoder_bypass_local_feature_names) != LOCAL_BYPASS_FEATURES:
        raise AssertionError("RIGNO did not retain local bypass provenance metadata")

    plans = {}
    for variant, (expected_bypass, expected_context) in EXPECTED.items():
        plan = build_v5_runner_plan(CONFIG, variant=variant, python_executable="python3")
        command = plan["command"]
        if plan["training_allowed"]:
            raise AssertionError(f"{variant}: prepare-only plan unexpectedly allows training")
        if _option(command, "--decoder-bypass-features") != expected_bypass:
            raise AssertionError(f"{variant}: decoder bypass flag did not map")
        if _option(command, "--global-context-mode") != expected_context:
            raise AssertionError(f"{variant}: global context flag did not map")
        if expected_bypass == "explicit_local_condition":
            if tuple(_option(command, "--decoder-bypass-local-feature-names").split(",")) != LOCAL_BYPASS_FEATURES:
                raise AssertionError(f"{variant}: local bypass schema drift")
        if expected_context == "film":
            names = _option(command, "--global-context-feature-names").split(",")
            if len(names) != 24 or names[0] != "log_s_phys_K":
                raise AssertionError(f"{variant}: Global FiLM schema did not map")
            if "--no-final-probe-eval-after-training" not in command:
                raise AssertionError(f"{variant}: unsupported V4 final probe was not disabled")
        runner_command = [sys.executable, *command[1:], "--dry-run"]
        completed = subprocess.run(
            runner_command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"{variant}: mapped runner command rejected options: {completed.stderr.strip()}"
            )
        json_lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
        if not json_lines:
            raise AssertionError(f"{variant}: mapped runner dry-run emitted no JSON payload")
        runner_payload = json.loads(json_lines[-1])
        if runner_payload.get("mode") != "dry_run" or runner_payload.get("training_runs") != 0:
            raise AssertionError(f"{variant}: runner dry-run did not remain non-training")
        plans[variant] = {
            "decoder_bypass_features": expected_bypass,
            "global_context_mode": expected_context,
        }
    print(
        json.dumps(
            {
                "status": "passed",
                "local_bypass_model_metadata_accepted": True,
                "variants": plans,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
