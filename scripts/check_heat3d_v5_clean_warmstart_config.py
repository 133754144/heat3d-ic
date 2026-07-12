#!/usr/bin/env python3
"""No-write regression check for the executable V5 short warm-start config."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/heat3d_v5/v5_clean_warmstart_short.yaml"
RUNNER = REPO_ROOT / "scripts/run_heat3d_v5_clean_first.py"
EXPECTED_VARIANTS = {
    "v4_global_film_legacy_target",
    "native_shape_scale",
    "native_shape_scale_global_film",
}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="heat3d_v5_mpl_") as mpl_cache:
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = mpl_cache
        completed = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--config", str(CONFIG), "--dry-run"],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise AssertionError(f"warm-start dry-run failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"warm-start dry-run did not emit one JSON payload: {completed.stdout!r}") from exc
    if payload.get("mode") != "dry_run" or payload.get("training_runs") != 0:
        raise AssertionError("warm-start fixture attempted training")
    if payload.get("epochs") != 12 or payload.get("planned_output_dir", "").endswith("_r2") is False:
        raise AssertionError("warm-start fixture drifted from the bounded r2 run")
    if payload.get("fit_roles") != ["train"] or payload.get("selection_roles") != ["valid_iid"]:
        raise AssertionError("warm-start split isolation drifted")
    if set(payload.get("variants") or ()) != EXPECTED_VARIANTS:
        raise AssertionError("warm-start comparison variants drifted")
    if payload.get("report_only_roles") != [
        "test_iid",
        "hard_train_holdout",
        "hard_challenge_valid",
        "hard_challenge_test",
    ]:
        raise AssertionError("warm-start report-only roles drifted")
    print(
        json.dumps(
            {
                "status": "passed",
                "epochs": payload["epochs"],
                "variants": payload["variants"],
                "training_runs": payload["training_runs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
