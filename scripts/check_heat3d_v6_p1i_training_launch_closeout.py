#!/usr/bin/env python3
"""Validate the post-launch P1i governance record without reading targets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = json.loads((ROOT / "configs/heat3d_v6_p1i/v6_p1i_training_launch_closeout.json").read_text())
    with (ROOT / "configs/heat3d_v6_p1i/v6_p1i_training_registry.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["config_id"] == payload["config_id"])
    checks = {
        "running_e600": payload["status"] == row["execution_status"] == "running_e600",
        "training_commit_bound": row["training_commit"] == payload["training_commit"] == "93ea04a52b5cfcc1a9e9af027bcd6747151737ae",
        "pid_bound": int(row["pid"]) == int(payload["launch"]["pid"]) == 71870,
        "full_field_frozen": payload["full_field"]["archive_sha256"] == "49023ac1205b8e7cf7c5bf782b89fcdb34997704b3f9aa2fb2d46cf1a59163cb",
        "smokes_passed": payload["smoke"]["B8"]["status"] == payload["smoke"]["B16"]["status"] == "passed",
        "test_closed": row["test_access"] == "closed_audited_holdout" == payload["launch"]["test_role"],
        "sealed_closed": row["sealed_access"] == "closed_confirmatory" == payload["launch"]["sealed_iid_role"],
        "randomblock_not_for_tuning": payload["randomblock_one_time_transfer"]["used_for_tuning"] is False,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
