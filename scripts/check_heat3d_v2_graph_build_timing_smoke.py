"""Smoke-check Heat3D v2 graph-build timing hooks without running training."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_heat3d_v1_medium_controlled_training_export as runner  # noqa: E402


@contextmanager
def _argv(args: list[str]):
    original = sys.argv[:]
    sys.argv = [str(SCRIPTS_DIR / "run_heat3d_v1_medium_controlled_training_export.py"), *args]
    try:
        yield
    finally:
        sys.argv = original


def main() -> int:
    profile_json = REPO_DIR / "output" / "heat3d_v2_runs" / "timing_smoke" / "profile_timing.json"
    with _argv(["--epochs", "1", "--profile-timing", "--profile-timing-json", str(profile_json)]):
        args = runner.parse_args()

    if not args.profile_timing:
        raise AssertionError("--profile-timing was not parsed")
    if args.profile_timing_json != profile_json:
        raise AssertionError("--profile-timing-json path was not parsed")
    if not runner._profile_timing_enabled(args):
        raise AssertionError("profile timing should be enabled")

    payload = runner._profile_timing_payload(
        timings={
            "dataset_load": 0.01,
            "group_build": 0.02,
            "epoch_loop": 0.03,
            "prediction_export": 0.04,
        },
        profile_counts={
            "graph_metadata_build_calls": 4,
            "graph_build_graphs_calls": 2,
        },
        epoch_records=[
            {
                "epoch": 1,
                "epoch_total_time_s": 0.03,
                "epoch_train_time_s": 0.01,
                "epoch_train_metrics_time_s": 0.01,
                "epoch_validation_time_s": 0.01,
                "train_batch_count": 1,
                "valid_batch_count": 1,
            }
        ],
        train_group_count=1,
        valid_group_count=1,
        all_group_count=2,
        train_batch_counts=[1],
        subset=runner.DEFAULT_SUBSET,
        output_dir=profile_json.parent,
    )
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    if decoded["counts"]["graph_metadata_build_calls"] != 4:
        raise AssertionError("profile count JSON roundtrip failed")

    if not runner.DEFAULT_SUBSET.exists():
        print(f"skipped real data smoke: subset not found locally: {runner.DEFAULT_SUBSET}")
    else:
        print("skipped real data smoke: helper-only local timing smoke; run real timing on SSH WSL")
    print("Heat3D v2 graph-build timing smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
