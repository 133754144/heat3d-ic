import argparse
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_schema import default_v1_samples_dir, validate_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Heat3D v1 metadata-first samples."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_samples_dir(REPO_DIR),
        help="Sample directory, samples/ directory, or subset directory to validate.",
    )
    parser.add_argument(
        "--allow-non-metadata-stage",
        action="store_true",
        help="Allow solver_smoke stage instead of requiring metadata_only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_path(
        args.path,
        require_metadata_only=not args.allow_non_metadata_stage,
    )

    print(f"Validated root: {report['root']}")
    print(f"Sample count: {report['sample_count']}")
    print(f"Error count: {report['error_count']}")
    print(f"Warning count: {report['warning_count']}")

    if report["sample_count"] == 0:
        print("ERROR: no sample_xxx directories found")
        return 1

    for result in report["results"]:
        meta = result["meta"]
        status = "OK" if not result["errors"] else "FAIL"
        print(
            f"[{status}] {meta.get('sample_id')} "
            f"stage={meta.get('stage')} split={meta.get('split')} "
            f"layers={meta.get('layer_count')} interfaces={meta.get('interface_count')}"
        )
        for error in result["errors"]:
            print(f"  ERROR: {error}")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")

    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
