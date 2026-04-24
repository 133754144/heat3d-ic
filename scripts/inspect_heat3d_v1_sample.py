import argparse
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_schema import default_v1_samples_dir, find_sample_dirs, summarize_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Heat3D v1 metadata-first samples."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_samples_dir(REPO_DIR),
        help="Sample directory, samples/ directory, or subset directory to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dirs = find_sample_dirs(args.path)
    if not sample_dirs:
        print(f"ERROR: no sample_xxx directories found under {args.path}")
        return 1

    print(f"Inspecting {len(sample_dirs)} sample(s) under {args.path}")
    for sample_dir in sample_dirs:
        summary = summarize_sample(sample_dir)
        print("\n" + summary["sample_id"])
        print(f"  path: {summary['sample_dir']}")
        print(f"  stage: {summary['stage']}")
        print(f"  split: {summary['split']}")
        print(f"  layers: {', '.join(summary['layers'])}")
        print(f"  boundary_types: {summary['boundary_types']}")
        print(f"  interfaces: {len(summary['interfaces'])}")
        print(f"  k_field shape: {summary['k_field_shape']}")
        print(
            "  q_field nonzero layers: "
            f"{summary['q_nonzero_layer_names']} "
            f"(ids={summary['q_nonzero_layer_ids']})"
        )
        for name, shape in summary["shapes"].items():
            print(f"  {name}: {shape}")

        counts = summary["parameter_source_counts"]
        print(
            "  parameter_sources: "
            f"literature_backed={counts['literature_backed']}, "
            f"provisional={counts['provisional_engineering_assumption']}, "
            f"requires_user_confirmation={counts['requires_user_confirmation']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
