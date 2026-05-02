#!/usr/bin/env python3
"""Run Heat3D v1 label diagnostics smoke on supervised samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_label_diagnostics import diagnose_sample, find_sample_dirs  # noqa: E402


DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-level label diagnostics for Heat3D v1 supervised labels."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_SUBSET,
        help="Sample, samples, or subset path to diagnose.",
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Optional subset path. Preserves the positional path for older smoke commands.",
    )
    return parser.parse_args()


def _print_sample(report: dict) -> None:
    sample_id = report.get("sample_id")
    split = report.get("split")
    status = report.get("overall_status")
    print(f"\n[{status}] {sample_id} split={split}")

    for error in report.get("errors", []):
        print(f"  ERROR: {error}")
    for warning in report.get("warnings", []):
        print(f"  WARNING: {warning}")

    arrays = report.get("arrays", {})
    if arrays:
        coords = arrays.get("coords.npy", {})
        k_field = arrays.get("k_field.npy", {})
        q_field = arrays.get("q_field.npy", {})
        temperature_array = arrays.get("temperature.npy", {})
        print(f"  coords shape/dtype/finite: {coords.get('shape')} {coords.get('dtype')} {coords.get('finite')}")
        print(f"  k_field shape/dtype/finite: {k_field.get('shape')} {k_field.get('dtype')} {k_field.get('finite')}")
        print(f"  q_field shape/dtype/finite: {q_field.get('shape')} {q_field.get('dtype')} {q_field.get('finite')}")
        print(f"  temperature shape/dtype/finite: {temperature_array.get('shape')} {temperature_array.get('dtype')} {temperature_array.get('finite')}")

    t_ref = report.get("t_ref")
    if t_ref:
        print(
            "  T_ref: "
            f"{t_ref.get('value')} K source={t_ref.get('source')} fallback={t_ref.get('fallback')}"
        )

    temperature = report.get("temperature")
    if temperature:
        print(
            "  T min/max/mean: "
            f"{temperature['T_min']:.6f}, {temperature['T_max']:.6f}, {temperature['T_mean']:.6f}"
        )
        print(
            "  DeltaT min/max/mean: "
            f"{temperature['DeltaT_min']:.6f}, {temperature['DeltaT_max']:.6f}, {temperature['DeltaT_mean']:.6f}"
        )
        print(
            "  peak: "
            f"T={temperature['peak_temperature']:.6f} index={temperature['peak_index']} "
            f"coord={temperature['peak_coord']}"
        )

    bottom = report.get("bottom_dirichlet")
    if bottom:
        print(
            "  bottom Dirichlet: "
            f"status={bottom.get('status')} max_abs_error_K={bottom.get('max_abs_error_K')}"
        )

    label_meta = report.get("label_meta", {})
    if label_meta.get("present"):
        print(
            "  label_meta: "
            f"status={label_meta.get('status')} "
            f"solver={label_meta.get('solver_name')}@{label_meta.get('solver_version')} "
            f"converged={label_meta.get('convergence_flag')} "
            f"residual_norm={label_meta.get('residual_norm')} "
            f"bottom_error={label_meta.get('bottom_dirichlet_error')} "
            f"solver_warning_count={label_meta.get('warning_count')}"
        )
        solver_warnings = label_meta.get("warnings") or []
        if solver_warnings:
            print(f"  label_meta solver warnings: {solver_warnings}")

    not_computed = report.get("not_computed", {})
    if not_computed:
        keys = ", ".join(
            f"{key}:{value.get('status')}" for key, value in sorted(not_computed.items())
        )
        print(f"  physics diagnostics not computed: {keys}")


def main() -> int:
    args = parse_args()
    target_path = args.subset if args.subset is not None else args.path
    sample_dirs = find_sample_dirs(target_path)
    if not sample_dirs:
        print(f"ERROR: no sample_* directories found under {target_path}")
        print("No data were generated. Run the relevant generator explicitly if needed.")
        return 1

    reports = [diagnose_sample(sample_dir) for sample_dir in sample_dirs]
    for report in reports:
        _print_sample(report)

    status_counts = Counter(report["overall_status"] for report in reports)
    split_counts: defaultdict[str, int] = defaultdict(int)
    split_status_counts: defaultdict[str, Counter] = defaultdict(Counter)
    warning_samples: list[str] = []
    fail_samples: list[str] = []
    label_meta_present_count = 0
    label_meta_missing_count = 0

    for report in reports:
        split = report.get("split") or "<missing>"
        status = report.get("overall_status")
        sample_id = report.get("sample_id")
        split_counts[split] += 1
        split_status_counts[split][status] += 1
        if status == "warning":
            warning_samples.append(sample_id)
        if status == "fail":
            fail_samples.append(sample_id)
        label_meta = report.get("label_meta", {})
        if label_meta.get("present"):
            label_meta_present_count += 1
        else:
            label_meta_missing_count += 1

    print("\nsummary")
    print(f"  diagnosed_sample_count: {len(reports)}")
    print(f"  status_counts: {dict(status_counts)}")
    print(f"  split_counts: {dict(sorted(split_counts.items()))}")
    print(
        "  split_status_counts: "
        f"{ {split: dict(counts) for split, counts in sorted(split_status_counts.items())} }"
    )
    print(f"  warning_samples: {warning_samples}")
    print(f"  fail_samples: {fail_samples}")
    print(f"  label_meta_present_count: {label_meta_present_count}")
    print(f"  label_meta_missing_count: {label_meta_missing_count}")
    print("  diagnostics_scope: smoke diagnostics only; not physics validation")

    return 1 if fail_samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
