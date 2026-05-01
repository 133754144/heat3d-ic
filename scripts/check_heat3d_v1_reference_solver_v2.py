#!/usr/bin/env python3
"""Verification smoke for the Heat3D v1 reference solver v2 minimal path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rigno.heat3d_v1_reference_solver_v2 import solve_reference_temperature_v2  # noqa: E402


DEFAULT_SUBSET = (
    REPO_ROOT
    / "data"
    / "heat3d-thermal-simulation"
    / "subsets"
    / "v1_multilayer_bc_eq_supervised_small"
)
DEFAULT_SAMPLE_IDS = ("sample_000", "sample_005")
BOTTOM_TOL_K = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reference solver v2 verification smoke on selected v1 samples."
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET,
        help="Subset root or samples directory containing sample_xxx directories.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Sample ids to verify.",
    )
    return parser.parse_args()


def _samples_root(path: Path) -> Path:
    if path.name == "samples":
        return path
    return path / "samples"


def _load_meta(sample_dir: Path) -> dict:
    with (sample_dir / "sample_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _bottom_error_from_coords(sample_dir: Path, temperature: np.ndarray) -> float:
    coords = np.load(sample_dir / "coords.npy")
    meta = _load_meta(sample_dir)
    bottom_t = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
    bottom_mask = np.isclose(coords[:, 2], float(np.min(coords[:, 2])))
    return float(np.max(np.abs(temperature[bottom_mask, 0] - bottom_t)))


def main() -> int:
    args = parse_args()
    root = _samples_root(args.subset)
    if not root.is_dir():
        print(f"ERROR: sample root does not exist: {root}")
        print("No data were generated. Restore ignored v1 supervised small data before running this smoke.")
        return 1

    failed = False
    summaries: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_solver_v2_") as tmp:
        tmp_path = Path(tmp)
        for sample_id in args.sample_ids:
            sample_dir = root / sample_id
            if not sample_dir.is_dir():
                print(f"ERROR: missing sample directory: {sample_dir}")
                failed = True
                continue

            temperature, label_meta = solve_reference_temperature_v2(sample_dir)
            out_temperature = tmp_path / f"{sample_id}_temperature_v2.npy"
            out_meta = tmp_path / f"{sample_id}_label_meta_v2.json"
            np.save(out_temperature, temperature)
            out_meta.write_text(json.dumps(label_meta, indent=2, sort_keys=True), encoding="utf-8")

            finite_temperature = bool(np.all(np.isfinite(temperature)))
            bottom_error = _bottom_error_from_coords(sample_dir, temperature)
            sample_ok = (
                temperature.ndim == 2
                and temperature.shape[1] == 1
                and finite_temperature
                and bool(label_meta["convergence_flag"])
                and np.isfinite(float(label_meta["residual_norm"]))
                and bottom_error <= BOTTOM_TOL_K
            )
            failed = failed or not sample_ok
            summaries.append(
                {
                    "sample_id": sample_id,
                    "shape": list(temperature.shape),
                    "finite": finite_temperature,
                    "T_min": float(np.min(temperature)),
                    "T_max": float(np.max(temperature)),
                    "T_mean": float(np.mean(temperature)),
                    "supported_k_mode": label_meta["supported_k_mode"],
                    "convergence_flag": bool(label_meta["convergence_flag"]),
                    "residual_norm": float(label_meta["residual_norm"]),
                    "bottom_error": bottom_error,
                    "warnings": label_meta["warnings"],
                    "sample_ok": sample_ok,
                }
            )

            print(f"\n{sample_id}")
            print(f"  temperature shape: {temperature.shape}")
            print(f"  finite: {finite_temperature}")
            print(
                "  T min/max/mean: "
                f"{np.min(temperature):.6f}, {np.max(temperature):.6f}, {np.mean(temperature):.6f}"
            )
            print(f"  supported_k_mode: {label_meta['supported_k_mode']}")
            print(f"  convergence_flag: {label_meta['convergence_flag']}")
            print(f"  residual_norm: {label_meta['residual_norm']:.6e}")
            print(f"  bottom_dirichlet_error: {bottom_error:.6e}")
            print(f"  top_robin_status: {label_meta['top_robin_status']['status']}")
            print(f"  side_adiabatic_status: {label_meta['side_adiabatic_status']['status']}")
            print(f"  interface_status: {label_meta['interface_status']['status']}")
            print(f"  energy_balance_status: {label_meta['energy_balance_status']['status']}")
            print(f"  temporary outputs: {out_temperature.name}, {out_meta.name}")
            print(f"  sample_ok: {sample_ok}")

        print("\nsummary")
        print(f"  subset: {args.subset}")
        print(f"  sample_ids: {list(args.sample_ids)}")
        print(f"  temporary_output_dir: {tmp_path}")
        print("  temporary outputs are removed after this smoke")
        print(f"  all_samples_ok: {not failed and len(summaries) == len(args.sample_ids)}")
        print("  scope: minimal research reference path; not high-fidelity validation")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
