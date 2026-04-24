import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_reference_solver import solve_reference_temperature
from rigno.heat3d_v1_supervised import default_v1_supervised_samples_dir


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
BASE_T = 300.0
SHIFTED_T = 350.0
SHIFT_DELTA = SHIFTED_T - BASE_T
TOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic smoke for T_ref baseline-shift invariance in v1 supervised samples."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_v1_supervised_samples_dir(REPO_DIR),
        help="Supervised smoke samples directory.",
    )
    return parser.parse_args()


def _copy_shifted_sample(source_dir: Path, temp_root: Path) -> Path:
    target_dir = temp_root / source_dir.name
    shutil.copytree(source_dir, target_dir)

    meta_path = target_dir / "sample_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["description"] = (
        f"{meta.get('description', '').strip()} "
        "Temporary diagnostic copy for T_ref baseline-shift invariance smoke only."
    ).strip()
    boundary_params = meta["boundary_params"]
    boundary_params["bottom"]["fixed_temperature_K"] = SHIFTED_T
    boundary_params["top"]["ambient_temperature_K"] = SHIFTED_T
    generation_config = dict(meta.get("generation_config", {}))
    generation_config.update({
        "diagnostic_only": True,
        "diagnostic_name": "tref_shift_invariance",
        "original_boundary_temperature_K": BASE_T,
        "shifted_boundary_temperature_K": SHIFTED_T,
        "formal_dataset_sample": False,
    })
    meta["generation_config"] = generation_config
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return target_dir


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.min(values)), float(np.max(values)), float(np.mean(values))


def _bc_feature_risk_summary(meta: dict) -> dict[str, float | str]:
    params = meta.get("boundary_params", {})
    top = params.get("top", {}) if isinstance(params, dict) else {}
    bottom = params.get("bottom", {}) if isinstance(params, dict) else {}
    return {
        "current_top_T_inf": float(top.get("ambient_temperature_K", np.nan)),
        "current_bottom_T_fixed": float(bottom.get("fixed_temperature_K", np.nan)),
        "feature_contract": "raw top_T_inf and bottom_T_fixed are included in condition_features",
        "risk": "300K-only training to 350K test is OOD in raw absolute temperature channels",
        "relative_feature_candidate": "T_ref, top_T_inf_minus_T_ref, bottom_T_fixed_minus_T_ref",
    }


def main() -> int:
    args = parse_args()
    sample_root = args.path
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if not (sample_root / sample_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing supervised smoke samples: {missing}")

    all_ok = True
    print("T_ref baseline-shift diagnostic smoke")
    print("  purpose: physical consistency diagnostic only, not training")
    print("  base case: bottom Dirichlet = 300 K, top Robin ambient = 300 K")
    print("  shifted case: bottom Dirichlet = 350 K, top Robin ambient = 350 K")
    print("  unchanged: coords, k_field, q_field, HTC, layers, regions, interfaces")
    print("  expected: T_shifted - T_base ~= 50 K and delta_T_shifted ~= delta_T_base")

    with tempfile.TemporaryDirectory(prefix="heat3d_v1_tref_shift_") as temp_name:
        temp_root = Path(temp_name)
        for sample_id in TARGET_SAMPLE_IDS:
            source_dir = sample_root / sample_id
            shifted_dir = _copy_shifted_sample(source_dir, temp_root)

            t_base = solve_reference_temperature(source_dir)
            t_shifted = solve_reference_temperature(shifted_dir)
            delta_base = t_base - BASE_T
            delta_shifted = t_shifted - SHIFTED_T
            shift_observed = t_shifted - t_base

            shift_error = float(np.max(np.abs(shift_observed - SHIFT_DELTA)))
            delta_error = float(np.max(np.abs(delta_shifted - delta_base)))
            sample_ok = shift_error <= TOL and delta_error <= TOL
            all_ok = all_ok and sample_ok

            meta = json.loads((source_dir / "sample_meta.json").read_text())
            risk_summary = _bc_feature_risk_summary(meta)

            t_base_min, t_base_max, t_base_mean = _summary(t_base)
            t_shift_min, t_shift_max, t_shift_mean = _summary(t_shifted)
            shift_min, shift_max, shift_mean = _summary(shift_observed)
            delta_base_min, delta_base_max, delta_base_mean = _summary(delta_base)
            delta_shift_min, delta_shift_max, delta_shift_mean = _summary(delta_shifted)

            print(f"\n{sample_id}")
            print(f"  T_base min/max/mean: {t_base_min:.9f}, {t_base_max:.9f}, {t_base_mean:.9f}")
            print(f"  T_shifted min/max/mean: {t_shift_min:.9f}, {t_shift_max:.9f}, {t_shift_mean:.9f}")
            print(f"  T_shifted - T_base min/max/mean: {shift_min:.9f}, {shift_max:.9f}, {shift_mean:.9f}")
            print(f"  max_abs((T_shifted - T_base) - 50K): {shift_error:.9e}")
            print(f"  DeltaT_base min/max/mean: {delta_base_min:.9f}, {delta_base_max:.9f}, {delta_base_mean:.9f}")
            print(f"  DeltaT_shifted min/max/mean: {delta_shift_min:.9f}, {delta_shift_max:.9f}, {delta_shift_mean:.9f}")
            print(f"  max_abs(DeltaT_shifted - DeltaT_base): {delta_error:.9e}")
            print(f"  smoke tolerance: {TOL:.1e}")
            print(f"  shift invariance ok: {sample_ok}")
            print("  BC feature encoding risk:")
            print(f"    current top_T_inf: {risk_summary['current_top_T_inf']}")
            print(f"    current bottom_T_fixed: {risk_summary['current_bottom_T_fixed']}")
            print(f"    feature contract: {risk_summary['feature_contract']}")
            print(f"    risk: {risk_summary['risk']}")
            print(f"    candidate relative features: {risk_summary['relative_feature_candidate']}")

    print("\nsummary")
    print(f"  all samples shift-invariant within tolerance: {all_ok}")
    print("  recommendation: record OOD risk; do not change canonical feature contract in this diagnostic batch")
    print("  next design candidate: relative BC temperature features tied to T_ref")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
