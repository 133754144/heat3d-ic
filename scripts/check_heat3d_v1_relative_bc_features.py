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

from rigno.heat3d_v1_native_supervised import Heat3DV1NativeSupervisedDataset
from rigno.heat3d_v1_reference_solver import solve_reference_temperature
from rigno.heat3d_v1_supervised import default_v1_supervised_samples_dir


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
BASE_T = 300.0
SHIFTED_T = 350.0
TOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic smoke for relative BC features and temperature-rise bridge policies."
        )
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
        "Temporary diagnostic copy for relative BC feature invariance smoke only."
    ).strip()
    meta["boundary_params"]["bottom"]["fixed_temperature_K"] = SHIFTED_T
    meta["boundary_params"]["top"]["ambient_temperature_K"] = SHIFTED_T
    generation_config = dict(meta.get("generation_config", {}))
    generation_config.update({
        "diagnostic_only": True,
        "diagnostic_name": "relative_bc_feature_invariance",
        "original_boundary_temperature_K": BASE_T,
        "shifted_boundary_temperature_K": SHIFTED_T,
        "formal_dataset_sample": False,
    })
    meta["generation_config"] = generation_config
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    shifted_temperature = solve_reference_temperature(target_dir)
    np.save(target_dir / "temperature.npy", shifted_temperature)
    return target_dir


def _feature_index(names: tuple[str, ...], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"Required feature {name!r} not found in {names}") from exc


def _range(values: np.ndarray) -> tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def _range_text(values: np.ndarray) -> str:
    lo, hi = _range(values)
    return f"{lo:.6f} .. {hi:.6f}"


def _sample_by_id(dataset: Heat3DV1NativeSupervisedDataset, sample_id: str):
    index_by_id = dataset.sample_index_by_id()
    if sample_id not in index_by_id:
        raise ValueError(f"Missing sample {sample_id!r}")
    return dataset[index_by_id[sample_id]]


def _check_target_excluded(feature_names: tuple[str, ...]) -> bool:
    blocked_names = {"temperature", "target_temperature", "target_u"}
    return all(name not in blocked_names for name in feature_names)


def _bridge_report(example, policy: str) -> dict[str, object]:
    bridge = example.build_temperature_rise_legacy_inputs_from_relative_features(
        bridge_policy=policy,
    )
    legacy_u = np.asarray(bridge.legacy_inputs.u)
    legacy_c = np.asarray(bridge.legacy_inputs.c)
    target_delta = np.asarray(bridge.target_delta_u)
    target_temperature = np.asarray(
        example.target.target_u.reshape(1, 1, example.target.target_u.shape[0], 1),
        dtype=np.float32,
    )
    reconstruction_error = float(
        np.max(np.abs(np.asarray(bridge.t_ref) + target_delta - target_temperature))
    )
    target_excluded = _check_target_excluded(bridge.condition_feature_names)
    return {
        "bridge": bridge,
        "legacy_u": legacy_u,
        "legacy_c": legacy_c,
        "target_delta": target_delta,
        "legacy_u_shape": tuple(legacy_u.shape),
        "legacy_u_range": _range(legacy_u),
        "legacy_c_shape": tuple(legacy_c.shape),
        "target_delta_shape": tuple(target_delta.shape),
        "target_delta_range": _range(target_delta),
        "target_excluded": target_excluded,
        "reconstruction_error": reconstruction_error,
    }


def _print_raw_relative_feature_check(base_example, shifted_example) -> bool:
    raw_names = base_example.condition.condition_feature_names
    raw_base = base_example.condition.condition_features
    raw_shifted = shifted_example.condition.condition_features
    top_raw_idx = _feature_index(raw_names, "top_T_inf")
    bottom_raw_idx = _feature_index(raw_names, "bottom_T_fixed")

    base_relative = base_example.get_relative_bc_feature_view()
    shifted_relative = shifted_example.get_relative_bc_feature_view()
    rel_top_idx = _feature_index(base_relative.condition_feature_names, "top_T_inf_minus_T_ref")
    rel_bottom_idx = _feature_index(
        base_relative.condition_feature_names,
        "bottom_T_fixed_minus_T_ref",
    )
    relative_diff = float(
        np.max(
            np.abs(
                shifted_relative.condition_features - base_relative.condition_features
            )
        )
    )
    raw_top_shift = float(np.max(raw_shifted[:, top_raw_idx] - raw_base[:, top_raw_idx]))
    raw_bottom_shift = float(
        np.max(raw_shifted[:, bottom_raw_idx] - raw_base[:, bottom_raw_idx])
    )
    relative_ok = relative_diff <= TOL
    raw_shift_ok = (
        abs(raw_top_shift - (SHIFTED_T - BASE_T)) <= TOL
        and abs(raw_bottom_shift - (SHIFTED_T - BASE_T)) <= TOL
    )

    print("  raw feature names:")
    print(f"    {raw_names}")
    print(f"  relative feature names:")
    print(f"    {base_relative.condition_feature_names}")
    print(f"  raw feature shape: {raw_base.shape}")
    print(f"  relative feature shape: {base_relative.condition_features.shape}")
    print(
        "  raw top_T_inf base/shifted range: "
        f"{_range_text(raw_base[:, top_raw_idx])} / {_range_text(raw_shifted[:, top_raw_idx])}"
    )
    print(
        "  raw bottom_T_fixed base/shifted range: "
        f"{_range_text(raw_base[:, bottom_raw_idx])} / {_range_text(raw_shifted[:, bottom_raw_idx])}"
    )
    print(
        "  relative top_T_inf_minus_T_ref base/shifted range: "
        f"{_range_text(base_relative.condition_features[:, rel_top_idx])} / "
        f"{_range_text(shifted_relative.condition_features[:, rel_top_idx])}"
    )
    print(
        "  relative bottom_T_fixed_minus_T_ref base/shifted range: "
        f"{_range_text(base_relative.condition_features[:, rel_bottom_idx])} / "
        f"{_range_text(shifted_relative.condition_features[:, rel_bottom_idx])}"
    )
    print(f"  max_abs(relative_features_shifted - relative_features_base): {relative_diff:.9e}")
    print(f"  raw-temperature channels shifted by 50K: {raw_shift_ok}")
    print(f"  relative feature invariance ok: {relative_ok}")
    print("  raw-temperature OOD risk: True for 300K-only training evaluated at 350K")
    return raw_shift_ok and relative_ok


def _print_bridge_policy_check(base_example, shifted_example, policy: str) -> bool:
    base_report = _bridge_report(base_example, policy)
    shifted_report = _bridge_report(shifted_example, policy)

    legacy_u_diff = float(
        np.max(np.abs(shifted_report["legacy_u"] - base_report["legacy_u"]))
    )
    legacy_c_diff = float(
        np.max(np.abs(shifted_report["legacy_c"] - base_report["legacy_c"]))
    )
    target_delta_diff = float(
        np.max(np.abs(shifted_report["target_delta"] - base_report["target_delta"]))
    )
    reconstruction_ok = (
        base_report["reconstruction_error"] <= TOL
        and shifted_report["reconstruction_error"] <= TOL
    )
    target_excluded = bool(base_report["target_excluded"] and shifted_report["target_excluded"])
    if policy == "zero_delta_u_bridge":
        baseline_shift_safer = (
            legacy_u_diff <= TOL
            and legacy_c_diff <= TOL
            and target_delta_diff <= TOL
            and target_excluded
        )
    else:
        baseline_shift_safer = (
            legacy_c_diff <= TOL
            and target_delta_diff <= TOL
            and target_excluded
            and legacy_u_diff <= TOL
        )

    base_bridge = base_report["bridge"]
    print(f"  policy: {policy}")
    print(f"    T_ref source: {base_bridge.t_ref_source}")
    print(f"    base legacy_inputs.u shape/range: {base_report['legacy_u_shape']} / {base_report['legacy_u_range']}")
    print(f"    shifted legacy_inputs.u shape/range: {shifted_report['legacy_u_shape']} / {shifted_report['legacy_u_range']}")
    print(f"    legacy_inputs.c shape: {base_report['legacy_c_shape']}")
    print(f"    target_delta shape/range: {base_report['target_delta_shape']} / {base_report['target_delta_range']}")
    print(f"    target_temperature excluded from inputs: {target_excluded}")
    print(f"    target_temperature == T_ref + target_delta_u: {reconstruction_ok}")
    print(f"    max_abs(u_shifted - u_base): {legacy_u_diff:.9e}")
    print(f"    max_abs(c_shifted - c_base): {legacy_c_diff:.9e}")
    print(f"    max_abs(target_delta_shifted - target_delta_base): {target_delta_diff:.9e}")
    print(f"    baseline-shift safer: {baseline_shift_safer}")
    print("    model forward: not run; this diagnostic checks bridge tensors and loss inputs only")
    return reconstruction_ok and target_excluded and legacy_c_diff <= TOL and target_delta_diff <= TOL


def main() -> int:
    args = parse_args()
    sample_root = args.path
    missing = [sample_id for sample_id in TARGET_SAMPLE_IDS if not (sample_root / sample_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing supervised smoke samples: {missing}")

    print("v1 relative BC feature and bridge-policy diagnostic")
    print("  native task: coords + condition_features -> target_temperature")
    print("  target role: temperature.npy is supervised target, not inference input")
    print("  temperature-rise target: DeltaT = T - T_ref")
    print("  shifted diagnostic: 300 K -> 350 K for bottom fixed and top ambient temperatures")
    print("  no training, no optimizer, no formal dataset write")

    all_ok = True
    with tempfile.TemporaryDirectory(prefix="heat3d_v1_relative_bc_") as temp_name:
        temp_root = Path(temp_name)
        for sample_id in TARGET_SAMPLE_IDS:
            _copy_shifted_sample(sample_root / sample_id, temp_root)

        base_dataset = Heat3DV1NativeSupervisedDataset(sample_root, k_encoding_mode="diag3")
        shifted_dataset = Heat3DV1NativeSupervisedDataset(temp_root, k_encoding_mode="diag3")

        for sample_id in TARGET_SAMPLE_IDS:
            print(f"\n{sample_id}")
            base_example = _sample_by_id(base_dataset, sample_id)
            shifted_example = _sample_by_id(shifted_dataset, sample_id)

            feature_ok = _print_raw_relative_feature_check(base_example, shifted_example)
            print("  bridge policy diagnostics:")
            tref_ok = _print_bridge_policy_check(base_example, shifted_example, "tref_u_bridge")
            zero_ok = _print_bridge_policy_check(base_example, shifted_example, "zero_delta_u_bridge")
            sample_ok = feature_ok and tref_ok and zero_ok
            all_ok = all_ok and sample_ok
            print(f"  sample diagnostic ok: {sample_ok}")

    print("\nsummary")
    print(f"  samples checked: {TARGET_SAMPLE_IDS}")
    print("  raw BC temperature features change under 300K -> 350K shift: True")
    print("  relative BC feature view remains invariant within tolerance if physics is otherwise unchanged")
    print("  tref_u_bridge keeps a shifting absolute-temperature legacy u")
    print("  zero_delta_u_bridge keeps legacy u invariant and leaves T_ref for reconstruction")
    print("  recommended next bridge for temperature-rise tiny training smoke: zero_delta_u_bridge")
    print(f"  diagnostic ok: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
