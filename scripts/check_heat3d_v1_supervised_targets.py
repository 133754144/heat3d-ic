import argparse
from pathlib import Path
import sys

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_reference_solver import solve_reference_temperature
from rigno.heat3d_v1_supervised import Heat3DV1SupervisedDataset, default_v1_supervised_samples_dir


TARGET_SAMPLE_IDS = ("sample_000", "sample_005")
BOTTOM_TOL = 1e-8
REPEAT_TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-level sanity checks for v1 supervised temperature targets."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Supervised smoke samples directory.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional sample ids to check. Defaults to the legacy two-sample smoke for the default path, or all samples for an explicit path.",
    )
    return parser.parse_args()


def _boundary_indices(meta: dict, name: str) -> list[int]:
    for region in meta.get("boundary_regions", []):
        if region.get("name") == name:
            return list(region.get("point_indices", []))
    raise KeyError(f"Missing boundary region {name!r}")


def _layer_name_by_id(meta: dict) -> dict[int, str]:
    return {int(layer["id"]): layer["name"] for layer in meta.get("layers", [])}


def _mean_kz_for_layers(sample: dict, layer_names: list[str]) -> float | None:
    name_by_id = _layer_name_by_id(sample["meta"])
    target_ids = {layer_id for layer_id, name in name_by_id.items() if name in layer_names}
    if not target_ids:
        return None

    mask = np.isin(sample["layer_id"], list(target_ids))
    if not np.any(mask):
        return None

    k_field = sample["k_field"]
    if k_field.shape[1] == 1:
        return float(np.mean(k_field[mask, 0]))
    if k_field.shape[1] == 3:
        return float(np.mean(k_field[mask, 2]))
    return None


def main() -> int:
    args = parse_args()
    explicit_path = args.path is not None
    sample_path = args.path if explicit_path else default_v1_supervised_samples_dir(REPO_DIR)
    dataset = Heat3DV1SupervisedDataset(sample_path, k_encoding_mode="diag3")
    sample_index_by_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    target_sample_ids = (
        tuple(args.sample_ids)
        if args.sample_ids is not None and len(args.sample_ids) > 0
        else tuple(sample_index_by_id) if explicit_path else TARGET_SAMPLE_IDS
    )
    missing = [sample_id for sample_id in target_sample_ids if sample_id not in sample_index_by_id]
    if missing:
        raise ValueError(f"Required supervised smoke samples are missing: {missing}")

    failed = False
    summaries = {}

    for sample_id in target_sample_ids:
        sample = dataset.samples[sample_index_by_id[sample_id]]
        meta = sample["meta"]
        temperature = sample["temperature"]
        bottom_indices = _boundary_indices(meta, "bottom")
        top_indices = _boundary_indices(meta, "top")

        fixed_bottom = float(meta["boundary_params"]["bottom"]["fixed_temperature_K"])
        ambient_top = float(meta["boundary_params"]["top"]["ambient_temperature_K"])

        bottom_values = temperature[bottom_indices, 0]
        top_values = temperature[top_indices, 0]
        max_abs_bottom_error = float(np.max(np.abs(bottom_values - fixed_bottom)))
        top_mean_rise = float(np.mean(top_values) - ambient_top)
        top_max_rise = float(np.max(top_values) - ambient_top)

        rerun_1 = solve_reference_temperature(sample["sample_dir"])
        rerun_2 = solve_reference_temperature(sample["sample_dir"])
        saved_vs_rerun = float(np.max(np.abs(temperature - rerun_1)))
        rerun_repeat = float(np.max(np.abs(rerun_1 - rerun_2)))

        heat_layers = list(meta.get("generation_config", {}).get("heat_layers", []))
        hot_path_layers = heat_layers + [name for name in ("tim_equiv",) if name not in heat_layers]
        mean_hot_path_kz = _mean_kz_for_layers(sample, hot_path_layers)

        sample_failed = False
        if not np.issubdtype(temperature.dtype, np.floating):
            sample_failed = True
        if max_abs_bottom_error > BOTTOM_TOL:
            sample_failed = True
        if top_mean_rise <= 0.0 or top_max_rise <= 0.0:
            sample_failed = True
        if saved_vs_rerun > REPEAT_TOL or rerun_repeat > REPEAT_TOL:
            sample_failed = True

        failed = failed or sample_failed
        summaries[sample_id] = {
            "t_min": float(np.min(temperature)),
            "t_max": float(np.max(temperature)),
            "t_mean": float(np.mean(temperature)),
            "bottom_error": max_abs_bottom_error,
            "top_mean_rise": top_mean_rise,
            "top_max_rise": top_max_rise,
            "mean_hot_path_kz": mean_hot_path_kz,
            "status": not sample_failed,
        }

        print(f"\n{sample_id}")
        print(f"  temperature shape: {temperature.shape}")
        print(f"  temperature dtype: {temperature.dtype}")
        print(f"  temperature min/max/mean: {summaries[sample_id]['t_min']:.6f}, {summaries[sample_id]['t_max']:.6f}, {summaries[sample_id]['t_mean']:.6f}")
        print(f"  bottom Dirichlet target: {fixed_bottom:.6f} K")
        print(f"  bottom max abs error: {max_abs_bottom_error:.6e}")
        print(f"  top ambient: {ambient_top:.6f} K")
        print(f"  top mean rise: {top_mean_rise:.6f} K")
        print(f"  top max rise: {top_max_rise:.6f} K")
        print(f"  hot-path kz proxy ({hot_path_layers}): {mean_hot_path_kz}")
        print(f"  solver rerun max abs diff vs saved: {saved_vs_rerun:.6e}")
        print(f"  solver rerun max abs diff repeat: {rerun_repeat:.6e}")
        print(f"  sanity status: {not sample_failed}")

    direction_ok = True
    direction_note = "cross-sample direction check skipped; sample_000/sample_005 not both selected"
    if "sample_000" in summaries and "sample_005" in summaries:
        sample_000 = summaries["sample_000"]
        sample_005 = summaries["sample_005"]
        direction_note = (
            "sample_005 uses lower effective through-plane conductivity on the heated path "
            "(active_die_0/tim_equiv) than sample_000, so higher peak/mean temperature is expected."
        )
        if (
            sample_000["mean_hot_path_kz"] is not None
            and sample_005["mean_hot_path_kz"] is not None
            and sample_005["mean_hot_path_kz"] < sample_000["mean_hot_path_kz"]
        ):
            direction_ok = (
                sample_005["t_max"] > sample_000["t_max"]
                and sample_005["t_mean"] > sample_000["t_mean"]
            )
        if not direction_ok:
            failed = True

    print("\nsummary")
    print(f"  checked sample count: {len(target_sample_ids)}")
    print(f"  all selected sample sanity ok: {all(item['status'] for item in summaries.values())}")
    print(f"  cross-sample direction note: {direction_note}")
    print(f"  cross-sample direction ok: {direction_ok}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
