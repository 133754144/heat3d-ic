"""Generate a small Heat3D v1 physics-label v2 smoke subset.

This tool copies selected metadata samples from the existing ignored
supervised-small subset, removes prior labels, and writes new temperature labels
with the v2 research reference solver. It is a smoke generation path only; it
does not create a formal benchmark or high-fidelity label set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from rigno.heat3d_v1_manifest_resolver import load_manifest  # noqa: E402
from rigno.heat3d_v1_reference_solver_v2 import solve_reference_temperature_v2  # noqa: E402


SOURCE_SUBSET_NAME = "v1_multilayer_bc_eq_supervised_small"
PHYSICS_LABEL_SUBSET_NAME = "v1_multilayer_bc_eq_physics_label_small_v2"
DEFAULT_SAMPLE_IDS = ("sample_000", "sample_005", "sample_014", "sample_015")
PROTECTED_OUTPUT_SUBSET_NAMES = {
    "v1_multilayer_bc_eq_demo",
    "v1_multilayer_bc_eq_supervised_smoke",
    "v1_multilayer_bc_eq_supervised_small",
}
REQUIRED_SOURCE_FILES = (
    "coords.npy",
    "layer_id.npy",
    "region_id.npy",
    "material_id.npy",
    "k_field.npy",
    "q_field.npy",
    "sample_meta.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small v2 physics-label smoke subset with solver metadata."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_DIR / "configs" / "heat3d_v1_supervised_small_manifest.json",
        help="Supervised-small manifest used to select and annotate samples.",
    )
    parser.add_argument(
        "--source-subset",
        type=Path,
        default=(
            REPO_DIR
            / "data"
            / "heat3d-thermal-simulation"
            / "subsets"
            / SOURCE_SUBSET_NAME
        ),
        help="Existing source subset root or samples directory.",
    )
    parser.add_argument(
        "--output-subset",
        type=Path,
        default=(
            REPO_DIR
            / "data"
            / "heat3d-thermal-simulation"
            / "subsets"
            / PHYSICS_LABEL_SUBSET_NAME
        ),
        help="Output subset root for generated v2 physics-label smoke samples.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Sample ids to generate. Defaults to representative v2 smoke samples.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write the v2 subset. Without this flag, only prints the plan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing v2 output subset.",
    )
    return parser.parse_args()


def _samples_root(path: Path) -> Path:
    path = path.resolve()
    if path.name == "samples":
        return path
    return path / "samples"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _manifest_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or "sample_id" not in sample:
            raise ValueError("every manifest sample must be an object with sample_id")
        result[str(sample["sample_id"])] = sample
    return result


def _validate_generation_request(
    manifest: dict[str, Any],
    manifest_path: Path,
    source_subset: Path,
    output_subset: Path,
    sample_ids: list[str],
    overwrite: bool,
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    manifest_samples = _manifest_by_id(manifest)
    missing = [sample_id for sample_id in sample_ids if sample_id not in manifest_samples]
    if missing:
        raise ValueError(f"requested sample ids are missing from manifest: {missing}")

    source_samples = _samples_root(source_subset)
    if not source_samples.is_dir():
        raise FileNotFoundError(
            f"source samples directory not found: {source_samples}. "
            "Regenerate the ignored supervised-small subset with existing tools before running this generator."
        )

    output_subset = output_subset.resolve()
    if output_subset.name in PROTECTED_OUTPUT_SUBSET_NAMES:
        raise ValueError(f"refusing to write into protected subset: {output_subset.name}")
    if output_subset.exists() and not overwrite:
        raise FileExistsError(f"output subset already exists: {output_subset}; use --overwrite")

    try:
        output_subset.relative_to(REPO_DIR / "data")
    except ValueError as exc:
        raise ValueError(f"output subset must be under ignored data/: {output_subset}") from exc

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    for sample_id in sample_ids:
        sample_dir = source_samples / sample_id
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"missing source sample directory: {sample_dir}")
        for name in REQUIRED_SOURCE_FILES:
            if not (sample_dir / name).is_file():
                raise FileNotFoundError(f"{sample_id} missing required source file: {name}")

    return source_samples, output_subset, manifest_samples


def _prepare_output_subset(output_subset: Path, overwrite: bool) -> Path:
    if output_subset.exists() and overwrite:
        shutil.rmtree(output_subset)
    samples_dir = output_subset / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    return samples_dir


def _copy_metadata_sample(source_sample_dir: Path, target_sample_dir: Path) -> None:
    shutil.copytree(source_sample_dir, target_sample_dir)
    for generated_name in ("temperature.npy", "label_meta.json"):
        generated_path = target_sample_dir / generated_name
        if generated_path.exists():
            generated_path.unlink()


def _update_sample_meta_before_solve(
    meta: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    source_subset: Path,
    manifest_sample: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(meta)
    meta["subset_name"] = PHYSICS_LABEL_SUBSET_NAME
    meta["stage"] = "supervised_smoke"
    meta["description"] = (
        f"{meta.get('description', '').strip()} "
        "This copy belongs to the v2 physics-label smoke subset. "
        "temperature.npy is generated by the v2 research reference solver and "
        "is a benchmark-candidate smoke label, not a high-fidelity label."
    ).strip()

    generation_config = dict(meta.get("generation_config", {}))
    generation_config.update({
        "source_manifest": str(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "scaffold_base_commit": manifest.get("scaffold_base_commit"),
        "source_subset": str(source_subset),
        "source_sample_id": manifest_sample.get("sample_id"),
        "reference_solver": "rigno/heat3d_v1_reference_solver_v2.py",
        "reference_solver_role": "minimal_research_reference_path",
        "temperature_role": "physics_label_smoke_target",
        "dataset_role": "physics_label_small_v2_smoke",
        "not_formal_benchmark": True,
        "not_high_fidelity_solver": True,
        "not_model_performance_evidence": True,
        "not_ood_generalization_evidence": True,
    })
    meta["generation_config"] = generation_config

    validation = dict(meta.get("validation", {}))
    validation.update({
        "expected_stage": "supervised_smoke",
        "temperature_required": True,
        "label_meta_required": True,
        "label_diagnostics_required": True,
    })
    meta["validation"] = validation
    meta["physics_label_v2"] = {
        "source_manifest_sample": manifest_sample,
        "subset_role": "physics_label_smoke",
        "benchmark_candidate_only": True,
        "not_formal_benchmark": True,
    }
    return meta


def _finalize_label_meta(
    label_meta: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    source_subset: Path,
    output_subset: Path,
    sample_id: str,
) -> dict[str, Any]:
    meta = dict(label_meta)
    meta.update({
        "sample_id": sample_id,
        "label_role": "physics_label_small_v2_smoke",
        "temperature_file": "temperature.npy",
        "label_meta_file": "label_meta.json",
        "source_manifest": str(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "scaffold_base_commit": manifest.get("scaffold_base_commit"),
        "source_subset": str(source_subset),
        "output_subset": str(output_subset),
        "not_formal_benchmark": True,
        "not_high_fidelity_solver": True,
        "not_model_performance_evidence": True,
        "not_ood_generalization_evidence": True,
    })
    return meta


def _write_one_sample(
    sample_id: str,
    source_samples: Path,
    target_samples: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    source_subset: Path,
    output_subset: Path,
    manifest_sample: dict[str, Any],
) -> dict[str, Any]:
    source_sample_dir = source_samples / sample_id
    target_sample_dir = target_samples / sample_id
    _copy_metadata_sample(source_sample_dir, target_sample_dir)

    meta_path = target_sample_dir / "sample_meta.json"
    sample_meta = _read_json(meta_path)
    sample_meta = _update_sample_meta_before_solve(
        sample_meta,
        manifest=manifest,
        manifest_path=manifest_path,
        source_subset=source_subset,
        manifest_sample=manifest_sample,
    )
    _write_json(meta_path, sample_meta)

    temperature, label_meta = solve_reference_temperature_v2(target_sample_dir)
    if temperature.ndim != 2 or temperature.shape[1] != 1:
        raise ValueError(f"{sample_id} solver v2 returned invalid temperature shape: {temperature.shape}")
    if not np.all(np.isfinite(temperature)):
        raise ValueError(f"{sample_id} solver v2 returned non-finite temperature")

    label_meta = _finalize_label_meta(
        label_meta,
        manifest=manifest,
        manifest_path=manifest_path,
        source_subset=source_subset,
        output_subset=output_subset,
        sample_id=sample_id,
    )
    np.save(target_sample_dir / "temperature.npy", temperature)
    _write_json(target_sample_dir / "label_meta.json", label_meta)

    return {
        "sample_id": sample_id,
        "split": sample_meta.get("split"),
        "temperature_shape": list(temperature.shape),
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "solver_name": label_meta.get("solver_name"),
        "solver_version": label_meta.get("solver_version"),
        "convergence_flag": label_meta.get("convergence_flag"),
        "residual_norm": label_meta.get("residual_norm"),
        "bottom_dirichlet_error": label_meta.get("bottom_dirichlet_error"),
        "solver_warning_count": len(label_meta.get("warnings") or []),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    source_subset = args.source_subset.resolve()
    output_subset = args.output_subset.resolve()
    sample_ids = [str(sample_id) for sample_id in args.sample_ids]

    manifest = load_manifest(manifest_path)
    source_samples, output_subset, manifest_samples = _validate_generation_request(
        manifest=manifest,
        manifest_path=manifest_path,
        source_subset=source_subset,
        output_subset=output_subset,
        sample_ids=sample_ids,
        overwrite=args.overwrite,
    )

    split_counts = Counter(manifest_samples[sample_id].get("split") for sample_id in sample_ids)
    print("Heat3D v1 physics-label small v2 generation")
    print(f"manifest: {manifest_path}")
    print(f"source_subset: {source_subset}")
    print(f"output_subset: {output_subset}")
    print(f"sample_ids: {sample_ids}")
    print(f"split_counts: {dict(split_counts)}")
    print("scope: research reference / physics-label smoke; not a formal benchmark")

    if not args.write:
        print("write_enabled: False")
        print("no_data_written: True")
        return 0

    target_samples = _prepare_output_subset(output_subset, overwrite=args.overwrite)
    summaries = [
        _write_one_sample(
            sample_id=sample_id,
            source_samples=source_samples,
            target_samples=target_samples,
            manifest=manifest,
            manifest_path=manifest_path,
            source_subset=source_subset,
            output_subset=output_subset,
            manifest_sample=manifest_samples[sample_id],
        )
        for sample_id in sample_ids
    ]

    print("write_enabled: True")
    print(f"wrote_sample_count: {len(summaries)}")
    for summary in summaries:
        print(
            "- "
            f"{summary['sample_id']} split={summary['split']} "
            f"T_shape={summary['temperature_shape']} "
            f"T_min={summary['temperature_min']:.6f} "
            f"T_max={summary['temperature_max']:.6f} "
            f"solver={summary['solver_name']}@{summary['solver_version']} "
            f"converged={summary['convergence_flag']} "
            f"residual_norm={summary['residual_norm']:.6e} "
            f"bottom_error={summary['bottom_dirichlet_error']:.6e} "
            f"solver_warning_count={summary['solver_warning_count']}"
        )
    print("temperature_written: True")
    print("label_meta_written: True")
    print("old_supervised_small_overwritten: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
