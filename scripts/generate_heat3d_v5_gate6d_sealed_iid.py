#!/usr/bin/env python3
"""Execute the frozen Gate 6D sealed-IID generation contract (never inference)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_heat3d_v4_p3c_smoke16 as generator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["status"] != "frozen_not_generated" or contract["model_inference_run"]:
        raise ValueError("sealed contract is not generation-ready")
    seed, count = int(contract["seed"]), int(contract["sample_count"])
    expected_ids = [f"sealed_iid_{seed}_{index:04d}" for index in range(count)]
    generator._sample_id_for_index = lambda index, sample_count: expected_ids[index]  # type: ignore[attr-defined]
    dataset_dir = ROOT / contract["data_dir"]
    output_dir = ROOT / contract["audit_output_dir"]
    generator.generate_smoke16(
        registry_path=ROOT / "configs/heat3d_v4/p3c_parameter_registry.json",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sample_count=count,
        seed=seed,
        force=args.force,
        reject_resample=True,
        max_candidates=1024,
        accepted_qc_classes={"clean_keep"},
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_ids = [row["sample_id"] for row in manifest["samples"]]
    if actual_ids != expected_ids:
        raise ValueError("generated IDs differ from frozen sealed IDs")
    for row in manifest["samples"]:
        row["split"] = "sealed_iid_test"
        meta_path = dataset_dir / row["sample_id"] / "sample_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["split"] = "sealed_iid_test"
        meta["split_policy"] = {"policy": "gate6d_single_sealed_role_v1", "target_derived": False}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["split_map"] = {sample_id: "sealed_iid_test" for sample_id in expected_ids}
    manifest["sealed_role"] = "sealed_iid_test"
    manifest["model_inference_run"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_payload = {
        "schema_version": "heat3d_v5_gate6d_sealed_split_v1",
        "role": "sealed_iid_test",
        "sample_count": count,
        "sample_ids": expected_ids,
    }
    provenance = json.loads(
        (ROOT / contract["artifacts"]["planned_provenance"]).read_text(encoding="utf-8")
    )
    provenance.update({
        "status": "generated_not_evaluated",
        "model_inference_run": False,
        "seed_changed_after_label_observation": False,
    })
    for directory in (dataset_dir, output_dir):
        (directory / "split_map.json").write_text(
            json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    sha = generator.build_sha256_manifest(dataset_dir)
    (dataset_dir / "sha256_manifest.json").write_text(json.dumps(sha, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(sha, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "generated_not_evaluated", "sample_count": count, "seed": seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
