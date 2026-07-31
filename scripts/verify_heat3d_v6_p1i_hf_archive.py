#!/usr/bin/env python3
"""Verify the immutable Hugging Face archive for P1i formal1024_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


REPO_ID = "133754144X/heat3d-thermal-simulation"
REPO_TYPE = "dataset"
DATASET_ID = "heat3d_v6_p1i_continuous_physics1024_v1"
REMOTE_ROOT = f"subsets/{DATASET_ID}"
DEFAULT_REVISION = "archive-p1i-formal1024-v1-27d2ea3b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def verify(repo_root: Path, revision: str) -> dict[str, Any]:
    manifest_path = (
        repo_root
        / "configs/heat3d_v6_p1i/v6_p1i_formal1024_v1_manifest.json"
    )
    audit_path = (
        repo_root
        / "configs/heat3d_v6_p1i/"
        "v6_p1i_formal1024_v1_training_preflight_audit.json"
    )
    readme_path = (
        repo_root / "docs/v6_p1i_formal1024_v1_archive_readme.md"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, Any]] = {}
    local_dataset = repo_root / "data" / DATASET_ID
    for sample in manifest["samples"]:
        for filename, digest in sample["file_sha256"].items():
            relative = Path(sample["relative_path"]) / filename
            local_path = local_dataset / relative
            expected[f"{REMOTE_ROOT}/{relative}"] = {
                "sha256": digest,
                "local_path": local_path,
            }
    expected_metadata = {
        f"{REMOTE_ROOT}/dataset_manifest.json": {
            "sha256": sha256(manifest_path),
            "local_path": manifest_path,
        },
        f"{REMOTE_ROOT}/training_preflight_audit.json": {
            "sha256": sha256(audit_path),
            "local_path": audit_path,
        },
        f"{REMOTE_ROOT}/README.md": {
            "sha256": sha256(readme_path),
            "local_path": readme_path,
        },
    }

    api = HfApi()
    info = api.repo_info(
        REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
    )
    entries = list(
        api.list_repo_tree(
            REPO_ID,
            path_in_repo=REMOTE_ROOT,
            recursive=True,
            revision=revision,
            repo_type=REPO_TYPE,
        )
    )
    files = {
        item.path: item
        for item in entries
        if isinstance(item, RepoFile)
    }
    missing = sorted((set(expected) | set(expected_metadata)) - set(files))
    extra = sorted(set(files) - (set(expected) | set(expected_metadata)))
    mismatches = []
    for path, record in expected.items():
        item = files.get(path)
        if item is None:
            continue
        local_path = Path(record["local_path"])
        if not local_path.is_file():
            raise RuntimeError(f"missing local frozen file: {local_path}")
        if item.lfs is not None:
            matched = item.lfs.sha256 == record["sha256"]
            actual_identity = item.lfs.sha256
            identity_type = "lfs_sha256"
        else:
            local_blob = git_blob_sha1(local_path)
            matched = item.blob_id == local_blob
            actual_identity = item.blob_id
            identity_type = "git_blob_sha1"
        if not matched:
            mismatches.append(
                {
                    "path": path,
                    "expected_sha256": record["sha256"],
                    "actual_remote_identity": actual_identity,
                    "identity_type": identity_type,
                }
            )
    for path, record in expected_metadata.items():
        item = files.get(path)
        if item is None:
            continue
        local_path = Path(record["local_path"])
        if item.lfs is not None:
            matched = item.lfs.sha256 == record["sha256"]
            actual_identity = item.lfs.sha256
            identity_type = "lfs_sha256"
        else:
            local_blob = git_blob_sha1(local_path)
            matched = item.blob_id == local_blob
            actual_identity = item.blob_id
            identity_type = "git_blob_sha1"
        if not matched:
            mismatches.append(
                {
                    "path": path,
                    "expected_sha256": record["sha256"],
                    "actual_remote_identity": actual_identity,
                    "identity_type": identity_type,
                }
            )

    passed = not missing and not extra and not mismatches
    return {
        "schema_version": "heat3d_v6_p1i_hf_archive_verification_v1",
        "dataset_id": DATASET_ID,
        "status": "passed" if passed else "failed",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": revision,
        "resolved_commit_sha": info.sha,
        "remote_root": REMOTE_ROOT,
        "expected_sample_file_count": len(expected),
        "expected_metadata_file_count": len(expected_metadata),
        "verified_file_count": len(files),
        "missing_paths": missing,
        "extra_paths": extra,
        "sha256_mismatches": mismatches,
        "formal_manifest_sha256": sha256(manifest_path),
        "formal_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "guardrails": {
            "other_subset_paths_modified_by_verification": False,
            "dataset_regeneration_runs": 0,
            "training_runs": 0,
            "model_inference_runs": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()
    result = verify(args.repo_root.resolve(), args.revision)
    if args.write_result is not None:
        output = args.write_result
        if not output.is_absolute():
            output = args.repo_root.resolve() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
