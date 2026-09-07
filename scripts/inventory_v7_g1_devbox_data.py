#!/usr/bin/env python3
"""Inventory devbox research-critical non-Git data without reading test/sealed.

The dataset inventory is driven by the frozen split manifest: only train and
valid_iid sample paths are hashed.  Other candidate roots are walked with
path-based exclusions.  The script hashes bytes for provenance only; it does
not parse checkpoints, predictions, logs, or temperature fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_TOKENS = ("test_iid", "sealed")
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".cache",
    "cache",
    "xla_cache",
    "jax_cache",
    "conda",
    ".conda",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in FORBIDDEN_TOKENS)


def record(path: Path, relative_path: str, role: str) -> dict[str, Any]:
    if forbidden(path) or not path.is_file():
        raise ValueError(f"refusing forbidden or missing inventory path: {path}")
    return {
        "path": relative_path,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file():
        if not forbidden(root):
            yield root
        return
    for path in sorted(root.rglob("*")):
        if any(part.lower() in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if forbidden(path):
            continue
        if path.is_file():
            yield path


def dataset_files(repo_root: Path, manifest_path: Path) -> list[tuple[Path, str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = repo_root / str(manifest["dataset_root"])
    entries: list[tuple[Path, str, str]] = []
    for sample in manifest.get("samples", []):
        split = str(sample.get("split_role"))
        if split not in {"train", "valid_iid"}:
            continue
        relative = str(sample["relative_path"])
        sample_dir = dataset_root / relative
        for path in sorted(sample_dir.iterdir()):
            if path.is_file():
                entries.append((path, str(path.relative_to(repo_root)), f"dataset {split}"))
    return entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-field-h5", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    full_field = args.full_field_h5.resolve()
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for path, relative, role in dataset_files(repo_root, manifest_path):
        if path not in seen:
            entries.append(record(path, relative, role))
            seen.add(path)

    # The full-field H5 is recorded as a file-level frozen artifact.  No H5
    # group/dataset is opened by this inventory, and split contents are not
    # inspected here.
    if full_field not in seen:
        entries.append(record(full_field, str(full_field.relative_to(repo_root)), "frozen full-field archive; file-level provenance"))
        seen.add(full_field)

    candidate_roots = (
        (repo_root / "checkpoints", "checkpoint/output asset"),
        (repo_root / "output", "checkpoint/output asset"),
        (repo_root / "logs", "training log"),
        (repo_root / "research_artifacts", "research artifact"),
    )
    for root, role in candidate_roots:
        for path in iter_files(root):
            if path in seen:
                continue
            entries.append(record(path, str(path.relative_to(repo_root)), role))
            seen.add(path)
    if args.tmp_root is not None:
        tmp_root = args.tmp_root.resolve()
        for path in iter_files(tmp_root):
            if path in seen:
                continue
            entries.append(record(path, f"/tmp/{path.relative_to(Path('/tmp'))}", "temporary recovery asset"))
            seen.add(path)

    entries.sort(key=lambda row: row["path"])
    total_bytes = sum(int(row["size"]) for row in entries)
    by_role: dict[str, dict[str, int]] = {}
    for row in entries:
        summary = by_role.setdefault(row["role"], {"files": 0, "bytes": 0})
        summary["files"] += 1
        summary["bytes"] += int(row["size"])
    payload = {
        "schema_version": "heat3d_v7_g1_devbox_data_inventory_v1",
        "status": "DEVBOX_INVENTORY_ONLY_WSL2_SKIPPED_BY_USER",
        "host": "devbox",
        "repo_root": str(repo_root),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "policy": {
            "hash_bytes_only": True,
            "dataset_split_source": "frozen manifest; train and valid_iid only",
            "excluded_tokens": list(FORBIDDEN_TOKENS),
            "excluded_directories": sorted(EXCLUDED_DIR_NAMES),
            "test_iid_access": False,
            "sealed_access": False,
            "wsl2_inventory": "skipped_by_user",
            "wsl2_sync": "skipped_by_user",
        },
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "by_role": by_role,
        "files": entries,
    }
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
