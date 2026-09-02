#!/usr/bin/env python3
"""Build a file-level manifest for the ignored V7 G1 evidence archive.

The archive itself is intentionally outside Git.  This manifest is the small,
reviewable control-plane record committed to Git.  It never opens JSON/NPZ/PKL
payloads; run identity and evidence role are derived from the path and file
name so the manifest builder cannot accidentally ingest result values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FORMAL_RUN_RE = re.compile(
    r"(?P<run_id>(?:Full|layout_agnostic_stratified_support|cv_only_support|"
    r"no_film|physics_scale_only|vanilla_RIGNO_capacity_matched)_seed(?P<seed>[0-9]+))"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(relative_path: Path) -> tuple[str | None, str | None, int | None]:
    match = FORMAL_RUN_RE.search(relative_path.as_posix())
    if match is None:
        return None, None, None
    return match.group("run_id"), match.group("run_id").rsplit("_seed", 1)[0], int(match.group("seed"))


def _role(relative_path: Path) -> str:
    name = relative_path.name
    path = relative_path.as_posix().lower()
    if name == "matrix_status.json":
        return "formal_matrix_status"
    if name == "v7_g1_formal_receipt.json":
        return "formal_receipt"
    if name == "params_best_sample_first.pkl":
        return "formal_checkpoint_best"
    if name == "params_final.pkl":
        return "formal_checkpoint_final"
    if name == "formal_stdout.log":
        return "formal_training_log_stdout"
    if name == "formal_stderr.log":
        return "formal_training_log_stderr"
    if name == "v7_g1_progress.json":
        return "formal_training_progress"
    if "h2_native_geometry" in path:
        return "h2_geometry_only_evidence"
    if name == "evaluation_receipt.json":
        return "h2_evaluation_receipt"
    if name == "per_sample_metrics.json":
        return "h2_per_sample_metrics"
    if name == "full_predictions_best.npz":
        return "h2_full_field_predictions_best"
    if name == "query_predictions_best.npz":
        return "h2_query_predictions_best"
    if name == "run_config.json":
        return "h2_run_config"
    if name in {"evaluation_contract.json", "support_reconstruction_provenance.json"}:
        return "h2_evaluation_provenance"
    if name == "implementation_provenance.json":
        return "h2_implementation_provenance"
    if "native_1024" in path:
        return "native_1024_evidence"
    if "formal_21_runs" in path or "formal_runs" in path:
        return "formal_evidence_other"
    if path.endswith(".log"):
        return "training_or_evaluation_log"
    return "archive_control_artifact"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-code-sha", required=True)
    parser.add_argument("--expected-formal-receipts", type=int, default=21)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    archive_root = args.archive_root.resolve()
    manifest_path = args.manifest.resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"missing archive root: {archive_root}")
    files = sorted(path for path in archive_root.rglob("*") if path.is_file())
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(archive_root)
        run_id, variant, seed = _identity(relative)
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "run_id": run_id,
                "variant": variant,
                "seed": seed,
                "evidence_role": _role(relative),
            }
        )
    formal_receipts = [entry for entry in entries if entry["evidence_role"] == "formal_receipt"]
    if len(formal_receipts) != args.expected_formal_receipts:
        raise ValueError(
            f"formal receipt count {len(formal_receipts)} != {args.expected_formal_receipts}"
        )
    if any("test_iid" in entry["path"].lower() or "sealed" in entry["path"].lower() for entry in entries):
        raise ValueError("test/sealed path entered the G1 evidence archive")
    payload = {
        "schema_version": "heat3d_v7_g1_formal_archive_manifest_v2",
        "archive_root": str(archive_root),
        "formal_code_sha": args.formal_code_sha,
        "file_count": len(entries),
        "formal_receipt_count": len(formal_receipts),
        "formal_run_ids": sorted({entry["run_id"] for entry in formal_receipts}),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "formal_receipt_count", "formal_run_ids")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
