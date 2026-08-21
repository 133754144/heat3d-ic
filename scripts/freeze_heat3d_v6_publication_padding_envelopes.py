#!/usr/bin/env python3
"""Materialize route-specific dummy-capacity sources from qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text())
    if qualification["status"] != "passed" or qualification["envelope_qualification"] != "GO":
        raise RuntimeError("graph envelope qualification did not pass")
    envelopes = qualification["padding_envelopes"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "status": "passed", "envelope_qualification": "GO",
        "source_qualification": str(args.qualification),
        "source_qualification_sha256": sha(args.qualification),
        "capacity_derivation": qualification["capacity_derivation"],
        "dummy_capacity_only": True, "real_graph_semantics_changed": False,
    }
    artifacts = {
        "native1024": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                       "route": "native1024_anchor",
                       "graph_cache": {"edge_targets": envelopes["native1024_anchor"]}},
        "E16384": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                   "route": "E16384_reconstruction",
                   "graph_cache": {"edge_targets": envelopes["E16384_reconstruction"]}},
        "E240825": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                    "route": "E240825_direct_control",
                    "graph_cache": {"edge_targets": envelopes["E240825_direct_control"]}},
        "U16384": {**common, "schema_version": "heat3d_v6_u_v2_padding_envelope_v1",
                   "route": "U_v2_16384_reconstruction", "sample_count": 32,
                   "train_only_warmup_count": 1,
                   "padding": {"actual_padding_envelope": {
                       "native": envelopes["native1024_anchor"],
                       "query": envelopes["U_v2_16384_reconstruction"]}}},
        "U240825": {**common, "schema_version": "heat3d_v6_u_v2_padding_envelope_v1",
                    "route": "U_v2_direct240825", "sample_count": 32,
                    "train_only_warmup_count": 1,
                    "padding": {"actual_padding_envelope": {
                        "native": envelopes["native1024_anchor"],
                        "query": envelopes["U_v2_direct240825"]}}},
    }
    manifest = {}
    for key, payload in artifacts.items():
        path = args.output_dir / f"v6_publication_padding_{key}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        manifest[key] = {"path": str(path), "sha256": sha(path)}
    manifest_path = args.output_dir / "v6_publication_padding_envelope_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "heat3d_v6_publication_padding_envelope_manifest_v1",
        "status": "passed", "envelope_qualification": "GO", "artifacts": manifest,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
