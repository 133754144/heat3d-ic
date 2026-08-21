#!/usr/bin/env python3
"""Materialize route-specific dummy-capacity sources from qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EDGE_FIELDS = (
    "p2r_edge_indices",
    "r2r_edge_indices",
    "r2r_edge_domains",
    "r2p_edge_indices",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _maximum(*sources: dict[str, int | None]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in EDGE_FIELDS:
        values = [int(source[field]) for source in sources if source.get(field) is not None]
        result[field] = max(values) if values else None
    return result


def _warmup_maximum(run: dict[str, Any], route: str) -> dict[str, int | None]:
    rows = [
        row["edge_counts"] for row in run["records"]
        if row["route"] == route and row["population"] == "train_only_static_warmup"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"{route}: expected exactly one train-only warmup graph")
    return {field: rows[0].get(field) for field in EDGE_FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-run", type=Path, required=True)
    parser.add_argument("--previous-seal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text())
    qualification_run = json.loads(args.qualification_run.read_text())
    previous_seal = json.loads(args.previous_seal.read_text())
    if qualification["status"] != "passed" or qualification["envelope_qualification"] != "GO":
        raise RuntimeError("graph envelope qualification did not pass")
    qualified = qualification["padding_envelopes"]
    previous = previous_seal["runtime_state"]

    # Native input graphs are the same semantic object for E and U.  Taking the
    # maximum over every historically frozen native/anchor envelope guarantees
    # that the common publication envelope can never shrink either route.
    previous_native = _maximum(
        previous["E16384_reconstruction"]["padding_envelope"]["anchor"],
        previous["E240825_direct_control"]["padding_envelope"]["anchor"],
        previous["U_v2_16384_reconstruction"]["padding_envelope"]["native"],
        previous["U_v2_direct240825"]["padding_envelope"]["native"],
    )
    previous_query = {
        route: previous[route]["padding_envelope"]["query"]
        for route in (
            "E16384_reconstruction",
            "E240825_direct_control",
            "U_v2_16384_reconstruction",
            "U_v2_direct240825",
        )
    }
    warmup = {
        route: _warmup_maximum(qualification_run, route)
        for route in qualified
    }
    envelopes = {
        "native1024_anchor": _maximum(
            previous_native, qualified["native1024_anchor"], warmup["native1024_anchor"]),
    }
    for route in previous_query:
        envelopes[route] = _maximum(previous_query[route], qualified[route], warmup[route])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "status": "passed", "envelope_qualification": "GO",
        "source_qualification": str(args.qualification),
        "source_qualification_sha256": sha(args.qualification),
        "capacity_derivation": (
            "elementwise_max_previous_same_semantic_frozen_capacity_"
            "qualified_valid32_max_train_only_warmup"
        ),
        "dummy_capacity_only": True, "real_graph_semantics_changed": False,
        "monotonic_expansion_only": True,
        "source_previous_seal": str(args.previous_seal),
        "source_previous_seal_sha256": sha(args.previous_seal),
        "source_qualification_run": str(args.qualification_run),
        "source_qualification_run_sha256": sha(args.qualification_run),
    }
    derivation = {
        "native1024_anchor": {
            "previous_same_semantic_frozen_capacity": previous_native,
            "qualified_valid32_max": qualified["native1024_anchor"],
            "train_only_warmup": warmup["native1024_anchor"],
            "monotonic_envelope": envelopes["native1024_anchor"],
        }
    }
    for route in previous_query:
        derivation[route] = {
            "previous_same_semantic_frozen_capacity": previous_query[route],
            "qualified_valid32_max": qualified[route],
            "train_only_warmup": warmup[route],
            "monotonic_envelope": envelopes[route],
        }
    artifacts = {
        "native1024": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                       "route": "native1024_anchor",
                       "capacity_inputs": derivation["native1024_anchor"],
                       "graph_cache": {"edge_targets": envelopes["native1024_anchor"]}},
        "E16384": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                   "route": "E16384_reconstruction",
                   "capacity_inputs": derivation["E16384_reconstruction"],
                   "graph_cache": {"edge_targets": envelopes["E16384_reconstruction"]}},
        "E240825": {**common, "schema_version": "heat3d_v6_padding_envelope_v1",
                    "route": "E240825_direct_control",
                    "capacity_inputs": derivation["E240825_direct_control"],
                    "graph_cache": {"edge_targets": envelopes["E240825_direct_control"]}},
        "U16384": {**common, "schema_version": "heat3d_v6_u_v2_padding_envelope_v1",
                   "route": "U_v2_16384_reconstruction", "sample_count": 32,
                   "train_only_warmup_count": 1,
                   "capacity_inputs": {
                       "native": derivation["native1024_anchor"],
                       "query": derivation["U_v2_16384_reconstruction"],
                   },
                   "padding": {"actual_padding_envelope": {
                       "native": envelopes["native1024_anchor"],
                       "query": envelopes["U_v2_16384_reconstruction"]}}},
        "U240825": {**common, "schema_version": "heat3d_v6_u_v2_padding_envelope_v1",
                    "route": "U_v2_direct240825", "sample_count": 32,
                    "train_only_warmup_count": 1,
                    "capacity_inputs": {
                        "native": derivation["native1024_anchor"],
                        "query": derivation["U_v2_direct240825"],
                    },
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
        "status": "passed", "envelope_qualification": "GO",
        "padding_semantics": "monotonic_dummy_capacity_only",
        "artifacts": manifest,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
