#!/usr/bin/env python3
"""Validate the preregistered U1 asymmetric-query audit and expected fail-closed result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_before_execution"
    assert protocol["scientific_contract"]["input_nodes"] == 1024
    assert protocol["scientific_contract"]["output_nodes"] == [8192, 32768]
    assert protocol["scientific_contract"]["regional_nodes"] == 256
    assert protocol["role_contract"] == {
        "accessed_roles": ["train_inputs_for_frozen_standardizer", "valid_iid_inputs"],
        "test_accessed": False,
        "sealed_accessed": False,
        "training_executed": False,
        "checkpoint_modified": False,
        "accuracy_evaluation_executed": False,
    }
    result_checked = False
    if args.result is not None:
        result = json.loads(args.result.read_text(encoding="utf-8"))
        assert result["status"] in {"passed_expected_no_go", "passed_probe"}
        assert result["checkpoint"]["unchanged"] is True
        assert result["role_contract"] == protocol["role_contract"]
        rows = result["resolutions"]
        assert [row["output_nodes"] for row in rows] == [8192, 32768]
        for row in rows:
            assert row["graph_build_status"] == "passed"
            assert row["native_encoder_graph_exact"] is True
            assert row["regional_nodes"] == 256
            assert row["input_nodes"] == 1024
            assert row["edge_counts"]["p2r"] > 0
            assert row["edge_counts"]["r2p"] > 0
            assert row["asymmetric_graph_audit"]["query_inside_native_domain"] is True
        if result["decision"]["u1"].startswith("NO_GO"):
            assert rows[0]["forward"]["status"] == "failed_structural_incompatibility"
            assert rows[0]["forward"]["decoder_core_reached_bypass"] is True
            assert rows[0]["tensor_contract"]["decoder_core_runtime_alignment_observed"] is True
            assert "decoder bypass requires" in rows[0]["forward"]["exception_message"]
            assert rows[1]["forward"]["status"].startswith("not_executed_fail_fast")
            assert len(result["decision"]["blockers"]) >= 2
        assert result["decision"]["production_route_replaced"] is False
        result_checked = True
    if args.report is not None:
        text = args.report.read_text(encoding="utf-8")
        for section in (
            "# V6 P1i U1", "## Probe", "## Interface audit",
            "## Structural potential", "## Blockers", "## Interpretation",
        ):
            assert section in text
    print(json.dumps({"u1_protocol_checked": True, "result_checked": result_checked}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
