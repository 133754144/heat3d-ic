"""Fail-closed checks for pre-G1 variant implementation qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "heat3d_v7"
RECEIPT_PATH = ROOT / "docs" / "v7_g1_variant_qualification_receipt.json"
SEMANTIC_ANCHOR_PATH = ROOT / "docs" / "v7_g1_full_p1i_semantic_anchor_receipt.json"
PROFILING_PATH = ROOT / "docs" / "v7_g1_synced_profiling_receipt.json"
CLOSEOUT_PATH = ROOT / "docs" / "v7_g1_scientific_closeout_receipt.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    receipt = _load(RECEIPT_PATH)
    matrix = _load(CONTROL / "v7_g1_variant_execution_matrix.json")
    fairness = _load(CONTROL / "v7_parameter_fairness_contract.json")
    protocol = _load(ROOT / "docs" / "v7_g1_scientific_protocol_freeze.json")
    _require(
        receipt.get("schema_version") == "heat3d_v7_g1_variant_qualification_receipt_v1",
        "variant qualification receipt schema drifted",
    )
    _require(len(str(receipt.get("code_commit", ""))) == 40, "qualification code commit is not pinned")
    _require(receipt.get("execution_role") == "variant_qualification", "qualification role drifted")
    for key in ("publication_evidence", "g1_formal", "scientific_evidence_eligible"):
        _require(receipt.get(key) is False, f"qualification evidence guard drifted: {key}")
    policy = receipt.get("qualification_policy", {})
    _require(policy.get("epochs") == 1 and policy.get("seed") == 0, "qualification budget/seed drifted")
    for key in ("formal_g1_multi_seed_started", "new_scientific_result", "solver", "new_data", "full_v6_evidence_modified"):
        _require(policy.get(key) is False, f"qualification prohibited-action guard drifted: {key}")

    variants = receipt.get("variants", {})
    for name in ("no_scale", "vanilla_RIGNO_capacity_matched"):
        row = variants.get(name, {})
        _require(row.get("status") == "COMPLETE", f"missing completed qualification: {name}")
        _require(len(str(row.get("receipt_sha256", ""))) == 64, f"missing source receipt SHA: {name}")
        _require(row.get("observation_only") is True, f"qualification must be observation-only: {name}")
    _require(
        variants["no_scale"].get("implementation", "").startswith("native_shape_scale"),
        "no_scale is not the native shape-scale route",
    )
    _require(
        "learned residual correction disabled" in variants["no_scale"].get("implementation", ""),
        "no_scale does not preserve physics-only scale semantics",
    )
    _require(
        variants["vanilla_RIGNO_capacity_matched"].get("node_latent_size") == 100
        and variants["vanilla_RIGNO_capacity_matched"].get("edge_latent_size") == 100,
        "capacity-matched Vanilla width drifted",
    )
    for name in ("generic_uniform_support", "volume_only_support", "no_context"):
        _require(
            receipt.get("unresolved_variants", {}).get(name, {}).get("status") == "blocked",
            f"unresolved variant was not fail-closed: {name}",
        )

    _require(matrix.get("schema_version") == "heat3d_v7_g1_variant_execution_matrix_v2", "variant matrix schema drifted")
    _require(matrix.get("formal_execution_started") is False, "formal G1 execution opened")
    no_scale = next(row for row in matrix["base_variants"] if row.get("variant") == "no_scale")
    _require("physics_scale_only" in no_scale.get("implementation_status", ""), "matrix no_scale semantics drifted")
    _require("qualified_nonpublication_1epoch" in no_scale.get("qualification_status", ""), "matrix no_scale qualification missing")
    for name in ("generic_uniform_support", "volume_only_support", "no_context"):
        row = next(row for row in matrix["base_variants"] if row.get("variant") == name)
        _require("blocked" in row.get("qualification_status", ""), f"matrix did not block {name}")
    _require(
        "qualified_nonpublication_1epoch" in matrix["capacity_matched_variant"].get("qualification_status", ""),
        "matrix capacity-matched qualification missing",
    )
    _require(fairness["observed"]["qualification_candidate"]["qualification_status"] == "qualified_nonpublication_rehearsal", "fairness qualification missing")
    _require(protocol["variant_qualification"]["formal_execution_started"] is False, "protocol variant qualification opened formal G1")
    _require(protocol["evidence_boundary"]["G1_scientific_ready"] is False, "unresolved variants must keep scientific readiness closed")
    _require(RECEIPT_PATH.exists() and len(_sha256(RECEIPT_PATH)) == 64, "qualification receipt is not readable")
    semantic = _load(SEMANTIC_ANCHOR_PATH)
    _require(semantic.get("status") == "FAIL_CLOSED", "semantic anchor must remain fail-closed")
    _require(semantic.get("execution_role") == "compatibility_audit", "semantic anchor role drifted")
    _require(semantic.get("v7", {}).get("code_commit", "").__len__() == 40, "semantic anchor commit is not pinned")
    equivalence = semantic.get("equivalence", {})
    _require(equivalence.get("prepared_inputs", {}).get("graphs", {}).get("exact") is True, "semantic graph equivalence missing")
    _require(equivalence.get("steps", {}).get("gradients", {}).get("exact") is True, "semantic gradient equivalence missing")
    _require(equivalence.get("steps", {}).get("parameters", {}).get("exact") is True, "semantic parameter equivalence missing")
    failure = semantic.get("failing_difference", {})
    _require(failure.get("new_tolerance_registered") is False, "semantic anchor introduced an unregistered tolerance")
    _require(float(failure.get("max_abs", 0.0)) > 0.0, "semantic anchor failure evidence missing")
    _require(SEMANTIC_ANCHOR_PATH.exists() and len(_sha256(SEMANTIC_ANCHOR_PATH)) == 64, "semantic anchor receipt is not readable")
    profiling = _load(PROFILING_PATH)
    _require(profiling.get("status") == "COMPLETE_nonpublication_instrumentation", "synchronized profiling status drifted")
    _require(profiling.get("timing_boundary", {}).get("step_end", "").startswith("after block_until_ready"), "profiling is not synchronized")
    _require(profiling.get("observations", {}).get("performance_claim") is False, "profiling performance claim opened")
    _require(profiling.get("observations", {}).get("scientific_evidence_eligible") is False, "profiling scientific evidence opened")
    for name in ("Full", "vanilla_RIGNO"):
        row = profiling.get("runs", {}).get(name, {})
        _require(row.get("compile_count") == 32, f"profiling compile count missing: {name}")
        _require(row.get("checkpoint_round_trip") is True, f"profiling checkpoint round-trip missing: {name}")
    _require(PROFILING_PATH.exists() and len(_sha256(PROFILING_PATH)) == 64, "profiling receipt is not readable")
    closeout = _load(CLOSEOUT_PATH)
    _require(closeout.get("status") == "FAIL_CLOSED_pre_G1", "scientific closeout must remain fail-closed")
    _require(closeout.get("readiness", {}).get("V7_G1_SCIENTIFIC_READY") == "FAIL_CLOSED", "scientific readiness was opened")
    _require(closeout.get("scientific_protocol", {}).get("formal_matrix", {}).get("planned_runs") == 21, "formal matrix run count drifted")
    _require(closeout.get("scientific_protocol", {}).get("formal_matrix", {}).get("formal_execution_started") is False, "formal G1 execution started")
    _require(closeout.get("gates", {}).get("full_v6_v7_semantic_anchor", {}).get("new_tolerance_registered") is False, "closeout registered an unapproved semantic tolerance")
    _require(CLOSEOUT_PATH.exists() and len(_sha256(CLOSEOUT_PATH)) == 64, "scientific closeout receipt is not readable")
    print("V7 variant qualification: PASS (supported variants); unresolved variants remain fail-closed")


if __name__ == "__main__":
    main()
