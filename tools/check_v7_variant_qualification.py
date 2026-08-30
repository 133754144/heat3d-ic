"""Fail-closed validation for pre-G1 variant qualification.

This checker validates implementation qualification and compatibility evidence.
It deliberately does not open the formal multi-seed G1 matrix.
"""

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
        receipt.get("schema_version") == "heat3d_v7_g1_variant_qualification_receipt_v3",
        "variant qualification receipt schema drifted",
    )
    _require(len(str(receipt.get("code_commit", ""))) == 40, "qualification code commit is not pinned")
    _require(receipt.get("execution_role") == "variant_qualification", "qualification role drifted")
    for key in ("publication_evidence", "g1_formal", "scientific_evidence_eligible"):
        _require(receipt.get(key) is False, f"qualification evidence guard drifted: {key}")
    policy = receipt.get("qualification_policy", {})
    _require(policy.get("epochs") == 1 and policy.get("seed") == 0, "qualification budget/seed drifted")
    _require(
        policy.get("role") == "nonpublication_variant_qualification",
        "qualification role contract drifted",
    )
    _require(
        policy.get("modified_variants") == ["layout_agnostic_stratified_support", "no_film"],
        "modified qualification variant set drifted",
    )
    _require(
        policy.get("not_requalified_variants")
        == [
            "Full",
            "vanilla_RIGNO",
            "vanilla_RIGNO_capacity_matched",
            "cv_only_support",
            "physics_scale_only",
        ],
        "carried-forward qualification set drifted",
    )
    for key in ("formal_g1_multi_seed_started", "new_scientific_result", "solver", "new_data", "full_v6_evidence_modified"):
        _require(policy.get(key) is False, f"qualification prohibited-action guard drifted: {key}")

    variants = receipt.get("variants", {})
    required = {
        "layout_agnostic_stratified_support",
        "cv_only_support",
        "no_film",
        "physics_scale_only",
        "vanilla_RIGNO_capacity_matched",
    }
    _require(required <= set(variants), "not all registered ablation variants are qualified")
    for name in sorted(required):
        row = variants[name]
        _require(row.get("status") == "COMPLETE", f"missing completed qualification: {name}")
        _require(row.get("observation_only") is True, f"qualification must be observation-only: {name}")
        _require(len(str(row.get("receipt_sha256", ""))) == 64, f"missing source receipt SHA: {name}")
    _require(
        variants["layout_agnostic_stratified_support"].get("provider_id") == "generic_stratified_v2",
        "generic support provider binding drifted",
    )
    generic = variants["layout_agnostic_stratified_support"]
    _require(
        generic.get("implementation", "").startswith("generic support"),
        "generic support implementation binding drifted",
    )
    generic_determinism = generic.get("support_determinism", {})
    for key in ("deterministic_reproduction", "label_independent", "sha256_bound"):
        _require(generic_determinism.get(key) is True, f"generic support evidence missing: {key}")
    _require(
        int(generic_determinism.get("sample_count", 0)) > 0,
        "generic support sample SHA evidence is missing",
    )
    _require(
        variants["cv_only_support"].get("provider_id") == "cv_only_v1",
        "CV-only support provider binding drifted",
    )
    _require(
        variants["no_film"].get("implementation", "").startswith("global_context_mode=none"),
        "no_film does not use the one-field FiLM delta",
    )
    _require(
        variants["no_film"].get("single_delta") == ["global_context_mode: film -> none"],
        "no_film single-delta audit missing",
    )
    _require(
        variants["no_film"].get("context_feature_dim") == 24
        and variants["no_film"].get("context_feature_names_preserved") is True
        and variants["no_film"].get("scale_semantics_preserved") is True,
        "no_film retained context/scale semantics are not bound",
    )
    _require(
        variants["physics_scale_only"].get("implementation", "").startswith("native_shape_scale"),
        "physics_scale_only is not the native shape-scale route",
    )
    _require(
        "learned residual correction disabled" in variants["physics_scale_only"].get("implementation", ""),
        "physics_scale_only does not preserve physics-only scale semantics",
    )
    capacity = variants["vanilla_RIGNO_capacity_matched"]
    _require(
        capacity.get("node_latent_size") == 100 and capacity.get("edge_latent_size") == 100,
        "capacity-matched Vanilla width drifted",
    )

    _require(matrix.get("schema_version") == "heat3d_v7_g1_variant_execution_matrix_v5", "variant matrix schema drifted")
    _require(matrix.get("formal_execution_started") is False, "formal G1 execution opened")
    expected_ids = {
        "V7-G1-Full-P1i",
        "V7-G1-Full-P1i:vanilla-RIGNO",
        "V7-G1-Full-P1i:layout-agnostic-stratified-support",
        "V7-G1-Full-P1i:cv-only-support",
        "V7-G1-Full-P1i:no-film",
        "V7-G1-Full-P1i:physics-scale-only",
        "V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched",
    }
    matrix_ids = {row.get("experiment_id") for row in matrix.get("base_variants", [])}
    matrix_ids.add(matrix.get("capacity_matched_variant", {}).get("experiment_id"))
    _require(matrix_ids == expected_ids, "variant matrix IDs drifted")
    physics_scale_only = next(row for row in matrix["base_variants"] if row.get("variant") == "physics_scale_only")
    _require(
        "physics_scale_only" in physics_scale_only.get("qualification_status", "")
        or "physics scale only" in str(variants["physics_scale_only"].get("implementation", "")),
        "matrix physics_scale_only semantics drifted",
    )
    _require(
        matrix["capacity_matched_variant"].get("qualification_status") == "qualified_nonpublication_1epoch; width_100",
        "capacity-matched qualification binding drifted",
    )
    _require(
        fairness["observed"]["qualification_candidate"]["qualification_status"] == "qualified_nonpublication_rehearsal",
        "fairness qualification missing",
    )
    _require(protocol["formal_matrix"]["formal_execution_started"] is False, "protocol opened formal G1")

    semantic = _load(SEMANTIC_ANCHOR_PATH)
    _require(semantic.get("status") == "PASS", "Full V6/V7 semantic anchor is not PASS")
    _require(semantic.get("execution_role") == "compatibility_audit", "semantic anchor role drifted")
    _require(len(str(semantic.get("v7", {}).get("code_commit", ""))) == 40, "semantic anchor commit is not pinned")
    equivalence = semantic.get("equivalence", {})
    _require(equivalence.get("prepared_inputs", {}).get("graphs", {}).get("exact") is True, "semantic graph equivalence missing")
    _require(equivalence.get("steps", {}).get("gradients", {}).get("exact") is True, "semantic gradient equivalence missing")
    _require(equivalence.get("steps", {}).get("parameters", {}).get("exact") is True, "semantic parameter equivalence missing")
    _require(equivalence.get("steps", {}).get("common_loss_components", {}).get("exact") is True, "semantic loss-component equivalence missing")
    _require(semantic.get("policy", {}).get("new_tolerance_created") is False, "semantic anchor introduced an unregistered tolerance")

    profiling = _load(PROFILING_PATH)
    _require(profiling.get("status") == "COMPLETE_nonpublication_instrumentation", "synchronized profiling status drifted")
    _require(profiling.get("timing_boundary", {}).get("step_end", "").startswith("after block_until_ready"), "profiling is not synchronized")
    _require(profiling.get("observations", {}).get("performance_claim") is False, "profiling performance claim opened")
    _require(profiling.get("observations", {}).get("scientific_evidence_eligible") is False, "profiling scientific evidence opened")
    _require(PROFILING_PATH.exists() and len(_sha256(PROFILING_PATH)) == 64, "profiling receipt is not readable")
    _require(RECEIPT_PATH.exists() and len(_sha256(RECEIPT_PATH)) == 64, "qualification receipt is not readable")
    _require(SEMANTIC_ANCHOR_PATH.exists() and len(_sha256(SEMANTIC_ANCHOR_PATH)) == 64, "semantic anchor receipt is not readable")
    _require(CLOSEOUT_PATH.exists() and len(_sha256(CLOSEOUT_PATH)) == 64, "scientific closeout receipt is not readable")
    print("V7 variant qualification: PASS (all registered ablations qualified; formal G1 remains closed)")


if __name__ == "__main__":
    main()
