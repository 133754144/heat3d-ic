"""Fail-closed checks for the V7 G1 protocol and budget qualification control plane."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "heat3d_v7"


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
    budget = _load(CONTROL / "v7_g1_budget_qualification.json")
    epoch = _load(CONTROL / "v7_g1_epoch_budget_contract.json")
    fairness = _load(CONTROL / "v7_parameter_fairness_contract.json")
    prereg = _load(CONTROL / "v7_g1_statistical_preregistration.json")
    support = _load(CONTROL / "v7_support_artifact_freeze.json")
    seed_bundle = _load(CONTROL / "v7_g1_seed_bundle.json")
    registry = _load(CONTROL / "v7_experiment_registry.json")
    protocol_freeze = _load(ROOT / "docs" / "v7_g1_scientific_protocol_freeze.json")
    budget_decision = _load(ROOT / "docs" / "v7_g1_budget_decision_receipt.json")
    variant_matrix = _load(CONTROL / "v7_g1_variant_execution_matrix.json")

    proposed = epoch["proposed_budget"]
    for key, expected in {
        "epochs": 200,
        "warmup_epochs": 10,
        "base_lr": 5.0e-4,
        "min_lr": 5.0e-5,
        "cosine_horizon_epochs": 200,
        "optimizer": "adamw",
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "checkpoint_selection_metric": "sample_first_relative_rmse_pct",
    }.items():
        _require(proposed.get(key) == expected, f"e200 proposal drifted: {key}")
    _require(
        epoch["historical_reference"]["epochs"] == 600,
        "historical e600 reference missing",
    )
    _require(
        "not e200 evidence" in epoch["historical_reference"]["warning"],
        "e600/e200 warning missing",
    )
    qualification = epoch["qualification"]
    _require(qualification["required_seed"] == 0, "qualification seed drifted")
    _require(
        qualification["required_variants"] == ["Full", "vanilla_RIGNO"],
        "qualification variants drifted",
    )
    _require(
        qualification["publication_evidence"] is False
        and qualification["g1_formal"] is False,
        "qualification is not isolated from G1 evidence",
    )
    _require(prereg["seed_set"] == [0, 1, 2], "formal seed set drifted")
    _require(prereg["forbidden_roles"] == ["test_iid", "sealed"], "split denylist drifted")
    _require(seed_bundle["seed_set"] == [0, 1, 2], "synchronized seed bundle drifted")
    _require(
        seed_bundle["formal_execution_guard"]["multi_seed_started"] is False,
        "formal G1 multi-seed execution is already open",
    )
    _require(
        seed_bundle["synchronization"]["batch_build_seed"]["value"] == 0,
        "fixed B24 batch-build seed drifted",
    )
    _require(
        protocol_freeze["formal_matrix"]["formal_execution_started"] is False,
        "formal G1 execution must remain closed in protocol freeze",
    )
    _require(
        protocol_freeze["e200_optimization_contract"]["cosine_horizon_epochs"] == 200,
        "protocol freeze e200 horizon drifted",
    )
    _require(
        epoch["decision_values"]["G1_epoch_budget"] == 200
        and epoch["qualification"]["qualification_decision"] == "PASS_e200",
        "final e200 budget decision is not frozen",
    )
    _require(
        fairness["capacity_matched_trigger"]["relative_parameter_gap_threshold"] == 0.05,
        "capacity trigger drifted",
    )
    _require(support["training_support_is_frozen"] is True, "support freeze missing")
    _require(support["temperature_or_model_error_used"] is False, "support became label-dependent")

    expected_ids = {
        "V7-G1-Full-P1i",
        "V7-G1-Full-P1i:vanilla-RIGNO",
        "V7-G1-Full-P1i:generic-uniform-support",
        "V7-G1-Full-P1i:volume-only-support",
        "V7-G1-Full-P1i:no-context",
        "V7-G1-Full-P1i:no-scale",
        "V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched",
    }
    actual_ids = {
        str(row["experiment_id"])
        for row in registry.get("registered_runs", [])
        if str(row.get("experiment_id", "")).startswith("V7-G1-Full-P1i")
    }
    _require(actual_ids == expected_ids, "formal G1 variant matrix is not six base plus capacity-matched entry")
    _require(registry.get("formal_variant_count") == 7, "formal G1 variant count is not seven")
    _require(registry.get("formal_matrix", {}).get("run_count") == 21, "formal G1 run matrix is not 21")
    _require(len(variant_matrix.get("base_variants", [])) == 6, "variant execution matrix base count drifted")
    _require(
        variant_matrix.get("capacity_matched_variant", {}).get("experiment_id")
        == "V7-G1-Full-P1i:vanilla-RIGNO-capacity-matched",
        "capacity-matched variant execution binding drifted",
    )
    _require(variant_matrix.get("formal_execution_started") is False, "variant matrix opened formal G1")
    csv_path = CONTROL / "v7_experiment_registry.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_ids = {str(row["experiment_id"]) for row in csv.DictReader(stream)}
    _require(
        {"V7-G1-BudgetQual-e200-Full-seed0", "V7-G1-BudgetQual-e200-Vanilla-seed0"}
        <= csv_ids,
        "budget qualification IDs missing from CSV registry",
    )

    entrypoint = ROOT / "scripts" / "run_heat3d_v7_formal_p1i_training.py"
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    modules: list[str] = []
    private_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
            private_imports.extend(alias.name for alias in node.names if alias.name.startswith("_"))
    _require(not any(module.startswith("scripts") for module in modules), "training entrypoint imports scripts")
    _require(not any("smoke" in module.lower() or "development" in module.lower() for module in modules), "training entrypoint imports legacy module")
    _require(not private_imports, "training entrypoint imports private symbols")
    source = entrypoint.read_text(encoding="utf-8")
    _require("sys.path" not in source and "monkey_patch" not in source, "training entrypoint mutates runtime state")
    _require((ROOT / "rigno" / "heat3d_training" / "evaluation.py").exists(), "Level-A evaluator missing")
    _require("EvaluationCore" in (ROOT / "rigno" / "heat3d_training" / "evaluation.py").read_text(encoding="utf-8"), "Level-A evaluator is not bound to EvaluationCore")

    for artifact in support["artifacts"]:
        path = ROOT / artifact["path"]
        _require(path.exists(), f"support freeze artifact missing: {path}")
        _require(_sha256(path) == artifact["sha256"], f"support freeze SHA mismatch: {path}")

    qualification_ids = {
        str(row["experiment_id"]): row
        for row in registry.get("registered_runs", [])
        if str(row.get("experiment_id", "")).startswith("V7-G1-BudgetQual-")
    }
    _require(
        all(
            row.get("status") == "completed_nonpublication"
            and row.get("execution_started") is True
            and row.get("publication_evidence") is False
            and row.get("g1_formal") is False
            for row in qualification_ids.values()
        ),
        "budget qualification completion/provenance drifted",
    )
    _require(
        budget_decision["decision"]["G1_epoch_budget"] == 200
        and budget_decision["decision"]["formal_execution_started"] is False,
        "budget decision receipt opened formal G1",
    )
    _require(
        protocol_freeze["evidence_boundary"]["G1_scientific_ready"] is False,
        "unresolved variant providers must keep G1 scientific readiness closed",
    )
    print("V7 G1 protocol control plane: PASS")


if __name__ == "__main__":
    main()
