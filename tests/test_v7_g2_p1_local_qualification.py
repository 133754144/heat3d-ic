import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "run_v7_g2_p1_local_qualification.py"
    spec = importlib.util.spec_from_file_location("g2_p1_local_qualification", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_qualification_ids_fail_closed():
    module = _load_runner()
    assert len(module.TRAIN_IDS) == 4
    assert len(module.VALID_IDS) == 2
    assert set(module.TRAIN_IDS).isdisjoint(module.VALID_IDS)
    assert all("test" not in item and "sealed" not in item for item in module.TRAIN_IDS + module.VALID_IDS)
    assert "temperature.npy" not in module.REQUIRED_FILES


def test_protocol_v2_preserves_model_specific_training_semantics():
    path = ROOT / "configs" / "heat3d_v7" / "g2_p1_protocol_v2.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["hard_boundaries"]["formal_training_started"] is False
    assert protocol["hard_boundaries"]["test_iid_access"] is False
    assert protocol["hard_boundaries"]["sealed_access"] is False
    assert protocol["fairness"]["epochs_lr_and_loss_forced_common"] is False
    assert protocol["quantitative_comparison_policy"]["same_main_table"] == ["GINO", "Transolver"]
    assert protocol["models"]["GINO"]["optimizer"]["name"] == "AdamW"
    assert protocol["models"]["GINO"]["schedule"] == {
        "name": "StepLR",
        "step_size_epochs": 50,
        "gamma": 0.5,
    }
    assert protocol["models"]["Transolver"]["schedule"]["T_max_epochs"] == 500
    assert protocol["models"]["DeepOHeat"]["p1i_representability"] == (
        "P1i_direct_comparison_not_identifiable_without_algorithm_change"
    )
