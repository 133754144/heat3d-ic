import importlib.util
import json
from pathlib import Path

import torch


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


def test_gino_normalization_is_global_channelwise_not_node_indexed():
    module = _load_runner()
    train = []
    for offset in (0.0, 10.0):
        coords = torch.zeros((3, 3), dtype=torch.float32).numpy()
        features = (torch.arange(33, dtype=torch.float32).reshape(3, 11) + offset).numpy()
        target = (torch.arange(3, dtype=torch.float32).reshape(3, 1) + offset).numpy()
        train.append((coords, features, target, {}))
    stats = module.normalizers({"train": train})
    assert tuple(stats["gino_feature_mean"].shape) == (1, 1, 11)
    assert tuple(stats["gino_feature_std"].shape) == (1, 1, 11)
    assert tuple(stats["gino_y_mean"].shape) == (1, 1, 1)
    assert tuple(stats["gino_y_std"].shape) == (1, 1, 1)


def test_protocol_v3_freezes_scope_normalization_and_checkpoint_policy():
    path = ROOT / "configs" / "heat3d_v7" / "g2_formal_preparation_protocol_v3.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["hard_boundaries"]["ssh_or_devbox_used"] is False
    assert protocol["hard_boundaries"]["formal_g2_training_started"] is False
    assert protocol["model_scope"]["p1i_common_task"] == ["GINO", "Transolver"]
    assert protocol["model_scope"]["excluded_from_formal_experiments"] == [
        "Geo-FNO",
        "DeepOHeat-v2",
    ]
    assert protocol["normalization"]["GINO"]["forbidden"] == (
        "per-node-index target normalization"
    )
    assert protocol["GINO"]["radius"]["upstream_candidate"] == 0.033
    assert protocol["GINO"]["radius"]["provisional_formal"] == 0.15
    assert protocol["Transolver"]["architecture"]["parameter_count"] == 716737
    assert protocol["Transolver"]["information_contract"].startswith(
        "coords plus 11 physical features only"
    )
    assert protocol["checkpoint_selection_policy"]["primary"] == {
        "split": "valid_iid",
        "metric": "sample_first_relative_rmse_pct",
        "tie_break": "earliest_epoch",
        "selection": "per_run_within_model_and_seed",
        "selection_frequency": "every_completed_epoch",
        "no_cross_model_cross_seed_or_cross_variant_selection": True,
    }
