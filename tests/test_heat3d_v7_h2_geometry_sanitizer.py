import json
import tempfile
import unittest
from pathlib import Path

from rigno.heat3d_v7_h2_geometry_sanitizer import (
    assert_geometry_only_output,
    assert_geometry_records_output,
    guard_geometry_input_paths,
    sanitize_json_file,
    sanitize_geometry_records,
    sanitize_payload,
)
from rigno.heat3d_v7_h2_governance_guard import (
    assert_gate_ab_input_paths,
    assert_gate_ab_output,
)


class GeometryOnlySanitizerTests(unittest.TestCase):
    def test_banned_fields_and_values_never_leak(self) -> None:
        payload = {
            "sample_id": "v6p1if1_0993",
            "geometry": {
                "coordinates_sha256": "coord-sha",
                "node_count": 1024,
                "accuracy": {"rmse": 1.25},
                "notes": "temperature prediction metric must disappear",
            },
            "support": {
                "support_indices_sha256": "support-sha",
                "selection_seed": 20260808,
                "target_count": 1024,
            },
            "graph": {
                "p2r_count": 3074,
                "r2r_count": 4075,
                "backend": "sparse_kdtree_v1",
                "loss": "forbidden",
            },
            "dependency": {"numpy": "2.4.2", "backend": "cpu"},
            "provenance": {"commit": "05b32ce", "notes": "prediction=never"},
            "metric": {"value": 999},
        }
        clean = sanitize_payload(payload)
        encoded = json.dumps(clean, sort_keys=True).lower()
        for token in ("accuracy", "metric", "rmse", "mae", "temperature", "prediction", "target", "loss"):
            self.assertNotIn(token, encoded)
        self.assertEqual(clean["sample_id"], "v6p1if1_0993")
        self.assertEqual(clean["geometry"]["node_count"], 1024)
        self.assertEqual(clean["graph"]["p2r_count"], 3074)
        assert_geometry_only_output(clean)

    def test_file_sanitizer_drops_mixed_content_without_logging_it(self) -> None:
        payload = {
            "sample_id": "v6p1if1_0993",
            "graph": {"final_p2r_count": 3074, "final_r2r_count": 4075},
            "accuracy": {"point_global_relative_rmse_pct": 3.4},
            "temperature": [300.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            clean = sanitize_json_file(path)
        self.assertEqual(clean["graph"]["final_p2r_count"], 3074)
        self.assertNotIn("accuracy", json.dumps(clean).lower())
        self.assertNotIn("temperature", json.dumps(clean).lower())

    def test_top_level_environment_fields_are_projected_to_dependency(self) -> None:
        clean = sanitize_payload({
            "python": "3.14.3",
            "numpy": "2.4.2",
            "platform": "Linux",
            "accuracy": {"value": 123},
        })
        self.assertEqual(clean["dependency"]["numpy"], "2.4.2")
        self.assertNotIn("accuracy", json.dumps(clean).lower())
        assert_geometry_only_output(clean)

    def test_nested_geometry_records_prune_banned_subtrees(self) -> None:
        clean = sanitize_geometry_records({
            "samples": [{
                "sample_id": "v6p1if1_0993",
                "graph": {
                    "final_p2r_count": 3074,
                    "final_r2r_count": 4075,
                },
                "accuracy": {
                    "final_p2r_count": 9999,
                    "rmse": 12.0,
                },
            }],
        })
        encoded = json.dumps(clean, sort_keys=True).lower()
        for token in ("accuracy", "metric", "rmse", "temperature", "prediction", "target", "loss"):
            self.assertNotIn(token, encoded)
        self.assertTrue(any(row.get("sample_id") == "v6p1if1_0993" for row in clean["records"]))
        self.assertTrue(any(row.get("final_p2r_count") == 3074 for row in clean["records"]))
        assert_geometry_records_output(clean)

    def test_direct_geometry_path_guard_is_fail_closed(self) -> None:
        guard_geometry_input_paths(["/tmp/coords.npy", "/tmp/support.npz", "/tmp/graph_config.json"])
        with self.assertRaises(ValueError):
            guard_geometry_input_paths(["/tmp/temperature.npy"])
        with self.assertRaises(ValueError):
            guard_geometry_input_paths(["/tmp/prediction_rows.json"])

    def test_gate_ab_input_roles_reject_model_and_result_paths(self) -> None:
        assert_gate_ab_input_paths({
            "geometry": ["/tmp/full_field_geometry.h5"],
            "support": ["/tmp/support_geometry.npz"],
            "config": ["/tmp/native_graph_config.json"],
        })
        for role, path in (
            ("geometry", "/tmp/truth_geometry.h5"),
            ("geometry", "/tmp/model_forward.h5"),
            ("support", "/tmp/checkpoint_support.pkl"),
            ("config", "/tmp/metric_config.json"),
        ):
            with self.assertRaises(ValueError):
                assert_gate_ab_input_paths({role: [path]})

    def test_gate_ab_output_allowlist_cannot_carry_banned_values(self) -> None:
        clean = sanitize_payload({
            "sample_id": "v6p1if1_0993",
            "geometry": {"node_count": 1024},
            "graph": {"final_p2r_count": 3074},
            "provenance": {"code_sha256": "abc"},
            "prediction": {"value": "must disappear"},
        })
        assert_gate_ab_output(clean)
        with self.assertRaises(ValueError):
            assert_gate_ab_output({
                "schema_version": "geometry_only_sanitized_v1",
                "sample_id": "v6p1if1_0993",
                "geometry": {"node_count": 1024},
                "model_forward": False,
            })


if __name__ == "__main__":
    unittest.main()
