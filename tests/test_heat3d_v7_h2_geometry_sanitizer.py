import json
import tempfile
import unittest
from pathlib import Path

from rigno.heat3d_v7_h2_geometry_sanitizer import (
    assert_geometry_only_output,
    guard_geometry_input_paths,
    sanitize_json_file,
    sanitize_payload,
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

    def test_direct_geometry_path_guard_is_fail_closed(self) -> None:
        guard_geometry_input_paths(["/tmp/coords.npy", "/tmp/support.npz", "/tmp/graph_config.json"])
        with self.assertRaises(ValueError):
            guard_geometry_input_paths(["/tmp/temperature.npy"])
        with self.assertRaises(ValueError):
            guard_geometry_input_paths(["/tmp/prediction_rows.json"])


if __name__ == "__main__":
    unittest.main()
