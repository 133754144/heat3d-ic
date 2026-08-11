#!/usr/bin/env python3
"""Check preregistered or completed P5-R resolution sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    require(protocol["status"] == "preregistered_before_execution", "protocol status")
    require(
        [(row["route"], row["resolution"]) for row in protocol["cells"]] == [
            ("native1024_reconstruction", 1024), ("B4096_reconstruction", 4096),
            ("B8192_reconstruction", 8192), ("E16384_reconstruction", 16384),
            ("E32768_reconstruction", 32768), ("E240825_direct", 240825),
        ],
        "registered cells",
    )
    role = protocol["role_contract"]
    require(not role["training"] and not role["test"] and not role["sealed"], "role boundary")
    require(protocol["timing_contract"]["accuracy_and_latency_same_execution"], "same execution")
    checked = False
    if args.json is not None:
        require(args.csv is not None and args.md is not None, "complete result paths")
        result = json.loads(args.json.read_text())
        require(result["status"] == "passed" and len(result["rows"]) == 6, "result rows")
        require(result["role_contract"] == role, "result role contract")
        for row in result["rows"]:
            require(row["accuracy_provenance"] == row["timing_provenance"], "same-execution provenance")
            require(float(row["matched_continuous_e2e_median_s"]) > 0, "finite timing")
            require(float(row["point_global_pct"]) > 0, "finite accuracy")
        with args.csv.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        require(len(csv_rows) == 6, "CSV rows")
        text = args.md.read_text()
        require("no old latency was joined to new accuracy" in text, "timing provenance note")
        if args.manifest is not None:
            require(args.raw_root is not None, "raw root required with manifest")
            manifest = json.loads(args.manifest.read_text())
            require(manifest["status"] == "completed_valid32", "manifest status")
            require(manifest["role_contract"]["accuracy_and_latency_same_execution"], "manifest timing role")
            for row in manifest["valid32_cells"]:
                path = args.raw_root / f"{row['route']}.json"
                require(path.is_file(), f"missing raw cell {path}")
                require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"raw SHA {path}")
        checked = True
    print(json.dumps({"p5r_protocol_checked": True, "result_checked": checked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
