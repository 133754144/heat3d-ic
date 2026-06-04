"""Text-only smoke for Heat3D v2 training-time optimization audit."""

from __future__ import annotations

from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
RUNNER = REPO_DIR / "scripts" / "run_heat3d_v1_medium_controlled_training_export.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    checks = {
        "train_metrics_schedule_cli": "--train-metrics-schedule" in text,
        "global_grad_norm": "_global_norm(grads)" in text,
        "optax_clip_by_global_norm": "optax.clip_by_global_norm" in text,
        "profile_block_until_ready": "block_until_ready" in text,
    }
    predict_index = text.find("predictions = _predict_temperatures")
    save_index = text.find("if args.save_predictions:")
    checks["predict_before_save_flag"] = (
        predict_index != -1 and save_index != -1 and predict_index < save_index
    )

    failed = [name for name, ok in checks.items() if not ok]
    print("Heat3D v2 training-time optimization audit summary:")
    for name, ok in sorted(checks.items()):
        print(f"  {name}: {'FOUND' if ok else 'MISSING'}")
    if failed:
        raise AssertionError(f"missing expected audit patterns: {failed}")
    print("Heat3D v2 training-time optimization audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
