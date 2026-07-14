#!/usr/bin/env python3
"""Build the reproducible Gate-5 final closeout and error attribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONFIGS = {
    "B0": "V4P5_04_local_bypass_global_film",
    "N0": "V4P5_05_native_physics_only",
    "N1": "V4P5_06_native_pooled_latent",
    "N3": "V4P5_07_native_pooled_latent_global_film",
}
ROLES = (
    "valid_iid",
    "test_iid",
    "hard_train_holdout",
    "hard_challenge_valid",
    "hard_challenge_test",
)
CLEAN_ROLES = ROLES[:2]
HARD_ROLES = ROLES[2:]
METRICS = (
    "point_global_relative_rmse_pct",
    "sample_first_cv_relative_rmse_pct",
    "raw_cv_weighted_rmse_K",
    "amplitude_ratio",
    "spatial_correlation",
    "hotspot_cv_weighted_rmse_K",
    "top5_cv_weighted_rmse_K",
    "strong_q_cv_weighted_rmse_K",
    "low_deltaT_background_bias_K",
    "low_deltaT_background_rmse_K",
    "low_deltaT_background_over_ratio",
    "shape_cv_rmse",
    "scale_log_rmse",
    "legacy_normalized_valid_base_mse",
)
NATIVE_METRICS = (
    "joint_relative_rmse_pct",
    "oracle_scale_relative_rmse_pct",
    "oracle_shape_relative_rmse_pct",
    "physics_scale_relative_rmse_pct",
)
FEATURES = {
    "total_power_W": "P_operator_W",
    "source_concentration": "source_concentration",
    "q_weighted_local_kz_W_mK": "q_weighted_local_kz_W_mK",
    "q_weighted_inverse_kz_mK_W": "q_weighted_inverse_kz_mK_W",
    "top_h_W_m2K": "log_top_h_W_m2K",
    "anisotropy_xy_over_z": "anisotropy_xy_over_z",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in CONFIGS:
        parser.add_argument(f"--{label.lower()}-evaluation", type=Path, required=True)
    parser.add_argument("--output-closeout-json", type=Path, required=True)
    parser.add_argument("--output-closeout-md", type=Path, required=True)
    parser.add_argument("--output-diagnostic-json", type=Path, required=True)
    parser.add_argument("--output-diagnostic-md", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    result = {metric: _finite(report[metric]) for metric in METRICS}
    for optional in ("native_shape_scale", "film_modulation"):
        if optional in report:
            result[optional] = {
                key: _finite(value) if key != "sample_count" else int(value)
                for key, value in report[optional].items()
            }
    result["sample_count"] = int(report["sample_count"])
    return result


def _mean_reports(run: Mapping[str, Any], checkpoint: str, roles: Sequence[str], field: str) -> float:
    return float(np.mean([_finite(run["reports"][checkpoint][role][field]) for role in roles]))


def _native_clean(payload: Mapping[str, Any], checkpoint: str = "best") -> dict[str, float]:
    values: dict[str, float] = {}
    for field in NATIVE_METRICS:
        values[field] = float(np.mean([
            _finite(payload["reports"][checkpoint][role]["native_shape_scale"][field])
            for role in CLEAN_ROLES
        ]))
    values["shape_cv_rmse"] = float(np.mean([
        _finite(payload["reports"][checkpoint][role]["shape_cv_rmse"])
        for role in CLEAN_ROLES
    ]))
    values["scale_log_rmse"] = float(np.mean([
        _finite(payload["reports"][checkpoint][role]["scale_log_rmse"])
        for role in CLEAN_ROLES
    ]))
    values["amplitude_ratio"] = float(np.mean([
        _finite(payload["reports"][checkpoint][role]["amplitude_ratio"])
        for role in CLEAN_ROLES
    ]))
    values["amplitude_ratio_absolute_error"] = abs(values["amplitude_ratio"] - 1.0)
    values["scale_replacement_gain_pp"] = (
        values["joint_relative_rmse_pct"] - values["oracle_scale_relative_rmse_pct"]
    )
    values["shape_replacement_gain_pp"] = (
        values["joint_relative_rmse_pct"] - values["oracle_shape_relative_rmse_pct"]
    )
    return values


def _mechanism(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    n1 = _native_clean(payloads["N1"])
    n3 = _native_clean(payloads["N3"])
    delta = {key: n1[key] - n3[key] for key in n1}
    shape_evidence = n1["oracle_scale_relative_rmse_pct"] - n3["oracle_scale_relative_rmse_pct"]
    scale_evidence = n1["oracle_shape_relative_rmse_pct"] - n3["oracle_shape_relative_rmse_pct"]
    if shape_evidence > 0.0 and scale_evidence > 0.0:
        dominant = max(abs(shape_evidence), abs(scale_evidence))
        smaller = min(abs(shape_evidence), abs(scale_evidence))
        classification = "joint_path" if smaller >= 0.5 * dominant else (
            "shape_dominant" if shape_evidence > scale_evidence else "scale_dominant"
        )
    elif shape_evidence > 0.0:
        classification = "shape_dominant"
    elif scale_evidence > 0.0:
        classification = "scale_dominant"
    else:
        classification = "no_clean_component_gain"
    film = {
        role: {
            key: _finite(value)
            for key, value in payloads["N3"]["reports"]["best"][role]["film_modulation"].items()
            if key != "sample_count"
        }
        for role in CLEAN_ROLES
    }
    return {
        "basis": "MSE-best, arithmetic mean of valid_iid and test_iid aggregate metrics",
        "classification": classification,
        "n1": n1,
        "n3": n3,
        "n1_minus_n3": delta,
        "oracle_interpretation": {
            "shape_path_evidence_pp": shape_evidence,
            "scale_path_evidence_pp": scale_evidence,
            "oracle_scale": "replaces predicted scale and leaves shape error",
            "oracle_shape": "replaces predicted shape and leaves scale error",
        },
        "n3_film_modulation": film,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation arrays differ or are too small")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _rows(payload: Mapping[str, Any], checkpoint: str, roles: Sequence[str]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for role in roles
        for row in payload["reports"][checkpoint][role]["per_sample"]
    ]


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample ID in diagnostic population")
    return result


def _tail(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    clean_rows = {label: _rows(payload, "best", CLEAN_ROLES) for label, payload in payloads.items()}
    clean_maps = {label: _row_map(rows) for label, rows in clean_rows.items()}
    ids = sorted(clean_maps["B0"])
    if any(sorted(rows) != ids for rows in clean_maps.values()):
        raise ValueError("clean sample IDs differ across models")
    for sample_id in ids:
        references = [clean_maps[label][sample_id] for label in CONFIGS]
        true_sums = [_finite(row["point_true_squared_sum"]) for row in references]
        true_scales = [_finite(row["true_scale_cv_rms_K"]) for row in references]
        if not np.allclose(true_sums, true_sums[0]) or not np.allclose(true_scales, true_scales[0]):
            raise ValueError(f"target-derived fields differ for {sample_id}")

    true_scale = np.asarray([
        _finite(clean_maps["B0"][sample_id]["true_scale_cv_rms_K"]) for sample_id in ids
    ])
    edges = np.quantile(true_scale, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bins: list[dict[str, Any]] = []
    for index in range(4):
        selected = [
            sample_id for sample_id, scale in zip(ids, true_scale, strict=True)
            if edges[index] < scale <= edges[index + 1]
        ]
        row: dict[str, Any] = {
            "bin": index + 1,
            "true_cv_rms_deltaT_K_lower": float(edges[index]),
            "true_cv_rms_deltaT_K_upper": float(edges[index + 1]),
            "sample_count": len(selected),
            "models": {},
        }
        for label in ("B0", "N1", "N3"):
            model_rows = [clean_maps[label][sample_id] for sample_id in selected]
            error = sum(_finite(item["point_error_squared_sum"]) for item in model_rows)
            truth = sum(_finite(item["point_true_squared_sum"]) for item in model_rows)
            total_error = sum(_finite(item["point_error_squared_sum"]) for item in clean_rows[label])
            row["models"][label] = {
                "point_global_relative_rmse_pct": 100.0 * math.sqrt(error / truth),
                "sample_first_cv_relative_rmse_pct": 100.0 * float(np.mean([
                    _finite(item["sample_cv_relative_rmse"]) for item in model_rows
                ])),
                "total_squared_error_contribution": error / total_error,
            }
        bins.append(row)

    models: dict[str, Any] = {}
    for label, rows in clean_rows.items():
        total = sum(_finite(row["point_error_squared_sum"]) for row in rows)
        ordered = sorted(rows, key=lambda row: _finite(row["point_error_squared_sum"]), reverse=True)
        features: dict[str, Any] = {}
        log_error = np.log1p(np.asarray([_finite(row["point_error_squared_sum"]) for row in rows]))
        relative = np.asarray([_finite(row["sample_cv_relative_rmse"]) for row in rows])
        for output_name, source_name in FEATURES.items():
            values = np.asarray([
                _finite(row["attribution_context"][source_name]) for row in rows
            ])
            if output_name == "top_h_W_m2K":
                values = np.exp(values)
            features[output_name] = {
                "pearson_with_log1p_sample_squared_error": _correlation(values, log_error),
                "spearman_with_sample_squared_error": _correlation(_rank(values), _rank(log_error)),
                "spearman_with_sample_cv_relative_rmse": _correlation(_rank(values), _rank(relative)),
            }
        models[label] = {
            "total_point_squared_error_K2": total,
            "top5_cumulative_contribution": sum(
                _finite(row["point_error_squared_sum"]) for row in ordered[:5]
            ) / total,
            "top10_cumulative_contribution": sum(
                _finite(row["point_error_squared_sum"]) for row in ordered[:10]
            ) / total,
            "top10_samples": [
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "contribution": _finite(row["point_error_squared_sum"]) / total,
                    "true_cv_rms_deltaT_K": _finite(row["true_scale_cv_rms_K"]),
                    "sample_cv_relative_rmse_pct": 100.0 * _finite(row["sample_cv_relative_rmse"]),
                }
                for row in ordered[:10]
            ],
            "feature_correlations": features,
        }

    gaps = {}
    for label in CONFIGS:
        point = float(np.mean([
            _finite(payloads[label]["reports"]["best"][role]["point_global_relative_rmse_pct"])
            for role in CLEAN_ROLES
        ]))
        sample = float(np.mean([
            _finite(payloads[label]["reports"]["best"][role]["sample_first_cv_relative_rmse_pct"])
            for role in CLEAN_ROLES
        ]))
        gaps[label] = {
            "clean_mean_point_global_pct": point,
            "clean_mean_sample_first_pct": sample,
            "point_minus_sample_first_pp": point - sample,
            "highest_true_scale_quartile_error_contribution": bins[-1]["models"].get(label, {}).get(
                "total_squared_error_contribution"
            ),
        }
    strongest_relations = {}
    for label, model in models.items():
        ranked = sorted(
            (
                {
                    "feature": feature,
                    "spearman_with_sample_squared_error": values[
                        "spearman_with_sample_squared_error"
                    ],
                    "spearman_with_sample_cv_relative_rmse": values[
                        "spearman_with_sample_cv_relative_rmse"
                    ],
                }
                for feature, values in model["feature_correlations"].items()
            ),
            key=lambda row: abs(row["spearman_with_sample_squared_error"]),
            reverse=True,
        )
        strongest_relations[label] = ranked
    compared_labels = ("B0", "N1", "N3")
    mean_high_tail_share = float(np.mean([
        gaps[label]["highest_true_scale_quartile_error_contribution"]
        for label in compared_labels
    ]))
    mean_gap = float(np.mean([
        gaps[label]["point_minus_sample_first_pp"] for label in compared_labels
    ]))
    point_sample_conclusion = (
        "point-global 偏高主要来自高温升样本的误差集中；这些样本在 true-energy 加权的 "
        "point-global 中权重大于不加权的 sample-first 均值"
        if mean_gap > 0.0 and mean_high_tail_share > 0.25
        else "point-global 与 sample-first 的差异不能主要由高温升尾部解释"
    )
    return {
        "population": "MSE-best valid_iid + test_iid; hard roles remain separately reported",
        "sample_count": len(ids),
        "squared_error_concentration": models,
        "true_cv_rms_deltaT_bins": bins,
        "point_global_vs_sample_first": gaps,
        "interpretation": {
            "strongest_relations_by_model": strongest_relations,
            "mean_highest_scale_quartile_error_contribution_B0_N1_N3": mean_high_tail_share,
            "mean_point_minus_sample_first_pp_B0_N1_N3": mean_gap,
            "point_global_vs_sample_first_conclusion": point_sample_conclusion,
        },
    }


def _core_runs(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    runs = {}
    for label, payload in payloads.items():
        runs[label] = {
            "config_id": CONFIGS[label],
            "training_git_commit": payload.get("training_git_commit"),
            "evaluator_git_commit": payload["evaluator_git_commit"],
            "registry_git_commit": payload["registry_git_commit"],
            "checkpoint_epochs": {
                name: int(payload["checkpoint_metadata"][name]["epoch"])
                for name in ("best", "final")
            },
            "checkpoint_sha256": {
                name: payload["checkpoint_metadata"][name]["sha256"]
                for name in ("best", "final")
            },
            "reports": {
                checkpoint: {
                    role: _compact_report(payload["reports"][checkpoint][role])
                    for role in ROLES
                }
                for checkpoint in ("best", "final")
            },
        }
    return runs


def _improvements(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for reference in ("B0", "N1"):
        comparisons = {}
        for checkpoint in ("best", "final"):
            comparisons[checkpoint] = {}
            for group, roles in (("clean", CLEAN_ROLES), ("hard", HARD_ROLES)):
                comparisons[checkpoint][group] = {
                    field + "_improvement": (
                        _mean_reports(runs[reference], checkpoint, roles, field)
                        - _mean_reports(runs["N3"], checkpoint, roles, field)
                    )
                    for field in (
                        "point_global_relative_rmse_pct",
                        "sample_first_cv_relative_rmse_pct",
                        "raw_cv_weighted_rmse_K",
                        "shape_cv_rmse",
                        "scale_log_rmse",
                    )
                }
        result[f"N3_vs_{reference}"] = comparisons
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _closeout_markdown(closeout: Mapping[str, Any]) -> str:
    lines = [
        "# V5 Gate 5 final closeout",
        "",
        f"统一 evaluator commit：`{closeout['evaluator_git_commit']}`。best 为最低 `valid_base_mse`；final 为 epoch 600。",
        "",
        "| Model | best epoch | valid point/sample % | test point/sample % | valid/test raw K | final valid/test point % | <20% |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, run in closeout["runs"].items():
        bv = run["reports"]["best"]["valid_iid"]
        bt = run["reports"]["best"]["test_iid"]
        fv = run["reports"]["final"]["valid_iid"]
        ft = run["reports"]["final"]["test_iid"]
        lines.append(
            f"| {label} | {run['checkpoint_epochs']['best']} | "
            f"{bv['point_global_relative_rmse_pct']:.3f}/{bv['sample_first_cv_relative_rmse_pct']:.3f} | "
            f"{bt['point_global_relative_rmse_pct']:.3f}/{bt['sample_first_cv_relative_rmse_pct']:.3f} | "
            f"{bv['raw_cv_weighted_rmse_K']:.4f}/{bt['raw_cv_weighted_rmse_K']:.4f} | "
            f"{fv['point_global_relative_rmse_pct']:.3f}/{ft['point_global_relative_rmse_pct']:.3f} | "
            f"{'pass' if closeout['threshold_assessment'][label] else 'fail'} |"
        )
    lines += [
        "",
        "全部角色、全部指标、best/final checkpoint SHA 与三类 commit 见 JSON。hard roles 仅作冻结后的描述性报告。",
        "",
        "## Hard report-only",
        "",
        "| Model | Role | best point/sample % | best raw K | final point/sample % | final raw K |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, run in closeout["runs"].items():
        for role in HARD_ROLES:
            best = run["reports"]["best"][role]
            final = run["reports"]["final"][role]
            lines.append(
                f"| {label} | {role} | {best['point_global_relative_rmse_pct']:.3f}/"
                f"{best['sample_first_cv_relative_rmse_pct']:.3f} | {best['raw_cv_weighted_rmse_K']:.4f} | "
                f"{final['point_global_relative_rmse_pct']:.3f}/"
                f"{final['sample_first_cv_relative_rmse_pct']:.3f} | {final['raw_cv_weighted_rmse_K']:.4f} |"
            )
    lines += [
        "",
        "## N3 improvement (reference minus N3; positive is better)",
        "",
        "| Reference | checkpoint | population | point pp | sample-first pp | raw K |",
        "|---|---|---|---:|---:|---:|",
    ]
    for reference in ("B0", "N1"):
        comparison = closeout["n3_improvements"][f"N3_vs_{reference}"]
        for checkpoint in ("best", "final"):
            for population in ("clean", "hard"):
                row = comparison[checkpoint][population]
                lines.append(
                    f"| {reference} | {checkpoint} | {population} | "
                    f"{row['point_global_relative_rmse_pct_improvement']:.4f} | "
                    f"{row['sample_first_cv_relative_rmse_pct_improvement']:.4f} | "
                    f"{row['raw_cv_weighted_rmse_K_improvement']:.4f} |"
                )
    lines += [
        "",
        f"最终候选结论：{closeout['final_candidate_conclusion']}",
    ]
    return "\n".join(lines) + "\n"


def _diagnostic_markdown(diagnostic: Mapping[str, Any]) -> str:
    mechanism = diagnostic["n1_to_n3_mechanism"]
    lines = [
        "# V5 Gate 5 error attribution",
        "",
        "## N1 to N3",
        "",
        f"归因：`{mechanism['classification']}`。下表为 best clean valid/test 均值。",
        "",
        "| model | joint % | shape CV-RMSE | scale log-RMSE | amplitude | oracle-scale % | oracle-shape % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("n1", "n3"):
        row = mechanism[label]
        lines.append(
            f"| {label.upper()} | {row['joint_relative_rmse_pct']:.4f} | {row['shape_cv_rmse']:.4f} | "
            f"{row['scale_log_rmse']:.4f} | {row['amplitude_ratio']:.4f} | "
            f"{row['oracle_scale_relative_rmse_pct']:.4f} | {row['oracle_shape_relative_rmse_pct']:.4f} |"
        )
    lines += [
        "",
        f"N3 FiLM 幅值：valid gamma/beta mean-abs="
        f"{mechanism['n3_film_modulation']['valid_iid']['film_gamma_mean_abs']:.4f}/"
        f"{mechanism['n3_film_modulation']['valid_iid']['film_beta_mean_abs']:.4f}；"
        f"test={mechanism['n3_film_modulation']['test_iid']['film_gamma_mean_abs']:.4f}/"
        f"{mechanism['n3_film_modulation']['test_iid']['film_beta_mean_abs']:.4f}。",
        "",
        "## High DeltaT tail",
        "",
        "| model | top-5 error share | top-10 error share | point minus sample-first pp | high-scale quartile error share |",
        "|---|---:|---:|---:|---:|",
    ]
    tail = diagnostic["high_deltaT_tail"]
    for label in CONFIGS:
        concentration = tail["squared_error_concentration"][label]
        gap = tail["point_global_vs_sample_first"][label]
        lines.append(
            f"| {label} | {concentration['top5_cumulative_contribution']:.4f} | "
            f"{concentration['top10_cumulative_contribution']:.4f} | "
            f"{gap['point_minus_sample_first_pp']:.4f} | "
            f"{_fmt(gap['highest_true_scale_quartile_error_contribution'])} |"
        )
    lines += [
        "",
        "分箱、top-10 样本及 power/source/conductivity/top-h/anisotropy 相关系数见 diagnostic JSON。",
        "",
        f"point-global 与 sample-first：{tail['interpretation']['point_global_vs_sample_first_conclusion']}。",
        "",
        "| model | strongest squared-error relation | Spearman rho |",
        "|---|---|---:|",
    ]
    for label in CONFIGS:
        strongest = tail["interpretation"]["strongest_relations_by_model"][label][0]
        lines.append(
            f"| {label} | {strongest['feature']} | "
            f"{strongest['spearman_with_sample_squared_error']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    payloads = {
        label: _read(getattr(args, f"{label.lower()}_evaluation"))
        for label in CONFIGS
    }
    for label, payload in payloads.items():
        if payload.get("config_id") != CONFIGS[label]:
            raise ValueError(f"{label}: config_id mismatch")
    evaluator_commits = {payload["evaluator_git_commit"] for payload in payloads.values()}
    if len(evaluator_commits) != 1:
        raise ValueError(f"mixed evaluator commits: {evaluator_commits}")
    runs = _core_runs(payloads)
    threshold = {
        label: (
            run["reports"]["best"]["valid_iid"]["point_global_relative_rmse_pct"] < 20.0
            and run["reports"]["best"]["test_iid"]["point_global_relative_rmse_pct"] < 20.0
        )
        for label, run in runs.items()
    }
    clean_score = {
        label: _mean_reports(run, "best", CLEAN_ROLES, "point_global_relative_rmse_pct")
        for label, run in runs.items()
    }
    winner = min(clean_score, key=clean_score.get)
    conclusion = (
        f"{winner} 的 MSE-best clean point-global 均值最低；"
        + ("通过" if threshold[winner] else "未通过")
        + "冻结的 valid/test 均 <20% 可信门槛。"
    )
    closeout = {
        "schema_version": "heat3d_v5_gate5_final_closeout_v1",
        "status": "complete_no_followup_training_started",
        "evaluator_git_commit": next(iter(evaluator_commits)),
        "selection": "best=lowest valid_base_mse; final=epoch 600",
        "report_only_roles": ["test_iid", *HARD_ROLES],
        "runs": runs,
        "threshold_formula": "best valid_iid and test_iid point-global true-RMS relative RMSE both <20%",
        "threshold_assessment": threshold,
        "n3_improvements": _improvements(runs),
        "final_candidate_conclusion": conclusion,
        "next_phase_started": False,
    }
    diagnostic = {
        "schema_version": "heat3d_v5_gate5_final_error_attribution_v1",
        "status": "complete",
        "evaluator_git_commit": next(iter(evaluator_commits)),
        "n1_to_n3_mechanism": _mechanism(payloads),
        "high_deltaT_tail": _tail(payloads),
    }
    for path in (
        args.output_closeout_json,
        args.output_closeout_md,
        args.output_diagnostic_json,
        args.output_diagnostic_md,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_closeout_json.write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_closeout_md.write_text(_closeout_markdown(closeout), encoding="utf-8")
    args.output_diagnostic_json.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_diagnostic_md.write_text(_diagnostic_markdown(diagnostic), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "evaluator_git_commit": next(iter(evaluator_commits)),
        "winner": winner,
        "threshold": threshold,
        "mechanism": diagnostic["n1_to_n3_mechanism"]["classification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
