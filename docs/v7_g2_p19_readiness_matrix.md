# V7 G2-P19 readiness matrix

## Closed

| gate | status | evidence |
|---|---|---|
| P14.1 valid-only evaluator | CLOSED | frozen valid128 U-v2 evaluator |
| P15 Heat3D e600×3 | CLOSED (conservative wording) | no clear boundary right-censoring; stable plateau unproven |
| P17 DeepOHeat full-minus-valid128×3 | CLOSED | valid128 excluded from training pool |
| P18 common-valid table | CLOSED | formal rows use Heat3D e600 only; e200/IDW historical |
| P19 inference/reload/provenance | CLOSED PASS | same-devbox valid-only profile, all finite |

## Model readiness

- **Heat3D-on-DeepOHeat-v1:** valid-only e600×3 ready; U-v2 is direct-query dense inference on 571,256 points.
- **DeepOHeat-v1 full-minus-valid128:** valid-only native-recipe best/final views ready.
- **DeepOHeat-v1 matched-768:** same physical-case budget view ready; not same information budget.
- **GINO:** prior E3-authoritative valid-only receipt remains in force; P19 does not alter it.
- **Transolver:** prior valid-only result remains in force; P19 does not alter it.
- **Therm-FM:** open/needs amendment; no large checkpoint downloaded.

## Open before P20

`P1i test_iid`、`sealed`、`DeepOHeat official100` 仍未授权。P20 需要人工审阅 P19
receipts，冻结 config/runner/split/checkpoint，确认 overlap 为零，批准一次性
evaluation-only unlock，并将解锁结果与 valid-only 表分开报告。P19 的 latency
数据虽来自同一 devbox，但 Heat3D batch=1/571,256 queries 与 DeepOHeat batch=4
的计时边界不同，当前不据此形成新的 Pareto 排名。
