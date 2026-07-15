# Gate 6D evaluator equivalence

冻结 evaluator commit：`639872abcb0f7afd3b6c2d319a7d395bde75c9a4`。比较范围为 best/final、五个 roles 下全部 numeric metric leaves。

| config | numeric paths | exact | non-exact | outside 1e-9/1e-12 tolerance | max abs delta | max relative delta |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 63156 | 29184 | 33972 | 33127 | 30.89813 | 0.64864865 |
| L2 | 63156 | 29044 | 34112 | 33289 | 9.0022145 | 1.4333333 |

结论：存在非零差异；按 Gate 6D 合同冻结 `639872ab` 结果为权威值，collector 旧值只保留作 provenance 对照。

每个配置按 relative delta 排序的前 25 个差异保存在 machine-readable JSON。
