# V7 G2-P16：existing-regime audit

本审计在 P15/P17 正式训练前冻结比较口径，且只使用已关闭的 valid-only receipts。

| regime | model / data | valid overlap | allowed interpretation |
|---|---|---:|---|
| native reference | DeepOHeat-v1 seed42, official 100000-function pool | 128/128 Heat3D valid rows are in training pool | `NATIVE_REFERENCE`; no held-out ranking |
| matched physical-case | DeepOHeat-v1 768 train / 128 valid, P14 | 0 within the cohort | `SAME_PHYSICAL_CASE_BUDGET`; not same information budget |
| Heat3D e200 | 768 train / 128 valid, three frozen seeds | 0 within the cohort | baseline e200; native/IDW/U-v2 views |
| Heat3D e600 | 768 train / 128 valid, fresh seeds | 0 within the cohort | P15 complete; fresh 600e convergence study, e200 scheduler not reused |
| full-minus-valid128 | official pool minus 128 valid IDs | 0 by P17 exclusion manifest | P17 native-recipe held-out validation; training in progress/receipt pending |

DeepOHeat uses PDE/BC/mesh physics supervision and native full-field operators; Heat3D uses
supervised temperature labels on sparse 1024 support. 因此任何跨行结论都只能称
`data-regime` / `physical-case efficiency comparison`，不能称 `SAME_INFORMATION_BUDGET`。

## 已有数值证据

- Heat3D e200 P14.1 U-v2 full-field：sample-first `0.81414 ± 0.00605%`；IDW
  `1.43150 ± 0.01659%`；native-1024 `0.77994 ± 0.01395%`。
- DeepOHeat matched P14 full-field best：`1.30517 ± 0.26832%`，final
  `1.43681 ± 0.30948%`；这不是与 native full regime 的同 split 结果。
- Native seed42 full-pool receipt 已审计为与 Heat3D valid128 overlap，故不用于 valid128
  accuracy ranking。official100 保持 sealed。

P15 e600 与 P17 full-minus-valid128 完成后，新增结果只能进入各自预注册行；不得回写
或重解释上述历史 receipt。

P15 e600 三 seed 已完成（native best `0.68562 ± 0.01167%`; scheduled U-v2
full-field e600 `0.70989 ± 0.00777%`; all three seeds classified
`CONVERGED_WITHIN_600`). P17 full-minus-valid128 training is a fresh native-recipe
cohort on the frozen 99,872-case pool; its validation remains the excluded valid128
and its official test boundary is closed until P18 completion.
