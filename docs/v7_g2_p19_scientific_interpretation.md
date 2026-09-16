# V7 G2-P19：scientific interpretation

P19 只回答推理边界和 provenance，不新增 accuracy 结论。Heat3D formal
benchmark 采用 fresh e600×3 的 U-v2 **direct-query dense inference**；e200 与
IDW 仍是历史诊断。

Heat3D 在固定 571,256-point domain 上的 steady E2E 中位数为 8.874 s，其中
query-graph construction 3.923 s、model forward 4.999 s，postprocess 约
0.002 s；因此本次计时未显示 postprocess 是主要成本。首个 case 为 18.386 s，
包含 cold initialization/compile。DeepOHeat 的同一 devbox batch=4 model-only
中位数为 0.001014 s（validation-selected best）或 0.001835 s（final），E2E
中位数约 0.004915/0.005517 s。各模型 timed rows 的 live peak 分别约为
0.461 GiB 与 0.035 GiB；同一进程累计 JAX allocator peak 5.194 GiB 仅作为
过程级诊断，不归因于 DeepOHeat 单模型。由于两者 query/batch boundary 和算法输出
契约不同，这些数字只能作为 execution-budget evidence，不能单独形成跨模型
Pareto 结论。

P19 profile 使用同一冻结 valid128 输入 provenance 和三类已存在 checkpoint，
未加载温度标签/真值，且所有输出 finite；因此它关闭了 P19 inference/reload
工程检查，但不改变 P18 valid-only metric、selection policy 或 training regime。

IDW 证据必须按 Heat3D V6/P1h utility 解释；DeepOHeat-v1 是直接 dense
field-output 路径。任何把 IDW 称为 DeepOHeat 算法、或把 U-v2 称为
“reconstruction”的表述均不再使用。

P15 的 e600 结论改为“无明确 boundary right-censoring，但稳定 plateau 未被
证明”。因此不扩展到 e1200，也不基于 P19 profiling 改变 e600 checkpoint、
模型或数据合同。下一阶段只能在人工审计后制定一次性 evaluation-only unlock。
