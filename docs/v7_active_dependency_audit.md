# V7 active dependency audit

状态：V7-D0/G0a 静态依赖审计交付已完成；G0b-1 stable runtime 已交付；G0
Code gate 仍未通过；未实施重构。

审计对象是当前 V6/P1i 的正式训练、checkpoint inference、high-N query、
full-field reconstruction、metrics 和 timing 路径。审计依据是仓库中的
import、symbol 定义、直接调用和运行时赋值关系。D0/G0a 静态阶段没有导入或
执行目标模块；本轮 G0b-1 只在 devbox 上对冻结 `valid_iid` fixture 做了受限
的 old/new inference equivalence control，没有训练、solver、数据生成、模型
评估，也没有访问任何 `test_iid`、held-out 或 sealed labels。

## 结论摘要

当前 V6/P1i 路径还不是 G0 要求的独立 `reproducible ML pipeline`。高-N
参考执行器和 production high-resolution inference 虽然承担正式证据路径，
但仍然跨脚本调用 V1 runner 的私有 API，并通过 V3 `*_smoke.py` 模块安装
checkpoint feature hook。V6/P1i 的数据适配、查询几何和 reconstruction 已
有相对清晰的 `rigno/heat3d_v6_*` 模块，但 feature assembly、checkpoint
materialization、metrics 和 timing 仍分散在多个 `scripts/` 文件中。

因此：

- D0 文档边界已记录；
- G0a 静态依赖审计交付**已完成**，但审计结论是 G0 Code gate **未通过**；
- G0b-1 已新增稳定 V7 inference runtime；devbox 上冻结 checkpoint 与
  `valid_iid` fixture 已可用，CPU deterministic control 的 runtime equivalence
  已通过，但默认加速器 backend 的重复 apply 尚未满足严格 equivalence；
- 历史 smoke/development/check 脚本未修改、未删除，且尚未被旧 V6 入口自动
  切换；它们仍是 legacy/reference path；
- 后续只应在新的明确授权下按本文件的最小 extract-core/preserve-legacy
  边界继续推进，不得借 G0b-1 开始 G0b-2 性能优化或高-N 路径重写。

## G0b-1 stable V7 runtime（本轮交付）

本轮在 `rigno/heat3d_runtime/` 建立了不依赖 `scripts/` 的 Heat3D 专用
inference core。它只复用稳定 `rigno/` library semantics，不导入
`*_smoke.py`、`check_*`、`*_development.py`，不修改 `sys.path`，也不向其
他模块写入 hook 或替换 symbol。

新依赖方向为：

```text
scripts/run_heat3d_v7_reference_inference.py
  -> rigno.heat3d_runtime.RuntimeSession
     -> checkpoint.py: checkpoint loading, stats materialization,
        decoder-bypass config resolution, device placement
     -> features.py: explicit V6 feature transform, normalization,
        global/native/qk/scale input assembly
     -> grouping.py: V6 Inputs + graph metadata/graph construction
     -> rigno/heat3d_v6_* + rigno/heat3d_v5_* + rigno/models/rigno.py
```

首批 public API 为 `CheckpointBundle`、`load_checkpoint`、
`materialize_checkpoint_stats`、`resolve_model_config`、`device_params`、`FeatureTransform`、
`GroupBuilder`、`RuntimeSession.from_paths`、
`RuntimeSession.from_checkpoint_and_config`、`RuntimeSession.build_group`、
`RuntimeSession.apply`、`RuntimeSession.predict_native_1024`、
`compare_metadata`、`compare_named_arrays` 和 `snapshot_group`。显式 transform
直接构造 model-visible `Inputs`，保留 V6 zero-delta bridge、归一化、global
context、native physics、q--k regional features 和 scale weights 的输入
语义，但不读取 `example.target`。

`scripts/run_heat3d_v7_reference_inference.py` 是新的 reference inference
入口：它的 split CLI 只允许 `valid_iid`，默认只打印 provenance/count summary，
不计算 metrics、不调用 solver、不生成数据、不写 output artifact。它是入口
适配器，不是 runtime core；runtime core 本身不反向导入任何 script。

### G0b-1 equivalence boundary

`rigno/heat3d_runtime/equivalence.py` 提供 old/new named-tensor comparison 和
snapshot flattening，覆盖 checkpoint interpretation（由
`CheckpointBundle.descriptor()` 提供）、normalization/feature tensors、
global/native/context/scale、graph/model input，以及 prediction/scale output
的记录接口。除 prediction 使用已有 V6 adapter/reference 的 `1e-6` K 语义外，
默认比较为 exact `0.0` max-abs/RMSE；调用方只能显式收紧，不会被工具自动
放宽。

本轮在 devbox 使用冻结 checkpoint
`V6_06_V5best_P1i_seed0_reliable_B24/params_best_valid_point_global.pkl`（epoch
559，SHA-256 为
`51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`）和冻结
`valid_iid` native-1024 fixture 的 32-sample development subset，执行了 old/new
control。CPU backend 下 checkpoint interpretation、352 组 model-visible
inputs、800 组 graph tensors、64 组 predictions 和 32 组 scales 均为
`max_abs=0.0`、`RMSE=0.0`，且 `test_iid_or_sealed_accessed=false`、
`training_or_solver_invoked=false`。这证明当前 stable V7 runtime 在受控 CPU
路径上与 legacy reference 的行为等价，但不是对所有 backend 的性能或数值
复现声明。

同一 checkpoint、同一输入在 devbox 默认加速器 backend 上重复 apply 也观察到
约 `1e-2 K` 量级的变化，因此该 backend 当前不能作为严格 equivalence oracle；
本轮不修改 backend、模型或容差。若冻结 checkpoint 或 valid fixture 不可用，
报告仍必须是 `runtime equivalence pending server validation`，不得伪造数字。

### G0b-1 replacement map and remaining legacy use

| 原 active smoke/private dependency | V7 stable replacement | 当前状态 |
| --- | --- | --- |
| V3 `install_checkpoint_feature_hooks` monkey patch | `FeatureTransform.transform` + checkpoint-configured transform | 新 V7 entrypoint 已替代；旧 V6 脚本仍保留原依赖 |
| `runner._load_params_checkpoint`, `_device_params` | `load_checkpoint`, `device_params` | 新 V7 runtime 已替代 |
| `common._materialize_checkpoint_stats` | `materialize_checkpoint_stats` | 新 V7 runtime 已替代 |
| `runner._resolve_decoder_bypass_model_config` | `resolve_model_config` | 新 V7 runtime 已替代 |
| `runner._make_v6_padded_groups_with_progress` 与 `_attach_*` | `GroupBuilder` + `FeatureTransform` | 新 V7 runtime 已替代 |
| `runner._model_apply` | `RuntimeSession.apply` | 新 V7 runtime 已替代 |
| V6 old high-N/production scripts | 未切换；仍为 legacy/reference | 保留，待后续授权后再决定 adapter 边界 |

因此，**新的 V7 reference path 不再使用 smoke 级别模块**；当前正式 V6
high-N/production/timing 旧路径仍使用 V3 smoke hook 和 V1 runner private
helpers，见上文 active call graph。这种依赖本身没有证据表明会必然降低模型
精度；但 monkey patch 和跨脚本私有 API 可能改变 feature view、normalization
或 model-input assembly 的运行时状态，因而带来静默输入漂移和复现风险，间接
可能改变预测。当前没有证据表明这些 smoke/private dependencies 必然降低
模型性能；但它们确实提高了输入语义漂移与复现失败的风险。CPU control 已观察
到 old/new 完全一致；默认加速器的非确定性则独立地阻塞严格 equivalence，不能
被解释成模型质量下降，也不能被容差放宽掩盖。

## 审计方法与证据边界

### 追踪范围

重点沿以下有向路径检查：

```text
V6/P1i config or frozen checkpoint
  -> V6/P1i reference/production script
  -> checkpoint loading and feature-hook installation
  -> V6 group construction and graph/model apply
  -> high-N query and reconstruction
  -> metrics and lifecycle timing
```

同时搜索了 `*_smoke.py`、`check_*`、`*_development.py`、V1/V3/V4 legacy
helpers、跨脚本私有 symbol、`sys.path` 注入、monkey patch，以及 feature、
normalization、checkpoint、metrics、reconstruction 的重复实现。

文件和 symbol 下方使用仓库相对链接及静态行号；行号是本次审计快照的定位
信息，不代表新运行结果。

本次还只读参考了用户提供的两份外部审计意见：
`publication_review_and_v7_readiness.md` 与 `heat3d-ic_审稿报告.md`。它们在
本记录中作为风险线索和审稿背景，不是本轮执行指令；分类和结论仍以当前
checkout 的实际 import/call 证据为准，外部文件未被修改。

## 关键 active call graph

### V6/P1i high-N/reference route

入口是
[`run_heat3d_v6_p1i_anchor_high_n_development.py`](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py)
中的 development-named reference executor：

```text
anchor_high_n_development
  -> benchmark_heat3d_v6_inference_qualification
  -> benchmark_heat3d_v6_p1i_resolution
  -> evaluate_heat3d_v6_common_valid_probe
  -> run_heat3d_v1_medium_controlled_training_export (runner)
  -> rigno.heat3d_v6_dataset / heat3d_v6_full_field / heat3d_v6_p1i_anchor_query
  -> rigno.models.rigno.RIGNO
  -> run_heat3d_v3_final_probe_checkpoint_smoke.install_checkpoint_feature_hooks
```

具体证据：

- `ROOT` 与 `ROOT/scripts` 被插入 `sys.path`，随后导入 qualification、
  resolution、common probe、V1 runner、V6 modules、RIGNO 和 V3 smoke hook，
  见 [anchor executor imports](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L31-L68)。
- `_checkpoint_runtime` 读取 checkpoint，调用
  `runner._load_params_checkpoint`、`common._materialize_checkpoint_stats`、
  `install_checkpoint_feature_hooks`、
  `runner._resolve_decoder_bypass_model_config` 和
  `runner._validate_model_config`，见
  [checkpoint runtime](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L233-L266)。
- `_physics_fields` 调用 `resolution_base.core.build_mesh` 与
  `resolution_base._continuous_fields`，见
  [physics field preparation](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L305-L321)。
- `_prepare_group` 调用 `qualification.FixedEdgeTargetBuilder`、
  `runner._make_v6_padded_groups_with_progress`、
  `common.standardize_v6_contexts`、
  `runner._global_context_row_for_example` 和多个 `_attach_*` helper，见
  [group preparation](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L695-L736)。
- `_predict_output` 以 `compiled(params, _model_group(group))` 执行模型，
  `_anchor_scale` 与 `_metric_row` 继续在脚本内处理尺度和结果行，见
  [prediction and metric row](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L743-L777)。
- `execute_resolution` 再以 `runner._model_apply` 构造 JAX compiled path，
  并处理 high-N/reconstruction，见
  [resolution executor](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L815-L831)。

该路径的 `development` 命名与正式 V6/P1i reference 角色冲突，是 G0a 的
直接证据；名称本身不是问题，问题是它承担了 publication evidence 所需的
依赖入口。

### V6/P1i training/compatibility route

仓库中仍保留一条 config-to-training compatibility route：

```text
scripts/run_heat3d_v4_config.py
  -> rigno.heat3d_v2_runner_command.build_training_command
  -> _training_script_for_config
  -> scripts/run_heat3d_v4_controlled_training.py 或
     scripts/run_heat3d_v1_medium_controlled_training_export.py
  -> legacy runner.main
```

`rigno/heat3d_v2_runner_command.py` 的 `TRAINING_SCRIPT` 和
`V4_TRAINING_SCRIPT` 分别指向 V1 medium runner 与 V4 wrapper，
`_training_script_for_config` 在配置条件下选择二者；`run_heat3d_v4_config.py`
随后通过 `subprocess.call` 启动命令。这条路径的具体 V6/P1i config role
需要在未来逐项注册确认，因此在本审计中标为 compatibility-only/uncertain，
而不是 V7 training entry。无论通过哪一个选择，V1 runner 的实际 training
semantics 仍来自其 V1/V5/V6 imports，见
[runner imports](../scripts/run_heat3d_v1_medium_controlled_training_export.py#L38-L114)。
这解释了当前正式 checkpoint lineage 为什么仍与 legacy runner 紧耦合。

### V6 production high-resolution route

[`run_heat3d_v6_production_highres_inference.py`](../scripts/run_heat3d_v6_production_highres_inference.py)
导入 anchored/common/source-aware evaluation、graph cache、V6 full-field、
RIGNO、V1 runner 和 V3 smoke hook，见
[imports](../scripts/run_heat3d_v6_production_highres_inference.py#L28-L51)。

其 active call direction 包括：

- `main`/checkpoint setup -> `runner._load_params_checkpoint`、
  `install_checkpoint_feature_hooks`、`runner._resolve_decoder_bypass_model_config`
  和 `runner._device_params`，见
  [checkpoint setup](../scripts/run_heat3d_v6_production_highres_inference.py#L534-L558)；
- group/model path -> `runner._make_v6_padded_groups_with_progress`、
  `runner._global_context_row_for_example`、`runner._attach_*` 和多处
  `runner._model_apply`，见
  [runtime calls](../scripts/run_heat3d_v6_production_highres_inference.py#L211-L339)；
- output -> `build_reconstruction_map` 与脚本内
  `_full_field_metrics`，见
  [reconstruction/metrics](../scripts/run_heat3d_v6_production_highres_inference.py#L354-L396)
  和 [metric collection](../scripts/run_heat3d_v6_production_highres_inference.py#L676-L701)。

它是 production-named，但仍没有独立 runtime core；因此属于当前 active
production dependency，同时也是 G0a blocker。

### Qualification, service timing, and U-v2 paths

- [`benchmark_heat3d_v6_inference_qualification.py`](../scripts/benchmark_heat3d_v6_inference_qualification.py)
  的 `ModelRuntime` 负责 checkpoint/stat materialization、hook 安装、模型
  apply 和 group construction；`metric_accumulate` 同时实现 point-global
  relative、sample-first CV-relative、raw、source、peak、background、layer
  和 interface metrics，见
  [runtime/metric symbols](../scripts/benchmark_heat3d_v6_inference_qualification.py#L315-L506)。
- [`benchmark_heat3d_v6_p1i_final_e_service.py`](../scripts/benchmark_heat3d_v6_p1i_final_e_service.py)
  的服务时序路径调用 high-N 的 `_full_shared`、`_physics_fields`、V1
  runner 的 `_device_params`/`_model_apply`、graph-scale candidate、
  reconstruction，并以 lifecycle schema 的 `serial_metrics`、`q2_metrics`
  和 `validate_cell` 写出 timing evidence，见
  [service imports](../scripts/benchmark_heat3d_v6_p1i_final_e_service.py#L26-L52)
  和 [service calls](../scripts/benchmark_heat3d_v6_p1i_final_e_service.py#L150-L246)。
- [`benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py`](../scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py)
  复用 qualification、high-N、p5r、graph-scale candidate、U1 adapter 和
  lifecycle schema；它还跨脚本调用 U1 private metadata、high-N physics 和
  V1 runner model apply，见
  [U-v2 imports](../scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py#L28-L50)
  和 [U-v2 calls](../scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py#L179-L306)。

这些路径共同说明：timing evidence 并不是只依赖一个 stable inference
function，而是依赖一组脚本私有 helper 和兼容层。

## 分类清单

分类含义：

- **production dependency**：当前 V6/P1i 正式 reference/production、公开
  high-resolution、E/U timing 或其直接 shared runtime 的实际依赖；
- **compatibility-only**：历史语义或 adapter，本应不成为新 production core，
  但当前调用链仍使用它；
- **test-only**：只应服务于 smoke、benchmark、confirmatory 或诊断路径，不能
  成为 publication runtime；
- **dead-or-unused**：本次静态搜索未发现从当前正式 V6/P1i 路径到达的调用；
- **uncertain**：存在条件式、候选式或历史入口，静态证据不足以确认其当前
  formal role。

### Production dependency

| 文件 | symbol / role | 调用方向与证据 | 风险 |
| --- | --- | --- | --- |
| `scripts/run_heat3d_v6_p1i_anchor_high_n_development.py` | `_checkpoint_runtime`, `_prepare_group`, `_predict_output`, `execute_resolution` | checkpoint -> V1 runner/private APIs + V3 hook -> V6 group/model/reconstruction/metric path | 正式 reference 依赖 development-named script |
| `scripts/run_heat3d_v6_production_highres_inference.py` | `main`, `_full_field_metrics` | production entry -> V1 runner/private APIs + V3 hook -> V6 reconstruction/metrics | production path 无 stable core |
| `scripts/benchmark_heat3d_v6_inference_qualification.py` | `ModelRuntime`, `metric_accumulate`, timing helpers | qualification -> checkpoint/hook/model apply -> metrics | 指标实现与 runtime 混在脚本中 |
| `scripts/benchmark_heat3d_v6_p1i_final_e_service.py` | E service/timing path | service -> high-N/p5r/graph-scale/V1 runner -> lifecycle schema | 依赖多个脚本 private API |
| `scripts/benchmark_heat3d_v6_p1i_u2_asymmetric_runtime.py` | U-v2 runtime | U-v2 -> qualification/high-N/p5r/U1/graph-scale -> model/reconstruction/timing | candidate/compatibility edge 渗入 publication timing |
| `rigno/heat3d_v6_dataset.py` | `Heat3DV6DualRobinDataset`, `V6DualRobinExample` | V6 entry -> dataset adapter -> V1 dataclass types/full-field arrays | active data contract，但仍复用 V1 domain types |
| `rigno/heat3d_v6_full_field.py` | `build_reconstruction_map`, `prepare_reconstruction_domain_partition` | high-N -> map -> full-field reconstruction | 可作为 core 候选，但调用方仍有 fallback/重复路径 |
| `rigno/heat3d_v6_p1i_anchor_query.py` | `deterministic_nested_query_order`, `conservative_selected_control_volume`, geometry cache | V6 group/query -> deterministic nested supports and weights | active query core，与脚本组装尚未完全隔离 |
| `rigno/graphBuilder_Heat3D.py`, `rigno/heat3d_graph_cache.py` | graph builder/cache/hash metadata | group preparation -> graph cache -> model input | active graph path，但 cache lifecycle 分散在 scripts |
| `rigno/models/rigno.py` | `RIGNO` | runtime -> graph neural operator backbone | stable model dependency；本轮不改模型 |
| `scripts/heat3d_v6_publication_lifecycle_schema.py` | `serial_metrics`, `q2_metrics`, `validate_cell` | E/U timing -> lifecycle row validation | active evidence schema，不应承担 model semantics |
| `scripts/heat3d_v6_publication_runtime_isolation.py` | `failure_record`, `validate_failure_record` | timing/service -> failure provenance | active failure schema，边界应保持窄 |

### Compatibility-only

| 文件 | symbol / role | 调用方向与证据 | 结论 |
| --- | --- | --- | --- |
| `scripts/run_heat3d_v1_medium_controlled_training_export.py` | `_load_params_checkpoint`, `_device_params`, `_model_apply`, `_make_v6_padded_groups_with_progress`, `_attach_*` | V6 high-N/production/resolution -> `runner.*` | 文件头明确为 V1 training export smoke，但已是 V6/P1i 当前 runtime 的实际依赖 |
| `scripts/check_heat3d_v1_small_train_valid_smoke.py` | `MODEL_CONFIG`, `_global_norm`, `_metrics`, `_sample_root`, `_selected_steps`, `_subset_split_ids` | V1 runner import -> check smoke private symbols | 只能保留为 legacy adapter；不应继续作为新 core 来源 |
| `rigno/heat3d_v1_normalization.py` | train-only stats/normalize/recover helpers | V1 runner -> V1 normalization -> V6 inherited path | compatibility semantics active；需后续显式迁移/冻结 |
| `rigno/heat3d_v1_training_semantics.py` | V1 bridge/training semantics | V1 runner/V3 hook -> legacy zero-DeltaT bridge | 不能从 compatibility edge 升级为 V7 claim |
| `scripts/run_heat3d_v4_controlled_training.py` | legacy wrapper `main` | V4 wrapper -> legacy V1 runner + V1 normalization | V4 legacy training route；本轮保留、不删除 |
| `rigno/heat3d_v2_runner_command.py` | `_training_script_for_config`, `build_training_command` | V2/V4 config -> V4 wrapper 或 V1 medium runner | 配置入口仍能选择 legacy training script；与 V7 reproducible pipeline 不同 |
| `scripts/run_heat3d_v4_config.py` | subprocess launch/dry-run | V4 config -> `rigno.heat3d_v2_runner_command` -> legacy training script | compatibility entry；不是 V7 Quick Start |

V1 runner 的具体 import 关系包括 V1 normalization、V1 training semantics、
V6 dataset/global context、V5 context/scale、V4 split map 和 V1 `MODEL_CONFIG`
等，见 [runner imports](../scripts/run_heat3d_v1_medium_controlled_training_export.py#L38-L114)。
这解释了为什么“V6 checkpoint runtime 已冻结”不等于“V6 runtime dependency
已经解耦”。

### Test-only

| 文件 | symbol / role | 调用方向与证据 | 约束 |
| --- | --- | --- | --- |
| `scripts/run_heat3d_v3_final_probe_checkpoint_smoke.py` | `install_checkpoint_feature_hooks` | V6 paths -> hook -> `runner_module._bridge_for = _bridge_for` | intent 是 checkpoint smoke；由于被正式路径 import，当前同时是 G0a blocker |
| `scripts/benchmark_heat3d_v6_p1i_resolution.py` | `_build_group`, `_predict_once`, `_metric_summary` | resolution benchmark -> V1 runner helpers + V6 continuous core -> FVM `_assemble`/`_solve` | benchmark/solver-only；本轮不运行 |
| `scripts/heat3d_v6_p1i_continuous_core.py` | `build_mesh`, `_continuous_fields`, `_assemble`, `_solve` | resolution benchmark -> continuous FVM reference path | solver reference only，不进入 neural production core |
| `scripts/evaluate_heat3d_v6_p1i_e16384_test_confirmatory.py` | confirmatory test route | frozen route -> test_iid once-only evidence | held-out test set 不得重开、重算选择或调参 |
| `scripts/evaluate_heat3d_v6_p1i_randomblock_transfer.py` | `layer_aware_fallback_map` and transfer diagnostics | qualification/diagnostic path -> random-block fallback | historical/diagnostic role，不能变成 V7 generalization claim |
| `scripts/check_heat3d_v1_validation_metrics_smoke.py`, `scripts/compare_heat3d_v1_smoke_vs_v2_labels.py`, `scripts/compare_heat3d_v1_medium_baselines.py` | V1 metric/check helpers | legacy check scripts -> `rigno.heat3d_v1_metrics` | test-only；不应被 V7 formal runtime import |

### Dead-or-unused

本次静态搜索没有给当前 V6/P1i formal path 归入“已确认 dead”的核心模块。
`rigno/heat3d_v1_metrics.py` 仍被 V1 check/compare 脚本直接导入，因此不能
称为 dead；它应分类为 test-only/legacy，而不是删除对象。历史模块的“未被
当前 production path 到达”不等于可以删除，本轮不做清理。

### Uncertain

| 文件 | symbol / role | 不确定性来源 | 当前处理 |
| --- | --- | --- | --- |
| `scripts/run_heat3d_v6_p1i_graph_scale_candidate.py` | graph-scale candidate | 被 E service/U2 间接使用，但属于 candidate path，是否为最终 publication dependency 取决于 route freeze | 记录为 conditional dependency，不纳入 V7 core 结论 |
| `scripts/heat3d_v6_supplemental_runtime.py` | supplemental runtime | 被 E service/U2 使用，属于补充/known-topology 语义 | 保留 supplemental boundary，不外推为 public E2E |
| `rigno/heat3d_v6_gpu_reconstruction.py` | GPU reconstruction helpers | 由 graph-scale candidate 使用，未证明是主 E reconstruction 的唯一实现 | 作为候选实现，需未来 route-specific audit |
| `scripts/run_heat3d_v4_config.py` + V2 command | config-to-training route | 可启动 legacy runner，但当前 V6/P1i config 是否经此入口正式训练需按 config role 逐项确认 | 保留 compatibility-only，不将其写成 V7 entry |

## Private API、monkey patch 与 runtime mutation

### 跨脚本 private API

当前 active paths 直接使用下列带下划线的跨模块 symbol：

- `runner._load_params_checkpoint`、`runner._device_params`、
  `runner._model_apply`；
- `runner._make_v6_padded_groups_with_progress`、
  `runner._global_context_row_for_example`；
- `runner._attach_global_context_to_groups`、
  `_attach_native_physics_to_groups`、`_attach_qk_region_features_to_groups`、
  `_attach_scale_deepsets_weights_to_groups`；
- `common._materialize_checkpoint_stats`、`resolution_base._continuous_fields`；
- high-N 的 `_full_shared`、`_physics_fields`、`_metric_row`；
- U1 的 `_u_v2_asymmetric_metadata` 等 private metadata helpers。

调用方向是“V6/P1i publication script -> legacy/benchmark script private
symbol”，不是“public script -> stable library API”。这会使文件移动、import
顺序、签名调整或私有默认值变化都可能破坏 checkpoint inference，而不会由
包级接口检查提前发现。

### Monkey patch / runtime mutation

[`install_checkpoint_feature_hooks`](../scripts/run_heat3d_v3_final_probe_checkpoint_smoke.py#L206-L220)
定义 checkpoint-specific bridge closure，随后执行
`runner_module._bridge_for = _bridge_for`。这不是纯函数依赖注入，而是修改
已导入 runner module 的全局 symbol。其实际方向为：

```text
V6 checkpoint runtime
  -> V3 smoke hook
  -> mutate run_heat3d_v1_medium_controlled_training_export._bridge_for
  -> subsequent runner group/model path
```

这构成 G0a 的明确 blocker：checkpoint feature semantics 取决于 hook 是否安装、
安装顺序以及同一 Python process 中的 module state。V7 合同要求显式依赖
注入；本轮只记录，不修改。

### `sys.path` mutation

高-N reference executor 在导入前把 repository root 和 `scripts/` 加入
`sys.path`，见 [imports](../scripts/run_heat3d_v6_p1i_anchor_high_n_development.py#L31-L34)。
这使 script-to-script import 在源码 checkout 中可用，但隐藏了 package
边界，也增加 clean checkout 与安装式运行的差异。

## 重复实现与语义分散

### Feature assembly

V1 runner 的 `_make_v6_padded_groups_with_progress` 及其四个 `_attach_*`
helper 负责当前主组装；V6 dataset 提供 domain examples；high-N、production
和 U2 脚本又各自持有 query/support、metadata、scale 或 group wrapper。结果是
feature schema 的“数据对象—组装—模型输入”边界分散在 V1 runner 与多个脚本，
而不是一个 V6 public feature builder。

### Normalization and bridge semantics

V1 normalization、V6 global-context helpers、V5 context/scale helpers、
`common.standardize_v6_contexts` 和 V3 checkpoint hook 共同决定当前特征语义。
它们并非一个显式的 `CheckpointStats -> FeatureTransform` public contract；
V3 hook 还会在 runtime 修改 bridge。

### Checkpoint loading and materialization

`runner._load_params_checkpoint` 是 legacy loader；V3 smoke 模块重新导入并暴露
它，high-N `_checkpoint_runtime`、qualification `ModelRuntime`、production
highres 和 p1i resolution 各自安排 loader、stats materialization、model
config validation 与 device placement。没有唯一的 V6 checkpoint runtime core。

### Metrics

当前发现至少四组实现/包装：

1. qualification 的 `metric_accumulate`；
2. source-aware evaluation 的 `_metrics`；
3. production highres 的 `_full_field_metrics`；
4. p1i resolution 的 `_metric_summary` 与 high-N 的 `_metric_row`。

此外，`rigno/heat3d_v5_metrics.py` 提供 control-volume weights，V1 check/compare
路径使用 `rigno/heat3d_v1_metrics.py`。这些模块的存在不表示结果错误，但会使
point-global、sample-first CV-relative、raw/source/peak/interface 指标的
定义和聚合边界难以由单一 public API 保证。

### Reconstruction

`rigno/heat3d_v6_full_field.py` 提供 `build_reconstruction_map`，但部分脚本
通过 runner re-export/fallback 使用它；random-block transfer 有
`layer_aware_fallback_map`；E service、U2 和 graph-scale candidate 还在各自
的 timing/route wrapper 中编排 reconstruction。需要区分“同一 map builder 的
复用”和“重复的 route-specific wrapper”，不能仅凭同名函数认定实现已经统一。

### Timing

`scripts/heat3d_v6_publication_lifecycle_schema.py` 对 row schema 和 validation
较集中，但 final E、U2、supplemental 和 production scripts 仍自行组织
`perf_counter` 阶段、fresh/resident/Q1/Q2 边界和错误记录。timing schema 是共享的，
timing orchestration 不是单一 reference inference pipeline。

## 最小 extract-core / preserve-legacy 边界（仅提案）

以下是后续 G0b 的最小边界，不是本轮实施计划，也不授权重构：

### 只抽取稳定语义

在不改变 V6 checkpoint、graph、feature 数学和 reconstruction 结果语义的前提
下，未来可逐步抽取以下 public core：

- `rigno/runtime/checkpoint.py`：checkpoint loading、stats materialization、
  config validation 和 device placement；
- `rigno/runtime/features.py`：V6 condition normalization、global/native
  physics/context/scale features 与明确版本字段；
- `rigno/runtime/grouping.py`：graph/group preparation 与 padding contract；
- `rigno/runtime/inference.py`：显式依赖注入的 model apply；
- `rigno/eval/metrics.py`：统一且命名明确的 point-global、sample-first、raw、
  source、peak、interface metrics；
- `rigno/eval/reconstruction.py`：对现有 V6 map builder 的窄封装与 route metadata；
- `rigno/runtime/timing.py`：fresh/resident/Q1/Q2/FVM boundary 的 schema-only
  orchestration。

抽取顺序应先建立 public API 和静态 import policy，再逐条让 production/reference
入口切换；不能先移动 legacy 文件再追错误。

### 明确保留 legacy

以下对象在未来仍保留为 compatibility/history/test adapter，不在本轮删除：

- V1 medium runner 及其 V1 normalization/training semantics；
- V3 checkpoint smoke hook；
- V1 check/compare smoke scripts 与 V1 metrics；
- V4 wrapper、V2 config command、V0/早期历史入口；
- 所有历史 V1–V6 docs、冻结结果、sealed-data provenance。

preserve-legacy 的约束是：legacy 可继续被历史测试或显式 compatibility entry
调用，但新的 production/reference import graph 不得反向依赖它们。

### 后续 G0b 的验收方向

未来若获得单独授权，G0b 至少需要静态 import graph 检查、无 script-private
imports、无 monkey patch、checkpoint provenance、clean-checkout dry-run，以及
对冻结 fixture 的 output/metric equivalence 证据。那时才可讨论如何逐步启用
新的 baseline/ablation/OOD evaluation；本轮不运行这些检查路径，也不接触任何
held-out/sealed labels。

## G0a 判定

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| D0 研究边界与术语 | 完成（文档层） | README 与 [V7 research contract](v7_research_contract.md) 已区分 V6 evidence、Level-A、Level-B 与 V7 objectives |
| G0a 静态依赖审计交付 | 完成 | 本文已记录 active call graph、分类、private API、monkey patch、重复实现与后续边界 |
| G0 Code gate：production path 不依赖 smoke/check/development | 未通过 | high-N/production 仍 import V3 smoke hook；high-N 文件本身为 development-named |
| 不依赖跨脚本 private API | 未完成 | V6 paths 调用 V1 runner、qualification、common、U1/high-N private symbols |
| 不使用 monkey patch/runtime mutation | 未完成 | V3 hook 给 `runner_module._bridge_for` 赋新 closure |
| feature/normalization/checkpoint/metrics/reconstruction 单一 core | 未完成 | 本文“重复实现与语义分散”所列多组路径 |
| 本轮禁止事项 | 已遵守 | 未训练、未求解、未生成数据、未访问 `test_iid`/held-out/sealed labels、未改模型/冻结 artifact；仅运行冻结 `valid_iid` 的受限 equivalence control |

结论：**V7-G0a 静态审计已完成；V7-G0b-1 的 stable runtime 与 CPU control
equivalence 已交付，但 G0 Code gate 仍未通过，不能进入正式 V7
baseline/ablation/OOD evaluation。** 默认加速器 backend 的重复 apply 尚未满足
严格 reproducibility/equivalence，需要后续单独冻结 backend/determinism policy；
本轮不做该项优化或修复。
本文件只冻结问题清单和后续最小边界；下一阶段若要开始 G0b，必须另行授权，
并继续保持 V6 frozen artifacts 与 sealed data 不变。
