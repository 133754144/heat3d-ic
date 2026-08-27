# V7 active dependency audit

状态：V7-D0/G0a 静态依赖审计、G0b-1 stable runtime 和 G0b-2 native/high-N
cutover audit 已交付；G0 Code gate 仍未通过；本轮未实施性能优化或历史代码
重构。

审计对象是当前 V6/P1i 的正式训练、checkpoint inference、high-N query、
full-field reconstruction、metrics 和 timing 路径。审计依据是仓库中的
import、symbol 定义、直接调用和运行时赋值关系。D0/G0a 静态阶段没有导入或
执行目标模块；本轮 G0b-1/G0b-2 只在 devbox 上对冻结 `valid_iid` fixture 做了受限
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
- G0b-1 已新增稳定 V7 inference runtime；G0b-2 又新增了不导入 script-private
  API 的 anchor-derived high-N runtime 和 reference entrypoint。devbox 上冻结
  checkpoint 与 `valid_iid` fixture 已可用；CPU deterministic control 与
  native-1024 stable high-N entrypoint 已通过，GPU 则单独观察到 backend
  reduction/aggregation 非确定性；
- 历史 smoke/development/check 脚本未修改、未删除，且尚未被旧 V6 入口自动
  切换；新的 V7 native/high-N reference path 已切断这些依赖，但旧 V6
  production/timing path 仍是 legacy/reference path；
- G0b-2d 已在临时、label-independent fixture 上完成 E16384/U-v2 16384
  与 E32768 的 CPU compatibility closure；稳定入口仍按 contract fail-closed，
  因 devbox 缺少 identity-level artifacts，historical binary reconciliation
  尚未完成。

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

## G0b-2 GPU repeatability 与 high-N stable-runtime cutover

本轮只做行为等价替代和问题审计，没有训练、solver、数据生成、模型/graph
algorithm、sampling、batching、cache policy、reconstruction algorithm 或模型
架构改动。所有运行均限制在冻结 `valid_iid`；`test_iid`、held-out/sealed labels
均未访问。

### 已交付的 stable high-N 边界

`rigno.heat3d_runtime.high_n` 提供 `FullFieldGeometry`、`SupportArtifact`、
`HighNCase` 和 `HighNRuntime`。它只接受已有 full-field geometry、已有
source-aware support 和已有 graph metadata；不会生成 support、调用 solver 或
写 prediction artifact。其 graph path 显式使用已有 `sparse_kdtree_v1` backend，
通过 `RuntimeSession.build_group_from_metadata` 保留 V6 feature/group assembly，
并将 reconstruction 委托给现有 `rigno.heat3d_v6_full_field` builder。已有的
anchor-derived conditioning/context/scale、support order、边界特征和
reconstruction semantics 未改变。

新的 [`run_heat3d_v7_high_n_reference.py`](../scripts/run_heat3d_v7_high_n_reference.py)
只允许 `valid_iid`，并对高于 1024 的 resolution 缺失 artifact fail-closed。其
stable import direction 为：

```text
run_heat3d_v7_high_n_reference
  -> rigno.heat3d_runtime.HighNRuntime
  -> RuntimeSession -> FeatureTransform / GroupBuilder
  -> heat3d_v6_p1i_anchor_query / graphBuilder_Heat3D / heat3d_v6_full_field
  -> RIGNO
```

该入口不导入 V1 runner private API、V3 smoke hook、`*_smoke.py`、
`*_development.py` 或其他 `scripts/` module，也不修改 `sys.path` 或 module
state。旧 V1–V6 路径保持原样，仅作为 legacy/reference control。

### Native-1024 equivalence 与 GPU repeatability

机器可读证据保存在
[`v7_g0b2_receipt.json`](v7_g0b2_receipt.json)。同一 frozen checkpoint、同一
`valid_iid` sample `v6p1if1_0003`、同一构造后的 native-1024 graph/model-visible
tensors 上，legacy 与 V7 各重复 10 次，并比较 100 个 ordered pairs：

| backend / comparison | feature、group、graph | prediction repeatability | scale repeatability | strict old/new 判定 |
| --- | --- | --- | --- | --- |
| CPU legacy vs V7 | named tensors 与 graph hash exact；max-abs/RMSE 均 0 | 两路径 100 对均 0 K | 两路径 100 对均 0 | 通过 |
| GPU legacy within-runtime | named tensors exact；graph hash exact | against-first max `0.00811767578125 K`；all-pairs max `0.009033203125 K` | all-pairs max `0.000274658203125` | repeatability envelope，不作 strict pass |
| GPU V7 within-runtime | named tensors exact；graph hash exact | against-first max `0.008026123046875 K`；all-pairs max `0.010009765625 K` | all-pairs max `0.000274658203125` | repeatability envelope，不作 strict pass |
| GPU legacy vs V7 | feature、group、graph hash exact | cross-runtime max `0.009674072265625 K`；RMSE max `0.0018008972268514405 K` | max `0.000274658203125` | 未通过 strict exactness |

GPU 的非零变化发生在相同 model-visible inputs 和相同 graph hash 已固定之后，
因此当前证据对“backend reduction/aggregation 非确定性”的归因强度为**中—高**；
但本轮没有运行 deterministic-GPU control 或 profiler，不能定位到具体 primitive，
也不能把它表述为模型性能下降。没有放宽 tolerance；receipt 中
`tolerance_forced_pass=false`。CPU 仍是 semantic-equivalence oracle。

### Resolution ladder 状态

- **native 1024：完成**。稳定 high-N reference entrypoint 在 devbox CPU 上实际
  执行，`v6p1if1_0003` 的 support hash 为
  `71e9d11396821ca08828509c144f634f8267d80cd055162f5026f11e904e56af`，graph
  tensor hash 为
  `5dce892465c2adf4ac3380d5ecbeb81d60380d1893dcf9752d368a65b88c4757`；未写
  graph cache。
- **E16384 small / extended：temporary compatibility closure**。G0b-2d 在
  `valid_iid` 上完成了 stable/legacy CPU 等价比较；devbox 仍没有可绑定的
  identity-level support、anchor prediction、graph/padding 或 reconstruction
  map evidence，因此 historical closure 仍 pending。本轮不生成 support、不调用
  continuous solver、不访问非 `valid_iid` labels。
- **E32768：temporary compatibility forward closure**。G0b-2d 对一个固定
  `valid_iid` sample 完成 graph/model input、forward、scale、direct prediction
  和 reconstruction-map 的 CPU exact comparison；历史 E32768 artifact 仍缺失，
  该结果不替代 historical high-resolution evidence。

因此本轮可以证明 stable API、native control 和 temporary high-resolution
compatibility closure；不能宣称 historical 16k/32k binary reconciliation 或
accuracy/latency claim 已完成。

### 当前局限与只审计不实施的优化候选

下表记录当前行为、潜在成本、候选方向和语义变化风险；本轮不实施任何一项。

| 项目 | current behavior | potential cost | candidate optimization direction | whether semantics may change |
| --- | --- | --- | --- | --- |
| repeated `FeatureTransform` / raw feature extraction | 每次 group/case 仍显式执行 feature transform 与 raw-field assembly | 重复 CPU/JAX materialization | 以 sample/geometry/feature-version 为 key 的 session memoization | 低，但 key 漂移或 stale cache 会改变输入 |
| batch-scoped `Heat3DGraphBuilder` construction | `GroupBuilder`/high-N metadata path 按构造边界创建 builder | builder setup 与 Python orchestration 重复 | session-scoped builder/context | 低，需证明 config/seed 完全相同 |
| graph/KD-tree reuse | graph cache 只有显式 artifact/cache-dir 才参与；默认 reference 不启用 resident reuse | high-N graph/KD-tree construction 成本高 | geometry/session-scoped graph/KD-tree reuse | 中，cache key/version 不完整会改变邻接 |
| reconstruction partition/map reuse | `HighNRuntime.reconstruction` 调用既有 map builder；未建立稳定 session reuse contract | full-field query/map 构造重复 | reuse `ReconstructionDomainPartition`/map | 中，stale partition 或排序变化可改 output |
| compiled/JIT executable cache | `RuntimeSession.apply` 没有明确的 keyed compiled executable cache | 重复 compile/warm-up 或 specialization | 按 device/model/group signature 建 cache | 低—中，shape/padding/device key 错误会改变行为 |
| high-N Python per-sample orchestration | case/support/graph/model 调用仍逐 sample 组织 | Python dispatch 与 host-device sync | 批量或异步 orchestration | 中，可能改变 query order、padding 或 aggregation |
| fixed-shape / padding memory | GroupBuilder 保留 V6 padded group contract | 高-N memory footprint 与 padding waste | shape buckets 或 mask-aware representation | 高，mask/dummy semantics 需重新证明 |
| audit/reference path mixing | legacy V6 audit/production/timing 仍混合脚本 private API；V7 stable path 已分离 | provenance、复现和维护成本 | extract-core / preserve-legacy adapters | 高，替换时必须逐项做 equivalence |
| metrics/timing duplicate implementation | qualification、production、resolution、high-N 仍有多处 metric/timing wrapper | aggregation boundary 和 lifecycle timing 漂移 | 统一命名的 metrics/timing core | 高，不能把不同 metric 定义静默合并 |
| GPU nondeterministic reduction/aggregation | CPU exact；GPU 同进程 10× prediction max 约 `0.010009765625 K` | repeatability envelope 与严格 regression 判定受限 | deterministic control、稳定 reduction 或 backend policy | 高，可能牺牲性能；本轮不用于性能 claim |

### G0b-2 dependency replacement status

| dependency | 当前 V7 native/high-N 状态 | 旧 V6 formal path 状态 |
| --- | --- | --- |
| V3 `install_checkpoint_feature_hooks` | 已由显式 `FeatureTransform` 替代 | 仍由 anchor high-N、production highres、qualification 路径使用 |
| V1 runner checkpoint/private model helpers | 已由 `CheckpointBundle`、`load_checkpoint`、`RuntimeSession.apply` 替代 | 仍是旧 reference/production/timing 的 active compatibility edge |
| V1 `_make_v6_padded_groups_with_progress` 与 `_attach_*` | 已由 `GroupBuilder`/`FeatureTransform` 替代 | 旧 V6 path 仍调用 |
| high-N support/graph/reconstruction orchestration | `HighNRuntime` 已提供 stable adapter；native 1024 与 temporary high-N fixture 已实跑 | historical 16k/32k identity artifacts 缺失，旧 high-N route 未被自动切换 |
| scripts/private metric/timing helpers | V7 reference entrypoint 不导入 | qualification/E/U/production wrappers 仍使用，尚未抽取 |

当前 G0 blocker 是：(1) 旧 V6 publication/timing path 尚未全部切到 stable
runtime；(2) historical identity-level E/U artifacts 缺失，无法完成 binary
reconciliation closure；(3) GPU backend repeatability policy 尚未冻结；(4)
metrics/timing 尚未形成单一 public evaluation core。上述 blocker 不表示本轮
stable native-1024 或 temporary compatibility closure 失败，也不允许提前进入
G0c/G1、性能优化或 publication claim。

## V7-G0b-2d schema correction and historical reconciliation

本节是本轮的 authoritative update。此前 G0b-2c receipt 中的
`conditioning_resolution`/`query_resolution` 只作为历史证据字段保留，不能再
作为 V7 G3 主比较变量。新的 machine-readable contract 在
[`v7_g0b2c_eu_contract_manifest.json`](../configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json)
中使用四个显式字段：

| route | anchor_context_resolution | encoder_input_resolution | output_query_resolution | reconstruction_resolution |
| --- | ---: | ---: | ---: | ---: |
| E16384 | 1024 | 16384 | 16384 | 240825 |
| E32768 compatibility | 1024 | 32768 | 32768 | 240825 |
| U-v2 16384 | 1024 | 1024 | 16384 | 240825 |

因此 U-v2 保留 native 1024 encoder/conditioning semantics，同时执行 16384
direct output query；它不是 reconstruction-only route。旧字段仅在
`deprecated_compatibility_fields` 中允许显式 legacy parsing，并标记为
`ambiguous_deprecated`，不可用于 G3 主变量。每份 G0b-2d receipt 还必须记录
`strategy`、上述四个字段、`direct_query` 和 `execution_role`。

devbox 的只读搜索找到了
`output/v6_supplemental_publication_8a81261` 下的 E16384、U-v2 16384 和
240825 route receipts、input plans、full-field prediction arrays 与 U-v2
padding record。checkpoint SHA256、dataset manifest SHA256 和 full-field SHA256
均与 V6/P1i freeze manifest 相符；U-v2 receipt 也显式记录了 frozen checkpoint
SHA。可是这些目录没有保存 V6 binding 所需的 identity-level support/order、
query coordinates、raw/padded graph metadata、完整 model-visible tensors、
anchor prediction/scale 或 reconstruction map；E receipt 还只保存了 parameter
tree before/after SHA，而非 checkpoint file SHA。devbox capture 因此只能作为
历史 route observation，不能完成三方 binary reconciliation。

本轮的机器可读记录是
[`v7_g0b2d_historical_reconciliation_manifest.json`](v7_g0b2d_historical_reconciliation_manifest.json)。
其状态为 `partial_observation_fail_closed`，并明确区分：

```text
historical_artifact_reconciliation = pending_missing_identity_artifacts
wsl2_mirror_reconciliation = pending
```

WSL2 镜像核查不是当前唯一 blocker，但在镜像恢复后仍必须执行。临时夹具仍
仅标记为 `V7 Refactor Compatibility Fixture`，不替代 V6 evidence，不具备
publication headline、G3 或 final-test 资格。

## G0b-2d E32768 forward compatibility

test-only harness 现在对一个 frozen `valid_iid` sample 执行 E32768 的 old/new
CPU forward compatibility comparison，比较 support/query、raw 与 padded graph
metadata、graphs/model inputs、raw prediction、query `s_hat`、anchor scale、
anchor-scaled direct prediction 和 reconstruction map/field。它不读取 target，
不计算 metrics，不把该结果升级为 valid32 accuracy study。该 forward 使用
temporary fixture，因为 devbox 没有历史 E32768 artifact；历史 E32768 状态仍为
`historical_artifact_found=false`。

## Production semantic-contract preflight

`rigno.heat3d_runtime.preflight.validate_semantic_contract` 是纯验证层；
`RuntimeSession` 在构造 `FeatureTransform`、`GroupBuilder` 和 `RIGNO.apply`
之前调用它。production/compatibility session 的 checkpoint stats、run config、
model config 与可选 E/U route contract 必须显式包含语义关键字段；缺失字段直接
抛出 `SemanticContractError`，不会以 `legacy_default` 静默补齐。当前检查覆盖：

| acceptance item | result | evidence |
| --- | --- | --- |
| production import graph excludes `*_smoke.py`, `check_*`, `*_development.py` | PASS for V7 entrypoints; legacy V6 excluded | `rigno/heat3d_runtime/`, `scripts/run_heat3d_v7_*` |
| cross-script private `_...` API | PASS for V7 production entrypoints; audit harness remains compatibility-only | V7 entrypoint AST/import checks |
| monkey patch/module-state rewrite | PASS | explicit `FeatureTransform`; no hook installation or `sys.path` mutation |
| semantic-critical config missing | PASS fail-closed negative tests | `tests/test_heat3d_runtime.py`, `preflight.py` |
| frozen checkpoint/sample old-new equivalence | native control PASS; high-N temporary fixture PASS; historical binary closure pending | G0b-2d receipt and reconciliation manifest |
| provenance role | PASS structurally | `execution_role` is one of `publication_training`, `production_inference`, `compatibility_audit`; this receipt is `compatibility_audit` |

V6 frozen evidence generators and historical scripts remain outside the V7
production graph. They may be reached only by the test-only compatibility audit. The
`u_split.py` dependency on RIGNO library-internal `_...` methods is recorded technical
debt; this round does not refactor it.

## V7-G0b-2d evidence boundary and blockers

The new receipt is [`v7_g0b2d_receipt.json`](v7_g0b2d_receipt.json). It records the
four-field route contract, devbox reconciliation state, temporary-fixture provenance,
E32768 forward result, production preflight result and prohibited-action flags. No
large NPZ/cache is committed. No training, solver, data generation, test/sealed-label
access, model change, graph/sampling/reconstruction semantic change, batching/cache/JIT
optimization, timing cutover or G1 experiment is included.

The remaining G0 blockers are:

1. Obtain and bind the missing identity-level historical E/U artifacts before calling
   historical reconciliation complete.
2. Replace the still-legacy V6 publication/high-N/timing dependency edges with
   equivalence-verified adapters; the V7 entrypoints themselves have passed the
   production import-graph check.
3. Complete the formal G0 acceptance matrix for the legacy exclusion boundary;
   GPU repeatability policy is now frozen separately from CPU semantic equivalence
   in G0b-2e.

## V7-G0b-2e：route binding hardening and historical behavioral reconciliation

Machine-readable receipt: [`v7_g0b2e_receipt.json`](v7_g0b2e_receipt.json)。

### Registered route binding

V7 E/U production entrypoints now require an explicit registered `route_id` for every
high-resolution request, together with the requested strategy, all four resolution
roles and the fixed padding envelope. The binder compares each requested value with
the registered route before constructing `RuntimeSession`; it does not infer a route
from output resolution. The registered route itself must satisfy:

- E：`encoder_input_resolution == output_query_resolution`；
- U-v2：`encoder_input_resolution == anchor_context_resolution == 1024`；
- `output_query_resolution == requested output resolution`；
- fixed edge-target envelope equality；
- route ID equality and explicit registration。

Consequently, `U@32768` fails closed: neither `U_v2_32768` nor another implicit
resolution alias is registered. Passing `U_v2_direct240825` with output resolution
32768 also fails. E32768 remains compatibility-only because its registered envelope
is unresolved; no scientific U32768 route was added.

Positive binding gates cover E16384 and U-v2 16384. Negative tests cover unknown route,
strategy mismatch, route ID mismatch, each resolution mismatch, padding mismatch,
deprecated/unresolved envelope and the U@32768 fallback case. The V7 production
import graph remains free of smoke/check/development modules, script-private imports
and monkey patches.

### Historical behavioral reconciliation

The devbox capture contains historical execution commit
`8a812619ab0112b4ecfc37ef18189f731180059d`, E16384/U-v2 16384 input plans, route
receipts and GPU prediction artifacts. Every available replay plan is bound to the
four train geometries `v6p1if1_0056`, `v6p1if1_0079`, `v6p1if1_0971` and
`v6p1if1_0393`; none is a frozen `valid_iid` plan. Because this round is restricted
to frozen `valid_iid`, historical replay was not executed. This is a concrete
fail-closed boundary, not an inferred numerical mismatch.

The two reconciliation dimensions are therefore recorded separately:

```text
historical_behavioral_reconciliation = pending_replay_not_executed_valid_iid_boundary
historical_identity_artifact_reconciliation = unavailable_missing_original_artifacts
```

The first is not promoted to complete: no historical replay ↔ V7 replay bridge was
executed under the allowed population. The second is an archival limitation: the
devbox capture lacks identity-level support/order, query-coordinate, raw/padded graph,
model-visible, anchor-scale and reconstruction-map artifacts. Existing temporary
fixture equivalence remains separate and does not substitute for either bridge.

### Frozen GPU equivalence policy

No new GPU experiment was run. The frozen policy is:

| role | policy |
| --- | --- |
| CPU deterministic backend | semantic-equivalence oracle |
| normal GPU backend | production/timing backend |
| GPU equivalence | use the existing repeatability envelope; do not reinterpret it as binary identity |
| deterministic GPU mode | optional diagnostic only; never a formal timing mode |

### G0 acceptance update

| acceptance item | result |
| --- | --- |
| production import excludes smoke/check/development | PASS for V7 entrypoints |
| no cross-script private API | PASS for V7 production graph |
| no monkey patch/module-state rewrite | PASS |
| semantic-critical fields fail closed | PASS, including negative tests |
| registered route binding fail closed | PASS, including U@32768 fallback rejection |
| CPU old/new equivalence | PASS for native and temporary valid-only compatibility fixtures |
| historical behavioral reconciliation | PENDING: only train-bound historical plans were available |
| historical identity artifacts | UNAVAILABLE: original identity-level artifacts missing |
| GPU policy | FROZEN as the existing envelope policy above |
| provenance role | PASS: this receipt is `compatibility_audit`; formal V7 inference is `production_inference` |

No feature, graph, sampling, reconstruction, batching, cache, JIT, metrics/timing
core or model semantics were changed. Historical V6 scripts remain read-only
compatibility oracles outside the V7 production graph.

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

G0b-2 已完成静态 import graph 检查、无 script-private imports、无 monkey patch、
checkpoint provenance、clean-checkout 级别的入口检查，以及 native-1024 冻结
fixture 的 output equivalence 证据。仍需在后续授权下完成旧 V6 formal path 的
全面切换、16k/32k artifact ladder、统一 metrics/timing core 和 backend
determinism policy；本轮不运行 solver，也不接触任何 held-out/sealed labels。

## G0a 判定

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| D0 研究边界与术语 | 完成（文档层） | README 与 [V7 research contract](v7_research_contract.md) 已区分 V6 evidence、Level-A、Level-B 与 V7 objectives |
| G0a 静态依赖审计交付 | 完成 | 本文已记录 active call graph、分类、private API、monkey patch、重复实现与后续边界 |
| G0 Code gate：production path 不依赖 smoke/check/development | 未通过 | 新 V7 native/high-N reference 已切断这些依赖，但旧 high-N/production/timing formal path 仍 import V3 smoke hook，且旧 high-N 文件为 development-named |
| 不依赖跨脚本 private API | 未完成 | V6 paths 调用 V1 runner、qualification、common、U1/high-N private symbols |
| 不使用 monkey patch/runtime mutation | 未完成 | V3 hook 给 `runner_module._bridge_for` 赋新 closure |
| feature/normalization/checkpoint/metrics/reconstruction 单一 core | 未完成 | 本文“重复实现与语义分散”所列多组路径 |
| 本轮禁止事项 | 已遵守 | 未训练、未求解、未生成数据、未访问 `test_iid`/held-out/sealed labels、未改模型/冻结 artifact；仅运行冻结 `valid_iid` 的 CPU/GPU repeatability control 与 native-1024 stable cutover |

结论：**V7-G0b-2d 已完成 stable native/high-N runtime 边界、CPU semantic
equivalence、temporary E/U high-N compatibility closure 和 GPU repeatability
characterization，但 G0 Code gate 仍未通过，formal 16k/32k historical ladder
仍 pending。** GPU 默认 backend 的重复 apply
尚未满足严格 reproducibility/equivalence，需要后续单独冻结 backend/determinism
policy；本轮不做该项优化或修复。本文件继续冻结 V6 artifacts、sealed data、
历史 legacy 路径和 extract-core / preserve-legacy 边界。

## V7-G0b-2c：E/U contract freeze 与 high-resolution equivalence closure

本节更新 G0b-2 的 pending 状态；历史 V1--V6 文件、V6 frozen evidence 和
sealed-data provenance 没有被修改。

### 冻结的 E/U contract

机器可读的精确 contract 在
[`v7_g0b2c_eu_contract_manifest.json`](../configs/heat3d_v6_p1i/v7_g0b2c_eu_contract_manifest.json)。
其关键边界是：

- E：`anchor_context_resolution=1024`；E16384 以
  `encoder_input_resolution=16384`、`output_query_resolution=16384` 做 direct
  high-resolution query，E32768 使用对应的 `32768/32768` 规则；
- U-v2：`anchor_context_resolution=1024`、
  `encoder_input_resolution=1024`，保留 native conditioning graph、context 和
  anchor scale，再以 `output_query_resolution=16384` 做 output-side direct
  query。U-v2 不是 reconstruction-only route；
- E/U 都显式保存 `anchor_context_resolution`、`encoder_input_resolution`、
  `output_query_resolution`、`reconstruction_resolution` 和 `direct_query`。未来
  matched-query 比较固定为 `E(N_anchor, N_encoder,E -> N_query)` 对
  `U(N_anchor, N_encoder,U -> N_query)`；本轮没有 latency 或 superiority claim；
- global context、q-k/scale features 和 anchor scale 均来自 native 1024
  conditioning side；reconstruction 是 direct prediction 之后的可选/共同下游；
- fixed edge targets 保留 V6 route envelope。E16384 的 fixed targets 为
  `p2r=107645`、`r2p=107645`、`r2r_domains=3573`、`r2r_indices=3573`；
  U-v2 同时记录 native、query、combined model-input envelopes；真实 edge
  顺序不变，dummy padding 只补到 envelope。

### Temporary compatibility fixture 与证据边界

devbox 已找到 V6/P1i checkpoint、1024 dataset 和共享 full-field geometry；历史
high-resolution binary support/graph/reconstruction artifacts 未找到。WSL2 连接
在本轮不可用，因此 receipt 和 fixture 明确记录：

```text
temporary_compatibility_fixture_due_to_wsl2_unavailable = true
historical_artifact_reconciliation = pending_missing_identity_artifacts
wsl2_mirror_reconciliation = pending
fixture_label = V7 Refactor Compatibility Fixture
```

fixture 只在 devbox 内存中从冻结 `valid_iid` sample metadata、共享 geometry 和
label-independent nested-support protocol 构造；没有读取温度/target label，未
访问 `test_iid` 或 sealed labels，未写入大型 NPZ/cache。它不是 V6 historical
evidence 的替代物，也不具备 publication headline、G3 或 final-test 资格。
WSL2 恢复后必须按
[`v7_g0b2c_compatibility_fixture.md`](v7_g0b2c_compatibility_fixture.md) 的
deferred reconciliation checklist 核对历史 binary artifacts、support/graph
metadata、reconstruction maps 和 hashes；完成前状态保持 pending。

### Equivalence closure

最终 CPU semantic-oracle receipt 为
[`v7_g0b2c_receipt.json`](v7_g0b2c_receipt.json)，执行环境为 devbox 的
`jax=0.9.1`、`TFRT_CPU_0`。结果如下：

| route | closure | 关键结果 |
| --- | --- | --- |
| native-1024 | complete | support、raw metadata、inputs、graphs、native physics、context/scale、prediction 和 `s_hat` 均 `max_abs=0`、`RMSE=0`；prediction SHA 为 `7b559765309c31be16400679db84998c8e61c6e9c4a034c6a41185d0941962fd` |
| E16384 | complete with temporary fixture | support/metadata、fixed padding、model-visible tensors、raw prediction、query/anchor scale、final direct prediction 和 reconstruction field 均 `max_abs=0`、`RMSE=0`；prediction SHA 为 `59a27149604f7c2f0874e5cf38c8d36d05862ca15b2e22a048b0550518e3b61c` |
| U-v2 16384 | complete with temporary fixture | native conditioning 与 high-resolution direct-query metadata/graphs/model-visible tensors、raw prediction、query scale 和 reconstruction field 均 exact；prediction SHA 为 `65a9b1c8719431510dd7997d958ad7e99930e0b63e44c8be80c70b161c463066` |
| E32768 | temporary compatibility forward | 1 个固定 `valid_iid` sample 的 support、raw/padded metadata、model-visible tensors、prediction、scale 和 reconstruction map exact；未计算 metrics |

E16384 的 temporary support fingerprints 为 support indices
`2bc5c34317d70d7a5b67b0b218c13ff1cf0f02b51e226c1156c0712ae3584427`、query
coords `6bbb729925f844c90b3dc589027ab44273a09b6a2bebd78d7a6bdfd3f861dc74`；
E reconstruction map old/new 都为
`f4c058e83a2ed59a8228e84e85ab570b405db102386890e4dd925f556386f4b7`。U-v2
的 query metadata fields（包括 `p2r`、`r2p`、`r2r`、`rnodes` 和 query
coordinates）均 exact；其 output-side `r2p` 仍是 direct-query graph，不是
reconstruction substitute。

### G0b-2c 后的 dependency cutover

新的 [V7 E reference entrypoint](../scripts/run_heat3d_v7_high_n_reference.py)、
[V7 U reference entrypoint](../scripts/run_heat3d_v7_u_high_n_reference.py) 和
`rigno.heat3d_runtime` 不 import `scripts/`、`*_smoke.py`、
`*_development.py`，不修改 `sys.path`，不安装 V3 hook，也不依赖 V1 private
runner。`RuntimeSession`、`FeatureTransform`、`GroupBuilder`、`HighNRuntime` 和
`UHighNRuntime` 已承接 checkpoint/stats/config、显式 feature transform、V6
feature/group assembly、device placement、model apply、E/U high-N metadata 和
fixed padding。

旧 V6 formal production/high-N/timing 路径仍保持原状，仍可到达：

- V3 `install_checkpoint_feature_hooks` 及其对 V1 runner module 的 monkey patch；
- V1 `_make_v6_padded_groups_with_progress`、`_attach_*`、`_model_apply` 等
  private symbols；
- qualification `ModelRuntime`、E/U timing wrappers、legacy metrics/timing
  wrappers 和历史 development-named entrypoints。

因此本轮完成的是 V7 compatibility cutover 与 CPU equivalence closure，不是
整个 V6 formal path 的删除或重构；legacy freeze manifest 仍要求这些文件不改、
不删。

### Current limitations and deferred optimization candidates

下表只记录审计结论；本轮没有实施任何优化，尤其没有改变 batching、cache、JIT、
KD-tree、padding 或 reconstruction semantics。

| 项目 | current behavior | potential cost | candidate direction | semantics may change |
| --- | --- | --- | --- | --- |
| repeated `FeatureTransform` / raw extraction | 每个 group/case 仍重复 materialize | 重复 CPU/JAX work | session/sample/geometry/version keyed memoization | 是，stale key 会改变输入 |
| batch-scoped `Heat3DGraphBuilder` | 构造边界内重复创建 builder | setup 与 Python overhead | session-scoped builder | 低，但 config/seed 必须 exact |
| graph/KD-tree reuse | 没有 session/geometry-scoped reuse contract | 重复 metadata/KD-tree construction | geometry/config fingerprint reuse | 是，geometry/config drift 会改 graph |
| reconstruction partition/map | map 可复用边界依赖 route wrapper；各 wrapper 编排不同 | 重复 partition/map work | explicit route-scoped map cache | 是，map key 漏字段会改 full field |
| compiled/JIT executable cache | 没有明确 keyed executable cache | 重复 compile/warm-up | device/model/shape/padding signature cache | 是，signature 错误会改执行路径 |
| high-N Python orchestration | support、metadata、group 和 apply 仍多次跨 Python 边界 | high-N launch overhead | narrow reference executor | 是，调用顺序可能影响行为 |
| fixed-shape/padding memory | 保留 V6 fixed edge-target dummy rows | 高-N memory/padding waste | shape buckets 或 mask-aware representation | 高，dummy semantics 必须重证 |
| audit/reference mixing | legacy audit harness 允许 old imports；V7 reference path 不允许 | audit 依赖和 production 依赖容易混淆 | separate audit-only adapter boundary | 是，边界错误会污染 evidence |
| metrics/timing duplication | qualification、E/U、production、resolution 各有 wrappers | 指标/时序定义漂移风险 | named metrics/timing core | 是，指标命名/聚合会变化 |
| GPU reduction/aggregation | CPU semantic oracle exact；GPU repeatability 仍是 envelope | strict reproducibility 受 backend 影响 | deterministic control/backend policy | 是，可能改变性能与数值路径 |

### G0 判定更新

| 项目 | 状态 |
| --- | --- |
| E/U historical contract parsing 与 explicit resolution separation | 完成；manifest 已冻结 |
| native-1024、E16384、U-v2 16384 CPU old/new equivalence | 完成，但 E/U high-N 证据仍标记 temporary fixture |
| E32768 G0 compatibility forward | 完成；temporary fixture 限定，无 metrics/accuracy claim |
| historical identity-level artifact reconciliation | 未完成，`pending_missing_identity_artifacts`；WSL2 mirror 独立为 `pending` |
| G0 Code：全部 formal production/timing path 脱离 smoke/check/development/private API | 未通过；旧 V6 path 仍保留 legacy dependencies |
| G1 experiments / publication / final test | 未开始；本轮没有性能或 accuracy claim |

结论：**V7-G0b-2e 的 route-binding hardening 已完成；E/U historical
behavioral reconciliation 因 devbox 仅提供 train-bound replay plans 而按
valid_iid 边界 fail-closed，identity-level artifacts 仍是 archival limitation。
整体 G0 仍受旧 V6 formal dependency graph、historical replay population 和
formal metrics/timing core 约束；WSL2 mirror pending 不再作为唯一 hard blocker。**
本轮未训练、未运行 solver、未生成数据、未访问 held-out/sealed labels、未修改
模型或 V6 frozen artifacts。

## G0b-2f → G0c-1 → G0c-2 收口证据

本节覆盖严格按顺序执行的三个 gate；前一 gate 的 PASS 是后一 gate 的前置条件。
对应的 machine-readable receipts 是
[`G0b-2f`](v7_g0b2f_receipt.json)、[`G0c-1`](v7_g0c1_receipt.json) 和
[`G0c-2`](v7_g0c2_receipt.json)。本节 supersede 之前记录的“Evaluation/Timing
Core 尚未形成”中间状态；历史 V1–V6 文件本身没有被修改。

### G0b-2f：historical behavioral closure

四个历史 train sample 仅用于 software equivalence，且使用不读取温度文件的
`V7 Refactor Compatibility Fixture`：
`v6p1if1_0079`、`v6p1if1_0971`、`v6p1if1_0393`、`v6p1if1_0056`。receipt 固定
`execution_role=compatibility_audit`、`sample_role=train`、
`labels_read=false`、`metrics_executed=false`、`model_selection=false`、
`scientific_evidence_eligible=false`。这些样本不具有科学评价资格。

在冻结 historical execution commit
`8a812619ab0112b4ecfc37ef18189f731180059d`、checkpoint epoch 559 和原始 route
契约下，CPU historical replay 与 V7 stable replay 的 native-1024、E16384、
U-v2 16384、E32768 可比较项均为 `max_abs=0`、`RMSE=0`。比较覆盖 support/order、
query、raw/padded metadata、model-visible groups、context/scale、raw/direct
prediction、`s_hat` 和 deterministic reconstruction map/field。此处的
`E32768` 是 compatibility forward，不是 valid32 accuracy study。

devbox 保存的 CUDA prediction artifact 已确认处于
`reconstructed_full_field` 阶段、表示为 `deltaT_K`；没有使用统一的 `1e-2 K`
阈值强行判定，也没有把 CUDA artifact 的 binary identity 冒充为 behavioral
identity。原始 graph/support/cache identity bundle 不完整，因此：

```text
historical_behavioral_reconciliation = complete
historical_identity_artifact_reconciliation = unavailable_missing_original_artifacts
```

后者是 archival limitation，不再阻塞 G0。

### G0c-1：Evaluation Core

`rigno.heat3d_runtime.evaluation.EvaluationCore` 现在是唯一稳定评价实现。inference
runtime 只产生 prediction；truth loading、可选/共同 reconstruction、metrics 和
receipt 属于 Evaluation Core。Core 显式限制 `evaluation_split=valid_iid`，并以
V6 定义冻结以下六项指标：point-global relative RMSE、sample-first relative
RMSE、raw K/CV RMSE、source-region RMSE、peak RMSE、interface RMSE；同时保存
SSE、count、denominator/normalization quantities 和 per-region accumulation。

使用冻结 valid32 reference 做 legacy ↔ V7 replication，六项 metric 最大绝对差为
`8.881784197001252e-16`；原始 sufficient statistics 也已保存于 receipt。该次
执行是 `compatibility_audit`，不是新 scientific evaluation，不用于模型/route
选择。旧 metrics wrappers 降级为 read-only historical oracle；正式 V7 evaluation
边界只允许通过 Evaluation Core。

### G0c-2：Timing Core

`rigno.heat3d_runtime.timing.TimingCore` 统一并验证 lifecycle semantics，阶段顺序
固定为 preprocessing/feature → graph build → compile/warmup → model forward →
reconstruction → synchronized result。正式 workload boundary 是
`k/q/BC -> preprocessing/graph -> model -> reconstruction -> synchronized
240825 field`；truth loading、metrics、accuracy audit 明确排除。fresh、resident、
Q1、Q2、throughput 的 lifecycle 状态验证通过。

本 gate 只证明 timing boundary 与旧 V6 粗粒度字段的 correspondence；旧字段把
compile/warmup 与 forward 部分合并，因此没有生成新的 latency 数值或 speedup
claim。GPU policy 冻结为：CPU deterministic 是 semantic-equivalence oracle，
normal GPU 是 production/timing backend，deterministic GPU 仅 optional diagnostic。
旧 V6 timing wrappers 是 read-only historical oracle。

### Final G0 acceptance matrix

| acceptance item | status | evidence |
| --- | --- | --- |
| smoke/check/development excluded from V7 production graph | PASS | stable runtime/reference entrypoints static import audit |
| cross-script private API | PASS for V7 production graph | no `scripts` private import; `u_split.py` library-internal `_...` technical debt remains outside this cutover |
| monkey patch/module-state rewrite | PASS | explicit feature transform; no hook installation |
| semantic-critical config fail-closed | PASS | production preflight and route binding |
| route binding fail-closed | PASS | unregistered route and resolution/padding mismatch negative gates |
| CPU old/new equivalence | PASS | G0b-2f receipt; all listed routes `0/0` |
| historical behavioral reconciliation | PASS | G0b-2f receipt |
| missing binary identity artifacts | archival limitation | original graph/support/cache identity bundle unavailable |
| Evaluation Core | PASS | G0c-1 receipt; max metric diff `8.88e-16` |
| Timing Core | PASS | G0c-2 lifecycle/boundary receipt; no formal latency |
| GPU policy | frozen | no new nondeterminism experiment in this gate |
| provenance roles | PASS | compatibility audit explicit; inference remains prediction-only |
| test/sealed access | PASS | no test_iid or sealed labels touched |

### Remaining G0 limitations and stop condition

没有 stable V7 contract 的 hard blocker；仍保留三类明确限制：原始 historical
binary identity 缺失是归档限制；旧 V6 formal wrappers 仍作为 read-only compatibility
oracle，尚未删除或重构；旧 timing 字段较粗，尚不足以支持新的正式性能 claim。所有
feature/graph/KD-tree/JIT/batching/padding/reconstruction 优化均 deferred。G0c-2
完成后停止，不自动进入 G1。
