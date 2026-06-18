# V4 Research Definition

Read this file only when defining V4 research goals, checking experiment
alignment, or deciding whether a proposed run belongs in V4.

## One-Sentence Goal

构建面向 3D IC 多层异质结构的稳态三维热仿真代理模型，目标是向可发表论文级模型推进。

## Default Baseline

B88 sample_shuffle / AdamW warmup_cosine / latent96-edge96-s6-mlp2 /
discrete_physical_coverage + repair_none / plain mse / valid_base_mse
selection。

## V4 Work Tracks

1. Control plane: YAML 继承+生成器、remote launch/check/sync scripts、统一指标与 run registry。
2. Research core: V0 遗留路径审计、模型结构修复、solver 升级、跨分辨率泛化。
3. Optional extensions: 数据扩展、物理约束 loss、FNO/其他 baseline；只有在控制面和核心证据稳定后推进。

## Alignment Rules

- V4 experiments must improve evidence toward the one-sentence goal, not only
  optimize a convenient validation scalar.
- Changes must state whether they target control-plane reliability, model
  capacity, graph/solver fidelity, cross-resolution generalization, dataset
  realism, training stability, or paper-facing evaluation.
- Runs that do not connect to multilayer heterogeneous 3D IC steady thermal
  simulation should stay diagnostic and must not be treated as V4 progress.
