# V4 Research Definition

Read this file only when defining V4 research goals, checking experiment
alignment, or deciding whether a proposed run belongs in V4.

## One-Sentence Goal

构建面向 3D IC 多层异质结构的稳态三维热仿真代理模型，目标是向可发表论文级模型推进。

## Default Baseline

B88 sample_shuffle / AdamW warmup_cosine / latent96-s6-mlp2 / discrete radius / mse。

## Alignment Rules

- V4 experiments must improve evidence toward the one-sentence goal, not only
  optimize a convenient validation scalar.
- Changes must state whether they target model capacity, graph coverage,
  dataset realism, training stability, or paper-facing evaluation.
- Runs that do not connect to multilayer heterogeneous 3D IC steady thermal
  simulation should stay diagnostic and must not be treated as V4 progress.
