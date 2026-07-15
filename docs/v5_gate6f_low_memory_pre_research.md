# Gate 6F low-memory pre-research

## Scope and access contract

- Branch: `research/v5-model-lab`.
- Execution host: `devbox`; WSL2 was not connected and the running V13 was not touched.
- Model/data roles: N3 best e402 and the frozen Gate 6D N3/L2 valid rows; only
  `train` and `valid_iid` were materialized for new work.
- `test_iid`, all hard roles, and sealed IID were not materialized, inferred,
  evaluated, selected, or tuned.
- No e600 or multi-seed run was started. The only parameter optimization in this
  gate is the explicitly allowed short training of small scale heads over frozen
  N3 features; the GNN never receives a backward pass.

## Valid-only amplitude diagnosis

The full machine-readable result is
`configs/heat3d_v5/gate6f/amplitude_valid_only.json`. N3 underestimates amplitude
in the highest true-CV-RMS quartile, but that bias is not concentrated in the
high q--low-k subgroup:

| valid_iid slice | N3 amplitude ratio | N3 signed log-scale bias | sample-first RMSE | point SSE K2 |
|---|---:|---:|---:|---:|
| true CV-RMS Q4 | 0.953105 | -0.064927 | 21.0632% | 2685.075 |
| q--low-k overlap Q4 | 1.004356 | -0.001791 | 16.8694% | 616.738 |
| high true CV-RMS and high q--low-k intersection, 11 samples | 0.991624 | n/a | n/a | n/a |

The Spearman correlation between q--low-k overlap and N3 amplitude
underestimation is `-0.042140`. Therefore the descriptive valid-only answer is
`high_temperature_underestimation_concentrated_in_high_q_low_k=false`.

## Existing q--k information paths

1. The 24D sample-global context already contains `q_weighted_local_kz`,
   `q_weighted_inverse_kz`, and `q_low_k_overlap_fraction`; it reaches both
   Global FiLM and the scale head.
2. The local decoder bypass retains node-local `k_x/k_y/k_z/q` plus the four BC
   masks.
3. The post-FiLM pooled latent carries q--k interactions created by encoder and
   processor message passing, but ordinary mean pooling discards their regional
   alignment.
4. The new q--k gated route derives 11 regional features only from raw
   `coords/k/q/BC` and the fixed P2R assignment. No target-derived quantity is an
   attention input.

## Frozen feature cache

The tracked manifest is
`configs/heat3d_v5/gate6f/frozen_feature_cache_manifest.json`.

| item | value |
|---|---|
| N3 checkpoint | best e402, SHA256 `3baebb9b751bf6054f36308444cdefe7a7b4f343665164b0aabdfe2610b5a228` |
| train artifact | 672 samples, SHA256 `ff41392ecc6ac508b7964bb8e3df56d9cfc88ca1a748a57c93a781dcdcfdf9bf` |
| valid artifact | 128 samples, SHA256 `445bcb77fcee12abc5a6233e15f1068dfadaa64f5046727b4e681cf19aa7cb8b` |
| rnodes | pre/post-FiLM `[N, 256, 96]` |
| other cached fields | 24D global context, q--k regional features, `phi_hat`, `s_phys`, `s_true` |
| cache peak host RSS | 3680.844 MB |

The context standardizer is train-only. The cache run is inference-only with
`gnn_backward=false`. On CPU, the default-disabled Gate 6F controls reproduce
the N3 parameter leaf set and `deltaT_hat/phi_hat/s_hat/pooled_rnodes` bitwise:
all maximum absolute differences are exactly zero.

## Frozen scale probes

`XLA_PYTHON_CLIENT_PREALLOCATE=false` was fixed for the low-memory result in
`configs/heat3d_v5/gate6f/frozen_probe_screen_lowmem.json`.

| rank | probe | scale log-RMSE | fixed-shape joint point-global RMSE | params | peak device MB |
|---:|---|---:|---:|---:|---:|
| 1 | mean | 0.172099 | 23.5378% | 7,809 | 514.0 |
| 2 | deep scale head | 0.176617 | 23.4498% | 16,129 | 514.0 |
| 3 | mean+max | 0.180698 | 24.1339% | 13,953 | 514.0 |
| 4 | mean+std | 0.181192 | 23.9174% | 13,953 | 514.0 |
| 5 | q--k gated pooling | 0.185044 | 24.3396% | 28,226 | 514.0 |
| 6 | latent attention pooling | 0.188339 | 24.3474% | 14,082 | 514.0 |
| 7 | pre-FiLM mean+std | 0.293703 | 29.0415% | 13,953 | 514.0 |

This short frozen-head screen favors the existing mean pool. It is a screening
result, not a full-model performance claim.

## q--k gated pooling and decoupling controls

The q--k gated design is `mean(post_film_rnodes) + attention_residual`. Its
attention logits and residual projection are zero-initialized, so initialization
is the exact mean path. The attention features are regional aggregates of raw
source intensity, inverse `k_z`, their interaction/overlap, source z location,
BC fractions, top-h, and bottom-temperature offset.

`pooled_latent_stop_gradient=false` and `scale_head_lr_multiplier=1.0` remain the
defaults, preserving N3/V13. The independent V21 e1 smoke is the only prepared
combination with stop-gradient enabled and multiplier `1.5`; it does not change
any existing N3/V13 YAML.

## Full-model e1 smoke

The registered low-memory sequential devbox run used
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. At the external SSH interruption:

- V14 and V15 had written complete `loss_summary.json` files.
- V14 passed finite gradients, native runtime width/positivity, train-only
  context, and final/best checkpoint reload. Its peak device memory was
  `8190.0 MB` and peak host RSS was `4327.4 MB`.
- V16 had started; V16--V21 final status had not yet been collected.

This section is explicitly partial and must be replaced by the eight-row
collector output after devbox connectivity is restored. It is not an e1
closeout and does not authorize a longer run.

`long_training_started=false`.
