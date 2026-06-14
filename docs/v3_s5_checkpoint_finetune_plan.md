# Heat3D v3 S5 Checkpoint Fine-Tune Plan

Purpose: prepare a small checkpoint-based fine-tune matrix after S5 without
starting training in this task.

S5 produced two useful checkpoint baselines:

- `params_best.pkl`: scalar-oriented checkpoint selected by valid loss.
- `params_final.pkl`: raw/stress-oriented checkpoint with stronger final raw
  diagnostics.

The first fine-tune batch uses constant `lr=1e-5` because S5 ended near the
`5e-5` learning-rate range; starting at `1e-4` would be an unnecessary jump for
this first pass.

Prepared configs:

| run | checkpoint | p_edge_masking | lr | epochs | purpose |
| --- | --- | ---: | ---: | ---: | --- |
| `S5best_FT_e100_lr1e-5_nomask` | S5 best | 0.0 | 1e-5 | 100 | scalar-oriented low-lr continuation |
| `S5final_FT_e100_lr1e-5_nomask` | S5 final | 0.0 | 1e-5 | 100 | raw/stress-oriented low-lr continuation |
| `S5best_EM_e100_lr1e-5_edgemask0p02` | S5 best | 0.02 | 1e-5 | 100 | conservative graph robustness check |
| `S5final_EM_e100_lr1e-5_edgemask0p02` | S5 final | 0.02 | 1e-5 | 100 | conservative graph robustness check |

This is configuration preparation only. It does not modify the model, decoder,
loss, objective, or graph policy, and it does not start training.
