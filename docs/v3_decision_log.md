# Heat3D v3 Decision Log

## Current State

- Best scalar validation reference: B6 best.
- Best raw mechanism reference: S3 final.
- Main unresolved question: whether B6 e400 is undertrained or whether the
  scalar/raw metric mismatch reflects objective or decoder-path behavior.

## S4 / B6-e600

S4 extends B6 from e400 to e600 with the same model, graph policy, seed,
batch plan, optimizer, learning rate, warmup cosine schedule, and min LR.
The only intended experimental variable is the epoch count.

Purpose:

- test whether B6 e400 stops before its scalar and raw mechanism metrics have
  converged;
- compare B6-e600 against B6-e400, S2, and S3 before launching new schedule
  exploration;
- keep this as a diagnostic run, not a formal benchmark.

## Hold Decisions

- Do not start P5 pointwise/local decoder work yet.
- Do not start P7 loss/objective changes yet.
- Wait for S4 plus paired per-sample mismatch evidence before deciding whether
  the next move is more training, decoder/path audit, or objective alignment.
