# V6-P1h shared-support acceptance

Status: **passed**. P1h preserves all 1024 P1g samples, 128 groups, physical inputs,
solver outputs and group-locked splits. Only the ordered operator support changed.

## Frozen identities

- coordinate SHA256: `2bda8e710c8c9f15b180783dc12132280253124688d3e3177296e97527798745`
- graph SHA256: `6d3d62830755872194766aad2a8ac7b0f1fabec57840dac78fcb2642a6ed771c`
- support-index SHA256: `6907f9bc23ffdb9822494e24669527c2589a4fb8fd35eac93f03c16cc9604304`
- full-field archive SHA256: `f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00`
- full-field archive: 1284507133 bytes; exact per-sample q and T rows
- durable dataset path: `/Users/xuyihua/.codex/worktrees/5c97/3D IC Heat/data/heat3d_v6_p1h_shared_support1024_v0` (manifest/archive hashes verified)

## Replay and support

- representative replay cases: 8; P1g files checked: 10240
- replay coordinates/k/q error: 0; projected T and solver metrics satisfy the frozen tolerances
- selected proposal: `source_dense_16x16_v1` using geometry-only source-domain coverage
- source coverage min/p05/median: 4/5/10; zero-covered sources: 0
- all 9 layers, all 8 interfaces, and 64 top + 64 bottom Robin nodes are covered

## Projection and conservation

- full-field reconstruction CV-RMSE median/p95: 1.075332/2.209430 K
- relative CV-RMSE median/p95: 3.308%/4.585%
- solver peak minus projected peak median/p95/max: 0.050657/0.192276/0.375363 K
- max layer-mean/drop error p95: 2.509728/3.924158 K
- maximum absolute energy-balance relative error: 1.203e-10

## Leakage and trainability

Support proposal selection used no temperature or test label. The B24 smoke
materialized train+valid only, fit normalization/global context on 768 train
samples, completed a finite forward/backward/AdamW update, and reproduced both
parameters and loss exactly after checkpoint reload. No formal training was
started and P1g remains canonical.
