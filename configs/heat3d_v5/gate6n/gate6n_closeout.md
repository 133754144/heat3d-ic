# Gate 6N closeout

Gate 6N is complete through a real P5 e3 execution smoke. No formal e600 run
was launched.

The upstream RIGNO default is p=0.5 and uses the same random
shuffle-and-prefix mask on p2r, r2r and r2p when a PRNG key is supplied. The
Heat3D runner previously never supplied that key, so its configured masking
was inactive. The fixed runner now derives a deterministic runner key from
model_seed/epoch/batch only for gradient-producing train updates. The audit
then reproduces the remaining real call chain: group-index fold-in, native
call split, encoder split and Processor split. Validation, prediction and
non-update metric passes receive no key and therefore use the complete graph.

Heat3D's train topology has 256 regional nodes and r2r min/median/P95/max
in-degree 5/13/25/41. Exhausting all 600×24=14,400 masks from the exact
Processor key schedule at p=0.05 found zero zero-in-degree events, zero
zero-out-degree events, zero isolated nodes and zero disconnected masks; the
maximum weak-component count was one. Gate 6N therefore retains p=0.05 on r2r
only; p2r and r2p remain complete even during training. The p=0.02 fallback
and e3 rerun were not required.

The p=0 old/new runner regression also matched exactly for loss, gradients,
updates and post-update parameters under the V36 AdamW first-step setup.

The e3 smoke passed finite-gradient, checkpoint/prediction save, five-way
checkpoint replay, 1024-node, B28/B32 and train-only context-fit checks.
V38 differs scientifically from V36 only in the two edge-masking fields. Its
e600 lifecycle remains `not_started`.
