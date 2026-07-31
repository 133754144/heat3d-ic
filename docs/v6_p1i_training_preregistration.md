# V6-P1i seed0 training preregistration

`V6_05_V5best_P1i_seed0` freezes the V6_03 canonical architecture, four-term
loss, AdamW warmup-cosine schedule, 600 epochs, effective B24, seed 0, graph
policy and point-global true-RMS validation selection. The only scientific
change is the frozen `formal1024_v1` input distribution. P1i uses an explicit
loader and `3×B8 -> B24` execution adapter because supports vary by sample;
this is an execution constraint, not a new loss or model option.

Only train is optimized and only `valid_iid` selects checkpoints. The existing
`test_iid` is an audited holdout and is not materialized by the training
loader. The separately preregistered sealed IID confirmatory role has no
generated/opened labels and cannot be opened until training and model choices
are frozen.

Launch is authorized only after the HF archive, full-field sidecar, B8/B16
one-update/reload smokes, batch-order prefix, wrong-graph-reuse guard and
cross-dataset compatibility gate all pass.
