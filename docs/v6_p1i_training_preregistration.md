# V6-P1i seed0 training preregistration

`V6_05_V5best_P1i_seed0_B24` freezes the V6_03 canonical architecture,
four-term loss, AdamW warmup-cosine schedule, 600 epochs, effective B24, seed 0,
graph policy and point-global true-RMS validation selection. The only scientific
change is the frozen `formal1024_v1` input distribution. P1i uses the native
V6best execution contract: one real B24 forward/backward/update per optimizer
step, with validation and prediction batches of 32.

Only train is optimized and only `valid_iid` selects checkpoints. The existing
`test_iid` is an audited holdout and is not materialized by the training
loader. The separately preregistered sealed IID confirmatory role has no
generated/opened labels and cannot be opened until training and model choices
are frozen.

The superseded B8/validation-B16 launch was stopped before its first epoch and
is retained only as a historical record. Launch of the corrected YAML is
authorized only after the HF archive, full-field sidecar, B24 resolved-config
check, batch-order prefix, wrong-graph-reuse guard and cross-dataset
compatibility gate all pass.

## Future YAML batch defaults

Unless a later experiment explicitly preregisters another contract, new YAML
profiles use `micro_batch_size = batch_size`,
`validation_batch_size = 32`, and `prediction_batch_size = 32`. The runner and
dry-run command builder resolve omitted/null micro-batch values to the training
batch size and omitted/null validation or prediction values to 32.
