# V7 G1 training dependency audit

The registered V7 path is library-level:

`run_heat3d_v7_formal_p1i_training.py → rigno.heat3d_training.p1i → V7FormalTrainer → RIGNO`

Validation is a separate explicit boundary:

`valid_iid prediction outputs → evaluate_level_a_validation → EvaluationCore`

The inference/training model callback receives prepared inputs and train/valid
targets only where the training objective or validation callback explicitly
requires them. It does not load test_iid or sealed sample data. The manifest
validator checks split counts, so the audit records role-count metadata
inspection separately from sample/label materialization.

The historical V2/V4/V1 chain, `check_*` private helpers, V3 feature hooks and
legacy metric/checkpoint wrappers remain read-only compatibility oracles. The
V7 static audit reports zero smoke/check/development imports, zero cross-script
private imports and zero module-state monkey patches in the production
entrypoint/library graph.

The completed e200 qualification uses the same P1i data preparation, graph, B24/B32
batching and optimizer contract for Full and the explicit vanilla RIGNO
control. Full uses the frozen native shape--scale objective. Vanilla disables
the Full-only FiLM, native scale head, q/k regional feature input and decoder
bypass, and uses the explicit normalized-DeltaT MSE control objective. Both
are registered as seed-0 `budget_qualification_only` runs and are excluded
from formal G1 evidence. Full selected epoch 173 and Vanilla selected epoch 164;
the budget decision is recorded in
[v7_g1_budget_decision_receipt.json](v7_g1_budget_decision_receipt.json).

The resulting formal budget is fixed at 200 epochs with the registered
warmup-cosine horizon. The six base registry variants share this contract. The
Full/Vanilla parameter gap is 7.4486%, above the pre-registered 5% threshold, so
the capacity-matched Vanilla is registered as the seventh variant. The frozen
matrix is therefore `7 × 3 × 200 = 21` runs, all still unexecuted.

P1i support semantics are now frozen as source-layout-aware block/interface/surface
and CV-weighted geometry support. The historical `local_regions` field is a block
quota alias; native support does not use numeric q, temperature, labels, solver
output or model error. Therefore H2 is limited to source-layout awareness, not
source-amplitude awareness. The audit is recorded in
[v7_g1_support_semantics_audit.md](v7_g1_support_semantics_audit.md).

The `no_scale` qualification is explicitly physics-scale-only: the physics scale
is retained and only the learned residual correction is disabled. It is not a
direct-output architecture. The capacity-matched Vanilla uses width 100 for both
node and edge latents, with a 0.3505% parameter gap to Full. Both are one-epoch
non-publication qualifications, recorded in
[v7_g1_variant_qualification_receipt.json](v7_g1_variant_qualification_receipt.json).

The only small efficiency-boundary change in this freeze is an explicit
`validation_outputs_fn`: validation prediction and validation loss share one
model application, while the model, graph, support, normalization, loss and
batch semantics stay unchanged. Cross-run feature/graph reuse, broader JIT
cache reuse, support-variant implementation, high-N optimization and formal
multi-seed execution remain deferred. generic-uniform support and volume-only
support still lack separately frozen label-independent providers; no-context has
no uniquely frozen delta because native scale/context coupling is unresolved.
The entrypoint fails closed for these unresolved definitions; they are not
silently mapped to the source-aware Full route. Consequently, the epoch budget
and supported variant implementations are qualified, but `G1 SCIENTIFIC READY`
is not yet granted.

The fixed Full P1i semantic anchor is recorded in
[v7_g1_full_p1i_semantic_anchor_receipt.json](v7_g1_full_p1i_semantic_anchor_receipt.json).
Prepared/model-visible arrays, gradients, updates and parameter evolution are
exact on CPU across three steps. The gate remains `FAIL_CLOSED` because one
legacy-wrapper versus direct-component `total_loss` value differs by
`1.9073486328125e-6` at step 2; no new tolerance is registered.

The synchronized one-epoch Full and canonical Vanilla CUDA instrumentation is
recorded in
[v7_g1_synced_profiling_receipt.json](v7_g1_synced_profiling_receipt.json).
Each actual step ends after `block_until_ready` over state and outputs, and
truth/metrics are outside the timed step. CUDA delay-kernel warnings are kept as
a timing-calibration limitation, not turned into a performance claim.
