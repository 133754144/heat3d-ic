# V7 G1 training dependency audit

The registered V7 path is now library-level:

`run_heat3d_v7_formal_p1i_training.py → rigno.heat3d_training.p1i → V7FormalTrainer → RIGNO`

Validation is a separate explicit boundary:

`valid_iid prediction outputs → evaluate_level_a_validation → EvaluationCore`

The inference/training model callback receives prepared inputs and train/valid
targets only where the training objective or validation callback explicitly
requires them. It never enumerates or loads test/sealed sample directories.

The historical V2/V4/V1 chain, `check_*` private helpers, V3 feature hooks and
legacy metric/checkpoint wrappers remain read-only compatibility oracles. The
V7 static audit reports zero smoke/check/development imports, zero cross-script
private imports and zero module-state monkey patches in the production
entrypoint/library graph.

The e200 qualification uses the same P1i data preparation, graph, B24/B32
batching and optimizer contract for Full and the explicit vanilla RIGNO
control. Full uses the frozen native shape--scale objective. Vanilla disables
the Full-only FiLM, native scale head, q/k regional feature input and decoder
bypass, and uses the explicit normalized-DeltaT MSE control objective. Both
are registered as seed-0 `budget_qualification_only` runs and are excluded
from formal G1 evidence.

The only small efficiency-boundary change in this freeze is an explicit
`validation_outputs_fn`: validation prediction and validation loss share one
model application, while the model, graph, support, normalization, loss and
batch semantics stay unchanged. Cross-run feature/graph reuse, broader JIT
cache reuse, support-variant implementation, high-N optimization and formal
multi-seed execution remain deferred.
