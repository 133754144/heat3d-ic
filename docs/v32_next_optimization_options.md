# V32 next optimization options

Unique recommendation: audit and control the physics-attention residual strength before any multi-seed expansion. The preferred next experiment is a preregistered residual-strength control (for example, zero-initialized learnable residual scale) against the unchanged V32 path.

Rationale: point-global improved while sample-first regressed; attention is demonstrably non-uniform and non-collapsed, but its residual norm is about three quarters of the mean-pool norm and its source/q correlations are negative. This is an attention-bias question, not evidence for more seeds or a larger model.

This document does not authorize training or test/hard/sealed access.
