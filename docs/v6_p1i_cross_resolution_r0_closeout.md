# V6 P1i cross-resolution R0 closeout

This valid-only closeout formally names Stage A a measure-conservative full-graph re-discretization diagnostic. It is not checkpoint-IID and not a formal same-distribution invariance result.

## R0 exact checkpoint replay

- fixed-32 support PG: 1.754522%
- full-128 support PG: 2.110188%
- formal replay tolerance: 0.005 (passed)
- frozen original 1024 coordinates, pointwise k/q, control-volume weights, Global Context, QK/scale inputs, graph config and graph seed were replayed without re-discretization.

## R0 to R1 discontinuity

- support PG jump: +11.921308 percentage points
- common full-field PG jump: +10.980968 percentage points
- R1-vs-R0 prediction drift on the common full field: 13.137264%
- graph hash equality fraction: 0.000000
- pointwise coords/k/q/weights equality fractions: {'coords': 0.0, 'k_xyz': 0.0, 'q': 0.0, 'weights': 0.0}

The R0-to-R1 jump demonstrates that the prior N=1024 re-discretization cell is already outside the checkpoint support/measure contract. Oracle improvement with N remains valid, while model degradation is associated primarily with support/measure, p2r graph-scale and context/scale-response drift; Nr growth remains secondary.

## Governance

- test accessed: false; sealed accessed: false; training/tuning: false.
- Direct-N cells remain diagnostics only and cannot be used for model selection or production speedup claims.
