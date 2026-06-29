# V4 P3a Closeout

Scope: solver-path closeout only. No dataset generation, training, registry
change, or default runner change is included here.

## Production Decision

V4 production/default contact model is fixed to:

```text
R_contact=0_perfect_contact
```

Finite interface thermal resistance remains an experimental smoke path for
solver research. It is not part of the V4 dataset generator, default solver, or
production label path.

## Implemented Gates

- Perfect-contact sparse assembly keeps the dense-v2 legacy equivalence gate.
- Contact `R_contact=0` matches perfect-contact sparse output within `1e-10`.
- Finite `R_contact>0` smoke checks monotonic interface jump, finite residuals,
  exact bottom Dirichlet enforcement, finite temperatures, and contact metadata.
- Solution audit now records residual, source power, top Robin flux, bottom
  flux, energy-balance residual, operator checksum, solver mode, and backend.

## Deferred

- Finite contact resistance is deferred from V4 production data generation.
- Publication-facing contact validation belongs to P3b or later.
