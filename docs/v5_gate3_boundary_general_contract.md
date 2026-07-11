# V5 Gate 3 Boundary-General Shape–Scale Contract

Gate 3 is a read-only diagnostic for frozen P5 targets and frozen V4P5_02
best/final raw-temperature predictions. It does not modify a model, loss,
training configuration, split, label, or P5 sample; it does not train or add a
learned scale branch.

## Frozen V5 Instance

The present V5 dataset uses bottom Dirichlet, top Robin, and adiabatic sides.
Its reference is the prescribed value of the metadata-selected `bottom`
Dirichlet region, so `DeltaT = T - T_bottom`. For every sample:

```text
scale = sqrt(sum(CV * DeltaT^2) / sum(CV))
shape = DeltaT / (scale + eps)
T = shape * (scale + eps) + T_bottom
```

`eps` is `1e-12 K`. Reconstruction and all boundary projection happen in raw
physical temperature space.

## Boundary-General Interface

The parser never assumes a coordinate location is Dirichlet. It reads each
region's type from `boundary_params`, its metadata point-index mask from
`boundary_regions`, and a matching binary mask from `bc_features`. The two
masks must match exactly. A Dirichlet region obtains its prescribed value from
metadata, and any number of arbitrary Dirichlet node sets can be projected.

The interface recognizes explicit `dirichlet`, `robin`, `neumann`, and
`adiabatic` region entries. Future mixed BC must be represented as several
explicit region entries; an ambiguous literal `mixed` type fails loudly rather
than falling back to bottom Dirichlet. Coordinate inference is disabled for
the frozen V5 run. If a future audit enables it, the metadata must provide an
explicit region-local `{axis, extremum, tolerance}` fallback.

Current P5 evidence covers only its bottom/top/side combination; this contract
does not claim generalization to arbitrary BC layouts.

## Oracle And Mechanism Diagnostics

For both frozen V4 best and final predictions, Gate 3 reports original,
predicted-shape plus true-scale (shape-only), true-shape plus predicted-scale
(scale-only), and boundary-projected original metrics. Metrics are
sample-first CV-weighted and cover clean, hard, and every formal P5 role.

The lateral fields are q-weighted local/inverse `kz`, q overlap with the
lowest-`kz` quintile, source-layer `kz` heterogeneity, source concentration,
and source z centroid. Their relations to the Gate 1 corrected 1D scale
residual are descriptive mechanism evidence only, not causal findings.

The machine-readable form is
`configs/heat3d_v5/v5_gate3_boundary_general_contract.json`.
