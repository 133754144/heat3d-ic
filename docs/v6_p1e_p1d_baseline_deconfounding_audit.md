# V6-P1e P1d baseline deconfounding audit

This is a read-only audit of the frozen P1d tracked artifacts.  No solver,
generator, model training, or model inference was run.

## Finding

P1d is retained as provenance but is not qualified as the formal V6 training
dataset.  It must be replaced by a deconfounded P1e dataset.

- top-h versus power: Pearson `0.885861`, Spearman `0.912070`;
- bottom-h versus power: Pearson `0.528623`, Spearman `0.603758`;
- top/bottom/area linear prediction of power: R2 `0.785541`;
- all 1024 samples have exactly eight sources;
- all samples use the same nominal equal-area rule and exactly equal source
  powers; solver-grid realization can make declared source areas differ slightly;
- no fixed-power, fixed-geometry sweep independently varies top and bottom h;
- no group-locked train/IID/OOD split map exists.

Therefore the balanced temperature histogram in P1d does not remove BC-power
coupling.  P1e will use common power levels for every BC family, pre-solve
orthogonal pairing, variable source geometry, and group-locked split/OOD roles.
