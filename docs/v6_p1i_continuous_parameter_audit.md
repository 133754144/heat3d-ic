# V6-P1i continuous-physics parameter audit

## Scope and inheritance boundary

P1i is a separately named dataset family. It does not modify P1h, any frozen
V6 checkpoint, or any V6 evaluation result. The geometric stack and 240825-node
layer-aligned FVM mesh are inherited only as a physical reference.

Every layer has an explicit `background_k_xyz_W_mK`. A local conductivity
region may overwrite nodes inside its active-die mask; every other node retains
the sampled background of its own layer. There is no scalar, global, or
hard-coded conductivity fallback. Omitting a layer background is a fatal schema
error and is covered by the checker.

The complete machine-readable layer table is
`configs/heat3d_v6_p1i/v6_p1i_background_k_contract.csv`. The table separates:

- reference value: a published or previously frozen package-model value;
- continuous range: either a directly supported material bracket or an
  explicitly labelled engineering uncertainty envelope;
- distribution: linear, log-uniform, isotropic, or correlated anisotropic;
- fallback: always the explicit sampled value of the same layer.

## Continuous design

The pilot uses one scrambled Sobol design. Package power, top/bottom Robin
coefficients, source geometry, source power fractions, layer allocation,
background conductivities, local conductivity regions, and their values are
continuous functions of Sobol coordinates. Integer source/region counts are
the only unavoidable discrete variables.

There is no four-bin power or temperature table, no BC-specific power range,
no per-sample thermal-resistance inversion, and no post-solve filtering or
replacement. Power is sampled before solving; volumetric `q` is then obtained
by exact conservation over each resolved source volume.

## Physical correlations retained

- anisotropic effective layers use a shared latent for in-plane and
  through-plane conductivity;
- local source power is a continuous positive multiplier times source area,
  then normalized to the preregistered package power;
- source count, total active area, source volume, surface power density, and
  volumetric `q` remain algebraically linked;
- all samples share the same explicit package stack, while the sampled
  material and Robin values remain inside their frozen envelopes;
- split assignment hashes sample identity only and never reads a target.

## Evidence boundary

The package-table values are references, not direct measurements of every
possible package. Narrow ranges around PCB, BT substrate, interposer, and
bump/underfill are therefore labelled engineering uncertainty envelopes.
Silicon, TIM, and copper endpoints have direct support from the cited thermal
model literature. No undocumented value is promoted to a literature range.

