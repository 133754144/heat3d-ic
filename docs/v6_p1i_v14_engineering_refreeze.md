# V6-P1i v14 engineering refreeze

Pilot v13 was stopped after 24 solves because its sample prefix accidentally
remained `v6p1i12_`. No v13 qualification was performed and its partial target
values were not used to change the scientific generation rule.

V14 keeps the frozen v13 power curve, top-h compensation, independent jitter,
physics ranges, split method, support contract, and acceptance gates exactly
unchanged. It changes only the dataset/sample namespace and uses the new,
previously unseen Sobol seed `612816`. The input-only preflight and freeze must
pass before any v14 solve.
