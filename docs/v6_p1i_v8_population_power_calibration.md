# V6-P1i v8 population power calibration

The v8 change is one target-independent global generation rule, not a
sample-selection rule. For each completed v3--v7 case, the steady linear heat
equation permits an exact counterfactual rescaling of peak temperature by the
ratio of proposed to original total power while holding geometry, material and
boundary conditions fixed.

A preregistered grid compared monotonic severity exponents, global base-power
ranges and independent multiplier ranges. The selected rule is:

- base power 2.8--11.0 W;
- monotonic severity exponent 0.8;
- top-h exponent 0.6;
- independent log-uniform multiplier 0.75--1.25.

This rule had the best aggregate frozen-gate score across v3--v7 and passed all
temperature gates counterfactually on the three most recent independent
populations (v5--v7). Older v3/v4 remain failures and are not reinterpreted.

No v8 output was available when this rule was frozen. No individual power was
backsolved from Rth, and no case may be filtered, replaced, or reassigned after
solving.
