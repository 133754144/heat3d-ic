# V6-P1i v9 population power calibration

V8 missed only the frozen occupancy-ratio gate (5.333 versus 5.0). The v9
revision remains a single monotonic population rule: base power 2.6--11.2 W,
severity exponent 0.71, top-h exponent 0.6, and independent multiplier
0.75--1.25.

Linear counterfactual scaling on four completed, independent pilot populations
(v5--v8) passed every frozen temperature gate for this rule. V9 uses a new
Sobol seed that was not inspected when the rule was frozen. No individual
sample is backsolved, filtered, replaced, or assigned using temperature.
