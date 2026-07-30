# V6-P1i pilot128 v2 preregistration

V1 passed all solver and conservation checks but failed its frozen temperature
coverage gate. A single global revision is preregistered for v2:

- top Robin range is narrowed from 500--2500 to 500--1600 W/(m2 K);
- a continuous severity maps to reference power 2.8--12.6 W;
- actual package power is the reference power multiplied by
  `(top_h / 894.4271909999159)^0.79`;
- high-severity cases use continuously larger source planforms through a fixed
  0.45 blend weight.

The exponent is the global v1 log-regression coefficient
`log(DeltaT_peak) ~ log(P) - 0.791 log(top_h) - 0.029 log(bottom_h)`, whose
three-variable fit had R2=0.976. It is used only to define one pre-solve
population rule. No per-sample Rth is predicted or inverted, and no solved
sample can be removed or replaced.

The v1 acceptance contract remains unchanged. Sobol seed 612803, generator
code, config and all ranges must be committed and pushed before the first v2
solve. Dry preflight covers all 128 cases without solving and reports:

- package power 2.122--18.101 W;
- q 1.152e9--3.178e10 W/m3;
- at least 240 solver control volumes per source;
- at least 4 projected nodes per local source/k region.

V2 completion does not automatically authorize formal1024 generation; its
distribution audit must be reported first.

