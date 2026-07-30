# V6-P1i formal1024 protocol

Formal1024 is a new dataset, not an in-place expansion of P1h or pilot128.
It freezes the accepted pilot-v2 physics and continuous sampling rule while
using a new scrambled Sobol seed (`612804`) and sample prefix (`v6p1if_`).

The 1024 samples are assigned 768/128/128 to train/valid_iid/test_iid without
targets. Each consecutive Sobol octet is independently hash-ordered and
contains six train, one valid and one test sample. This preserves exact counts
and local Sobol coverage without consulting solved temperatures.

Generation is resumable only from a complete per-sample directory whose
dataset ID, Sobol index, split role and file set match the frozen contract.
Resume cannot replace or selectively regenerate a solved sample.

Acceptance covers:

- all 1024 finite FVM solves, residual and energy conservation;
- explicit background-k and local source/k support coverage;
- continuous peak/mean/CV-RMS distribution and absence of four-bin clustering;
- preregistered parameter and temperature split-consistency statistics;
- exact sample/file hashes and zero model training or inference.

No sample may be removed based on temperature, solver output or model error.

