# V6-P1i pilot128_v3 closeout

Status: **generated complete; qualification failed**.

The frozen 128 cases were retained exactly as generated. No sample was
filtered, replaced, or power-backsolved, and no training or model inference was
run.

## Evidence

- 121/128 (94.531%) peak values are inside 30--150 K.
- One sample (`v6p1i3_0036`) is outside 20--180 K at 203.449 K.
- The 12-bin nonzero max/min ratio is 6.0, above the frozen limit 5.0.
- The core q05--q95 maximum gap is 9.389 K, above 8.0 K.
- The full maximum gap is 27.707 K, above 15.0 K.
- All split, conservation, residual, finite-value, source-resolution, local
  support, and physical-response gates pass.

The independent power perturbation and source-size decoupling worked:
severity--source-area Spearman is 0.030, severity--peak is 0.864, and
source-area--peak is -0.017. Top-h response remains physically visible after
controlling power.

## Decision

`formal1024_v1` remains forbidden. The next attempt must use a new dataset ID
and seed and may only revise a global pre-solve generation rule. The v3 sample
set is a permanent negative qualification artifact and cannot be repaired by
removing or replacing its tail sample.
