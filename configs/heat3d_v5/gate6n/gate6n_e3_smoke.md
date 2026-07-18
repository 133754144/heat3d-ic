# Gate 6N e3 execution smoke

`V4P5_37_gate6n_v36_r2r_mask_p005_smoke_e3` completed three epochs on WSL2
at training commit `c792a61`. It used the formal P5 train/valid_iid split
(672/128), 1024 nodes, train B28 and validation/prediction B32.

- r2r-only edge masking: p=0.05, deterministic epoch/batch keys.
- No OOM, NaN or Inf; gradients and runtime architecture checks were finite.
- All five saved checkpoint kinds replayed successfully. Parameter reload was
  exact and the worst recovered-field replay error was 0.001434326171875 K
  against a 0.02 K tolerance.
- Global-context fitting remained train-only with 672 samples.
- test/hard groups were not built and no test/hard inference ran.
- Best base MSE was 0.3042105734 at epoch 2; final base MSE was 0.3082233667.
  These are smoke diagnostics, not formal model-performance results.
- The formal V38 e600 candidate remains `not_started`.
