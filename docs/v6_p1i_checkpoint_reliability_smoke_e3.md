# P1i checkpoint reliability e3 smoke

The real `formal1024_v1` train/valid path ran three epochs on devbox with 32
real B24 updates per epoch. Test and sealed IID remained closed. Losses,
gradients and predictions were finite; no OOM, NaN or Inf occurred.

The runner atomically saved optimizer-aware epoch-0 best variants and latest,
then updated best/latest through epochs 1–3 and wrote final before any
post-training metadata. The deliberate metadata failure occurred only after
the final and best prediction archives existed. Despite the expected non-zero
exit, `params_best.pkl`, `params_latest.pkl` and `params_final.pkl` all retained
params, optimizer state, epoch and complete best-state records; immediate
parameter and optimizer reload error was exactly zero.

| epoch | elapsed s | valid base MSE | point-global true-RMS |
|---:|---:|---:|---:|
| 1 | 307.67 | 1.48 | 46.06% |
| 2 | 147.14 | 0.235 | 18.36% |
| 3 | 206.12 | 0.0584 | 9.15% |

The formal-training checkpoint gate passed. The injected failure option is set
only in the smoke YAML and is absent from all e600 configurations.
