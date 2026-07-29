# V6 hard/OOD closeout

The canonical checkpoint and 4096/8192/16384 Anchor-derived workflow were frozen before this descriptive evaluation. 32768 was excluded.

Canonical P1h has no registered distribution-shift OOD role, so no OOD labels were opened. `hard_input_stress` is a preregistered IID stress subgroup within the already-opened corrected confirmatory holdout, not OOD.

| Mode | Role | Support point-global % | Full point-global % | Full RMSE K | Peak K | Source K | Layer/interface K | Bias K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | valid_iid | 1.3359 | 3.5241 | 1.4290 | 1.5608 | 0.8609 | 0.7912/0.6191 | 0.0233 |
| 4096 | corrected_confirmatory_holdout | 1.4669 | 3.5874 | 1.4550 | 1.0373 | 1.0240 | 0.8496/0.6765 | 0.0311 |
| 4096 | hard_input_stress | 1.2528 | 1.6978 | 0.9866 | 1.2415 | 1.2211 | 0.8198/0.8041 | 0.0280 |
| 8192 | valid_iid | 1.4189 | 3.0214 | 1.2252 | 1.8559 | 0.9375 | 0.7471/0.6228 | 0.0299 |
| 8192 | corrected_confirmatory_holdout | 1.5437 | 3.0976 | 1.2563 | 1.2082 | 1.0158 | 0.8014/0.6872 | 0.0389 |
| 8192 | hard_input_stress | 1.2966 | 1.6211 | 0.9420 | 1.3120 | 1.1578 | 0.8017/0.8003 | 0.0317 |
| 16384 | valid_iid | 1.6443 | 2.7980 | 1.1346 | 1.7962 | 1.1748 | 0.7855/0.7099 | 0.0449 |
| 16384 | corrected_confirmatory_holdout | 1.7704 | 2.8652 | 1.1620 | 1.2075 | 1.2155 | 0.8336/0.7693 | 0.0473 |
| 16384 | hard_input_stress | 1.4805 | 1.7853 | 1.0374 | 1.3173 | 1.3469 | 0.8858/0.8984 | 0.0370 |

The hard and corrected-confirmatory rows are descriptive only and cannot change the canonical model, checkpoint, resolution roles, or any tuning decision.
