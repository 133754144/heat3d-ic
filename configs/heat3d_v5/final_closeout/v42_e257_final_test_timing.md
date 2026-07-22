# V42 e257 final test_iid evaluation and batch-1 timing

Frozen checkpoint only; test_iid was opened after V5 selection was complete. No hard/sealed access.

## Core metrics

| split | point-global | sample-first | raw CV RMSE K |
|---|---:|---:|---:|
| valid_iid (frozen) | 21.936815% | 19.250517% | 0.156347 |
| test_iid | 23.249616% | 19.497664% | 0.201885 |

## Timing

| path | mean ms | median ms | P90 ms | N |
|---|---:|---:|---:|---:|
| model forward | 342.084 | 344.911 | 355.428 | 128 |
| end-to-end | 438.097 | 441.093 | 459.833 | 128 |
