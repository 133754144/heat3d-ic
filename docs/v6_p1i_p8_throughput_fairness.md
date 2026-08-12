# V6 P1i P8 throughput fairness

All data are frozen valid32 on WSL2. Worker startup is outside steady spans; KDTree workers per case are fixed to one.

## CPU preprocessing backends

| backend | steady wall (s) | samples/s | exact |
|---|---:|---:|---:|
| serial | 31.194521 | 1.026 | True |
| thread2 | 15.525602 | 2.061 | True |
| thread4 | 16.277224 | 1.966 | True |
| thread8 | 17.393331 | 1.840 | True |
| process2 | 29.949465 | 1.068 | True |
| process4 | 18.671785 | 1.714 | True |
| process8 | 14.595083 | 2.193 | True |

## Persistent FVM

| processes | startup (s) | steady valid32 (s) | samples/s |
|---:|---:|---:|---:|
| 1 | 4.920468 | 55.626921 | 0.575 |
| 2 | 4.971798 | 37.905397 | 0.844 |
| 4 | 4.902212 | 50.037438 | 0.640 |
| 8 | 5.138958 | 57.578713 | 0.556 |

Winning exact neural backend is `process8`. Best fresh neural cell is B16 at 2.808 samples/s. Persistent FVM saturates at P=2 and 0.844 samples/s. Publication-safe throughput ratio: **3.33x**.

Fresh, streamed-prepared, resident, and persistent-FVM semantics remain separate. No training or test/sealed access occurred.
