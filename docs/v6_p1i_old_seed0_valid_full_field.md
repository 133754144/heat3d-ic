# P1i old seed0 valid full-field closeout

Only `valid_iid` was read. Test and sealed IID remained closed. The oracle row
is the reconstruction-only sampling floor from exact original 1024 solver nodes;
model rows combine prediction error and the same 240825-node reconstruction.

| kind | point-global true-RMS % | sample-first CV % | raw CV RMSE K | peak RMSE K | source/background K | layer mean/drop K | interface K | top/bottom K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reconstruction_only | 3.066851 | 3.584837 | 2.730219 | 0.324126 | 0.838549/2.733729 | 0.993753/1.431863 | 1.004971 | 0.106948/0.104791 |
| point_global_best | 3.439267 | 3.963344 | 2.990927 | 3.905090 | 2.891310/2.991205 | 1.450669/1.435661 | 1.471860 | 0.766220/1.115488 |
| final | 3.451445 | 3.972401 | 2.993689 | 3.960272 | 2.910357/2.993922 | 1.468631/1.439769 | 1.496799 | 0.776320/1.150525 |

Worst samples are recorded in the machine-readable JSON.
