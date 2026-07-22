# V5 final Q4 root-cause audit

Read-only scope: train/valid_iid. Coverage uses only the frozen 24D input-derived context; no test/hard/sealed access.

## Q4 decomposition

| model | total SSE | Q4 SSE | Q4 shape | Q4 scale | Q4 cross |
|---|---:|---:|---:|---:|---:|
| V38 | 2606.999944 | 2060.561002 | 1251.157167 | 941.209813 | -131.805978 |
| V42 | 2605.075914 | 2151.275789 | 1071.576234 | 1175.539533 | -95.839979 |
| V44 | 2634.467621 | 2225.700281 | 1146.412003 | 1267.105937 | -187.817659 |
| V45 | 2895.095304 | 2408.174403 | 1199.952911 | 1425.718288 | -217.496796 |
| V46 | 3019.012428 | 2539.105070 | 1146.067493 | 1599.258424 | -206.220847 |

Detailed overlap, coverage, correlations, and per-sample physical features are in the JSON/CSV.
