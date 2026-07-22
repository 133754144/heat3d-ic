# V5 final phase assessment

Scope: frozen true-RMS evaluation on `valid_iid`; train was read only to verify frozen normalization/context. No test/hard/sealed access and no training.
Metrics use checkpoint-bound saved prediction NPZ artifacts. Training-time reload audits passed; documented direct CPU cross-backend replay drift is retained only as a diagnostic.

## Point-global-best ranking

| rank | model | epoch | point-global % | sample-first % | raw CV K | shape | scale log | <20% |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | V42 | 257 | 21.936815 | 19.250517 | 0.156347 | 0.143008 | 0.147133 | no |
| 2 | V38 | 231 | 21.944915 | 20.605917 | 0.157328 | 0.148421 | 0.178126 | no |
| 3 | V44 | 329 | 22.060219 | 18.907835 | 0.158943 | 0.140377 | 0.153252 | no |
| 4 | V43 | 276 | 22.450097 | 20.068498 | 0.161387 | 0.143975 | 0.158713 | no |
| 5 | V45 | 358 | 23.125696 | 19.993253 | 0.166421 | 0.144355 | 0.167092 | no |
| 6 | V46 | 484 | 23.615429 | 19.797011 | 0.169723 | 0.140332 | 0.169015 | no |

## Gate 6R paired attribution

Negative deltas mean improvement.

| comparison | point-global pp | sample-first pp | raw CV K | point SSE K2 | point win | Q4 SSE |
|---|---:|---:|---:|---:|---:|---:|
| V45_minus_V38 | 1.180782 | -0.612665 | 0.009093 | 288.095360 | 0.6016 | 347.613401 |
| V46_minus_V38 | 1.670514 | -0.808906 | 0.012395 | 412.012485 | 0.6094 | 478.544068 |
| V46_minus_V45 | 0.489733 | -0.196242 | 0.003302 | 123.917125 | 0.5781 | 130.930666 |

## Closeout verdict

- Scientific success: **no**. Best valid point-global result is 21.936815%, above the frozen <20% threshold; required valid+test success therefore cannot be established.
- Phase closure: **yes, as a completed negative/inconclusive research phase**. V45/V46 add no point-global improvement over the V38 lineage; no new V5 training is recommended.
- The sealed/test/hard roles remain unopened in this closeout. This is not a generalization claim.

## Main merge assessment

- Git topology is technically fast-forwardable: main `11e9d2feb1b9` is the merge base and V5 is ahead by 136 commits with no main-only commits.
- The branch changes 375 files (+481394/-453).
- Recommendation: **do not fast-forward the full research branch into main as-is**. Create a reviewed integration branch/PR containing reusable runner, metric, architecture, and tests; keep generated YAML/registries/large research evidence as an archival V5 ref or separate research-history merge.

No merge, tag, test/hard/sealed evaluation, or training was executed by this assessment.
