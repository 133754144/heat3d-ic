# V6_01 V4-best WSL2 e600 postmortem

## Outcome

The 600 training epochs completed and all requested final/best checkpoints and
prediction archives were saved. The process returned code 1 only during the
post-export checkpoint prediction replay. This was a false failure of the
old `max_abs <= 0.1 K` replay threshold, not a training, checkpoint, or NPZ
serialization failure.

The parameter tree reloaded exactly and the NPZ archive reloaded exactly.
Six inference-only repeats on the original WSL2 GPU showed that the same final
checkpoint can differ from itself by as much as 0.298 K at an isolated point,
while whole-field replay RMSE remains 0.0020--0.0027 K. This is the known
pointwise nondeterminism of irregular-graph GPU scatter/reduction ordering.

The repaired gate keeps exact parameter and NPZ requirements, keeps the strict
`RMSE <= 0.01 K` requirement, and adds `p99.99 <= 0.15 K` together with a
bounded `max <= 0.5 K`. Broad or heavy-tail drift still fails.

## Valid-IID result

| checkpoint | epoch | base MSE | point-global true-RMS | sample-first | raw RMSE K | amplitude | correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy best | 407 | 0.243180 | 13.8426% | 10.2327% | 5.82177 | 1.00919 | 0.87969 |
| final | 600 | 0.362727 | 16.9060% | 11.0228% | 7.11014 | 1.03145 | 0.87725 |

Both checkpoints are below the requested 20% point-global valid threshold, but
the e407 checkpoint is the correct result under the preregistered
`valid_base_mse` selector. No test or all-role groups were evaluated.

## Training diagnosis

The final train point-global error is 1.469% and train base MSE is 0.002739,
while final valid point-global error is 16.906% and valid base MSE is 0.362727.
The final valid base MSE is 1.492 times the e407 best: this is strong late
overfitting, not incomplete optimization.

Final e600 improves sample-relative error on 71.1% of valid samples and its
median sample improvement is 1.414 percentage points, yet the mean
sample-first metric worsens by 0.790 points. A small tail of geometry groups
therefore dominates the aggregate regression. The largest regressions include
groups `p1g_g104`, `p1g_g111`, `p1g_g097`, and `p1g_g102`.

By true DeltaT RMS quartile, point-global error changes from best to final as:

- Q1: 16.70% to 21.90%
- Q2: 15.58% to 19.39%
- Q3: 15.09% to 17.94%
- Q4: 11.35% to 13.45%

The degradation is therefore broad in point-SSE terms, strongest
proportionally in lower-temperature Q1/Q2, while a minority of hard layout
groups creates the sample-first tail. The e407 checkpoint should be retained;
the final checkpoint is diagnostic only.

## Scope

This is a valid-IID postmortem. It does not establish test performance, OOD
performance, or model selection beyond the frozen valid selector. No training
was restarted and no original run artifact was overwritten.
