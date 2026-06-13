# Heat3D v3 S2/S3 Long-Run Mechanism Audit

## Core Matrix

Best predictions:

| run | valid_iid | valid_stress | amplitude | corr | zRMSE | top-k | peak_rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B6 best | 0.0203209 | 0.0306349 | 0.992415 | 0.990175 | 0.100659 | 0.930469 | 0.041156 |
| S2 best | 0.0259946 | 0.0454090 | 0.979809 | 0.990325 | 0.0797057 | 0.943555 | 0.0445478 |
| S3 best | 0.0218745 | 0.0386520 | 0.983642 | 0.991366 | 0.0814905 | 0.942578 | 0.0404904 |
| S1 best | 0.0501334 | 0.0677455 | 0.930790 | 0.988526 | 0.0995608 | 0.946875 | 0.105233 |

Final predictions:

| run | valid_iid | valid_stress | amplitude | corr | zRMSE | top-k | peak_rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B6 final | 0.0206497 | 0.0308075 | 0.991861 | 0.990075 | 0.101761 | 0.927734 | 0.0413955 |
| S2 final | 0.0270849 | 0.0459213 | 0.990963 | 0.990478 | 0.0739975 | 0.946094 | 0.0394159 |
| S3 final | 0.0231263 | 0.0383088 | 0.998004 | 0.992146 | 0.0674796 | 0.951953 | 0.0349814 |
| S1 final | 0.0501334 | 0.0677455 | 0.930792 | 0.988526 | 0.0995612 | 0.946875 | 0.105235 |

## Conclusions

- S3 is clearly stronger than S2 on scalar loss and most mechanism metrics, and
  is close to B6.
- B6 remains strongest on scalar valid_iid and valid_stress.
- S3 final has the strongest mechanism metrics: near-unit amplitude, highest
  centered correlation, lowest z-score RMSE, highest top-k overlap, and lowest
  peak relative error.
- S3 best-valid occurs at epoch 596, while final epoch 1200 has better raw
  mechanism metrics. This means scalar valid loss and prediction-level
  mechanism metrics are not perfectly aligned.
- The next step should audit scalar-loss versus mechanism-metric mismatch before
  selecting the next long-run criterion.

## Caveat

These are diagnostic results, not formal benchmarks or publication-ready
claims. All outputs are from existing predictions; no model, decoder, loss, or
objective changes are implied.
