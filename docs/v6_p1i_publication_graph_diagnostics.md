# V6 P1i vs P1h cached graph diagnostics

This is an offline cache audit. It performs no model inference and reads no test/sealed labels.

| family | N | Nr | P2R edges | R2R edges | P2R regional degree | radius median (m) | P2R length median (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1i_sample_varying | 1024 | 256.0 | 2594.8 | 4119.2 | 10.136 | 9.271004e-04 | 7.159474e-04 |
| P1i_sample_varying | 4096 | 1024.0 | 11878.0 | 17728.4 | 11.600 | 5.036276e-04 | 4.234282e-04 |
| P1i_sample_varying | 8192 | 2048.0 | 22700.1 | 35973.8 | 11.084 | 3.875722e-04 | 3.356460e-04 |
| P1i_sample_varying | 16384 | 4096.0 | 38712.2 | 73177.2 | 9.451 | 3.497322e-04 | 2.696609e-04 |
| P1i_sample_varying | 32768 | 8192.0 | 69092.1 | 146035.4 | 8.434 | 3.125000e-04 | 2.209710e-04 |
| P1i_sample_varying | 65536 | 16384.0 | 132434.5 | 291338.0 | 8.083 | 2.209711e-04 | 1.987844e-04 |
| P1h_shared_support | 1024 | 256.0 | 3326.0 | 3942.0 | 12.992 | 6.464108e-04 | 7.876240e-04 |
| P1h_shared_support | 4096 | 512.0 | 15836.0 | 7764.0 | 30.930 | 6.505857e-04 | 6.350445e-04 |
| P1h_shared_support | 8192 | 1024.0 | 28902.0 | 16024.0 | 28.225 | 4.997655e-04 | 4.687500e-04 |
| P1h_shared_support | 16384 | 2048.0 | 58830.0 | 32654.0 | 28.726 | 3.513930e-04 | 3.513924e-04 |
| P1h_shared_support | 32768 | 4096.0 | 112654.0 | 65408.0 | 27.503 | 3.125000e-04 | 2.842890e-04 |

Interpretation is recorded in the publication-pipeline closeout after correlation with the frozen accuracy curve.
