# V6 P1i P5-R resolution sweep

Status: **PASS**. Accuracy and latency in every neural row come from the same valid32 execution.

| Route | Mode | N | Nr | PG (%) | raw (K) | source (K) | peak (K) | interface (K) | continuous median (s) | p95 (s) | VRAM (GB) | FVM speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native1024_reconstruction | reconstruction | 1024 | 256 | 3.072016 | 2.930207 | 2.962902 | 2.675661 | 1.416680 | 1.672186 | 1.738907 | 0.155 | 1.01x |
| B4096_reconstruction | reconstruction | 4096 | 512 | 2.795165 | 2.612941 | 3.519986 | 3.688481 | 0.645706 | 3.265824 | 3.619745 | 0.159 | 0.52x |
| B8192_reconstruction | reconstruction | 8192 | 1024 | 2.739829 | 2.458792 | 3.587505 | 3.479191 | 0.510097 | 3.387036 | 3.873410 | 0.171 | 0.50x |
| E16384_reconstruction | reconstruction | 16384 | 256 | 2.702270 | 2.284806 | 3.872082 | 4.017769 | 0.385735 | 3.223886 | 3.624537 | 0.301 | 0.52x |
| E32768_reconstruction | reconstruction | 32768 | 256 | 2.672444 | 2.208698 | 3.903574 | 4.051111 | 0.384492 | 3.366696 | 3.714717 | 0.437 | 0.50x |
| E240825_direct | direct | 240825 | 256 | 3.067096 | 2.488376 | 4.734618 | 7.206823 | 0.474360 | 2.476497 | 2.586074 | 4.090 | 0.68x |

## Decision

Pareto routes: native1024_reconstruction, B4096_reconstruction, B8192_reconstruction, E16384_reconstruction, E32768_reconstruction, E240825_direct.
Frozen 0.1 percentage-point non-inferiority plus latency rule recommends **E16384_reconstruction**.
The FVM timing is historical reuse. Every neural accuracy and continuous latency above is newly measured together; no old latency was joined to new accuracy.
