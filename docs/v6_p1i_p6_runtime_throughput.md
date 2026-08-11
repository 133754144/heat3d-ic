# V6 P1i P6 runtime and throughput closeout

Accuracy is reused from P5-R because graph/model/support semantics are unchanged. All latency values below are newly measured on WSL2 (RTX 5070 / Ryzen 7 9700X).

| route | fresh B1 med/p95 (s) | resident replay (s) | B16 samples/s | B16 avg (ms) | B1 speedup vs serial FVM |
|---|---:|---:|---:|---:|---:|
| native1024 | 0.298376 / 0.319771 | 0.001526 | 2248.28 | 0.445 | 5.27x |
| E16384 | 0.778409 / 0.819143 | 0.005920 | 185.50 | 5.391 | 2.02x |
| E32768 | 0.861476 / 0.947140 | 0.009293 | 117.00 | 8.547 | 1.83x |

Serial FVM B1 median/p95: 1.573042/1.833450 s; 32-case serial wall 51.374773 s (0.623 samples/s), one process and one thread.

## P5-R timer explanation

- native1024: graph 1.198→0.021 s; group+H2D 0.352→0.166 s. The old values were coarse GPU-default combined spans; P6 uses synchronized CPU-host graph/pack and explicit H2D boundaries.
- E16384: graph 2.202→0.074 s; group+H2D 0.615→0.371 s. The old values were coarse GPU-default combined spans; P6 uses synchronized CPU-host graph/pack and explicit H2D boundaries.
- E32768: graph 2.291→0.111 s; group+H2D 0.609→0.384 s. The old values were coarse GPU-default combined spans; P6 uses synchronized CPU-host graph/pack and explicit H2D boundaries.

Single-case latency, resident inference throughput, and streamed prepared-host throughput are separate workload semantics and must not be interchanged.
No training, test/sealed access, checkpoint change, dataset change, or graph semantic change occurred.
