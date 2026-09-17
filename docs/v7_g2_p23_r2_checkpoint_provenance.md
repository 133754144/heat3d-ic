# P23-R2 checkpoint provenance / recovery receipt

## Verdict

`FORENSIC_RECOVERY_EXHAUSTED_CHECKPOINT_NOT_RECOVERED`

The two V6 canonical files required for P23-R replay were not found on devbox:

- `V6_07_V5best_P1i_seed1_reliable_B24` →
  `7197157969278d99648ef9b40d74005d759f32e52e9282e72a5586003d1e71f7`
- `V6_08_V5best_P1i_seed2_reliable_B24` →
  `d67e0dac2e8ed8009ce7dcdf0b2de4543b10bc005c0bfaa51027ea721bb2ab49`

The only V6 point-global checkpoint found and hashed was the existing seed0
reference (`51567afe17e38cb6ed8c95c4dd39598e647c1699de9351358e7729fecc20b90e`).
It is retained as reference evidence only.

All inspected historical U-v2 receipts (S1/S2/S3, direct 240825 and 16384
groups) carry the same seed0 checkpoint SHA. They therefore do not establish
seed1/seed2 identity and cannot provide canonical P23 predictions.

No recovery copy was created. No P23-R evaluator, checkpoint, protocol, or
existing result was changed. Seed1/2 replay, six-archive common evaluation,
paired bootstrap, condition analysis, and Table B remain blocked until the
two exact files are recovered and independently SHA-verified.

Machine receipt: `docs/v7_g2_p23_r2_checkpoint_provenance.json`.
