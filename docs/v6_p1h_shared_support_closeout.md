# V6-P1h shared solver-node support closeout

Status: `passed`. The dataset contains 1024 cases in 128 frozen P1g groups. All samples share one ordered 1024-node support and one graph (`6d3d62830755872194766aad2a8ac7b0f1fabec57840dac78fcb2642a6ed771c`).

The original-field search inspected 63,598 NumPy files across all configured local
worktree/tmp/project/Codex roots and found no original 240,825-node solver array or
usable archive. The deterministic replay then verified eight factor-cover cases and
all 10,240 P1g manifest file hashes with zero coords/k/q/projected-T/solver-metric
difference. Full generation took 1603.077 s, including 1527.901 s in the solver.

Source support points: min=4, p05=5.0, median=10.0; zero-covered sources=0. Full-field CV-RMSE median=1.075332 K and p95=2.209430 K.

The support was selected from stack/layer/interface/Robin/source-allowed geometry only. No temperature or test label entered proposal selection. P1g and the canonical dataset designation remain unchanged.

The reusable dataset and full-field archive are stored at
`/Users/xuyihua/.codex/worktrees/5c97/3D IC Heat/data/heat3d_v6_p1h_shared_support1024_v0`.
The archive SHA256 is
`f58141b3f365c5c90a57ec3802ae57c7e7afbf83ba0ab988060a617164b14c00`;
the dataset manifest SHA256 is
`324ca50a85698223d36c12a05d3e26b5cbc9aa00b559d067619baeb37f11e9d5`.
