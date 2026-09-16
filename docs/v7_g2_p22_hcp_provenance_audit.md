# P22 HCP provenance-only audit

状态：`REFERENCE_ONLY / OFFICIAL_ACCURACY_ASSETS_BLOCKED`，不阻塞 Therm-FM。

官方 HCP commit 为 `c7d856f9b665e29b64d9faf931f86db1d6a1f26c`。training script
`train_chiplet-packagepower_hcp.py` 与 evaluation script `eval.py` 都把 geometry、mesh、
power、BC 和 parameter sweep 内联在脚本中；二者不是可由 V7 P1i split 直接复用的
split-preserving loader。仓库不含训练 checkpoint、raw FVM outputs 或完整 published accuracy
bundle，因此不能宣称论文 accuracy reproduction。已有 native single-HTC one-epoch HCP
projection smoke 仅证明官方 source 可运行。

该原算法要求固定 cuboidal structured mesh/LHS 与 inline source/BC/material，不能原生接收
P1i variable geometry/material/source/dual-Robin point cloud。本轮不改 HCP branch/input/interface，
不生成 reconstructed benchmark，保持 reference-only；任何后续 accuracy 工作需官方 bundle
或另行批准的 HCP-domain reconstruction。未访问 P1i `test_iid`、sealed 或 DeepOHeat official100，
未启动 HCP formal training。
