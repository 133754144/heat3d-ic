# V7 G2 P20-C：HCP compatibility audit

状态拆分：native training `TRAINING_READY`；official accuracy `OFFICIAL_ACCURACY_ASSETS_BLOCKED / REFERENCE_ONLY`。P21 完成一 epoch native smoke；本轮不训练正式模型、不下载 checkpoint/raw outputs、不访问 test/sealed。

官方来源为 [HCP-enhanced DeepONet repository](https://github.com/LIMIGwenshao/HCP-enhanced-DeepONet)，冻结 commit `c7d856f…`，MIT license。README 将方法定义为 hard-constraint projection enhanced neural operator with hybrid structured-mesh/LHS sampling，并给出 Microelectronics Reliability 2026 论文和 DOI。仓库树包含训练和 eval 脚本，但没有随 Git 提交 benchmark 数据或 checkpoint；README 明确约 326 MB 的 `.npz/.npy/.pth` 与图表需另行索取。

## Native contract

官方 `train_chiplet-packagepower_hcp.py` 把几何、网格分辨率、volumetric power map、face BC/interface 和 conductivity 直接写在每个 domain config 中；HCP projection 要求完整、均匀的 structured 3-D mesh。官方 single-HTC 示例使用 `[0,1]×[0,1]×[0,0.55]`、root mesh `[20,20,11]`、内层 volumetric power 和底部 HTC 参数化。`dataio`/`eval.py` 通过脚本配置和 `.pth` checkpoint 工作，而不是 V7 的 variable-geometry point cloud loader。

因此，P1i 的 irregular coordinates、heterogeneous `k(x)`、q/source 布局和双 Robin BC 没有被证明可以被原实现原生接收。大幅重写 HCP branch/input/interface 后再称“original HCP baseline”不被接受。本 track 已拆分为 `TRAINING_READY` 与 `OFFICIAL_ACCURACY_ASSETS_BLOCKED / REFERENCE_ONLY`：后续优先复现 HCP native single-HTC benchmark；只有在取得官方资产并验证 native evaluation 后，才评估同域 Heat3D adaptation。
