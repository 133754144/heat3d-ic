# V7 G2 P21 HCP 独立资格审计

HCP-enhanced DeepONet 固定在官方 commit `c7d856f9b665e29b64d9faf931f86db1d6a1f26c`，只运行官方 `single_htc_bc/train_chiplet-packagepower_hcp.py` 的原生五域配置。没有构造 P1i adapter，也没有读取任何 P1i test/sealed 数据。

## Native smoke

在 devbox RTX 5070 上用临时兼容环境完成一 epoch：官方五域 geometry、SMT LHS 路径、HCP projection、PDE/BC/interface loss、forward/backward、Adam step 和 checkpoint save/reload 均通过；75,042 个参数，耗时约 2.54 s，loss finite。Python 3.12 缺少已移除的 `imp` 模块，因此只提供 import shim；`gstools` 在该 native 路径未被调用，未替换 HCP projection 或模型代码。该环境是 reconstructed compatibility smoke，不应称为官方 pinned environment。

## 训练与 accuracy 分开

因此 native training 状态为 `TRAINING_READY`，但官方 checkpoint、raw outputs 和 benchmark data bundle 不在仓库，论文 accuracy 复现状态为 `OFFICIAL_ACCURACY_ASSETS_BLOCKED`。不能以 smoke loss 或自行生成的目标宣称论文精度。官方 HCP 输入是固定 cuboidal structured mesh、inline volumetric power 和 face BC；它不能原生接收 V7 variable geometry/material/source/dual-Robin point-cloud，因此 P1i 直接比较保持关闭。若后续获得官方 bundle，应在独立环境先做 native evaluation；否则只能走 HCP native benchmark + Heat3D same-domain adaptation 的参考路线。

详见 `configs/heat3d_v7/g2_p21_hcp_native_smoke_protocol.json` 与 `docs/v7_g2_p21_hcp_native_feasibility.json`。
