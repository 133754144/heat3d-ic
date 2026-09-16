# V7 G2 P20：devbox recheck

时间：2026-09-16 18:26:52（`XYH-Desktop`）。本次只读核对，没有拉取远端分支、启动训练、读取任何 test/sealed 或 DeepOHeat official100。

## 远端与标签资产

- devbox G2 checkout：`research/v7-g2-baselines`，commit `2ed1ac937757c2763a1aea1a886153aa87293393`，工作树干净；该 checkout 落后于本地已推送审计分支，未做 pull 或修改。
- `/home/xyh/myCodeGitOnly/heat3d-ic-g2/data/g2/deepoheat_v1_volumetric_labels` 已存在，目录大小 `2,094,469,815` bytes，包含 768 train case directories 和 128 valid case directories。
- label receipt SHA256：`a4bb99638a977b2004a93a88b469166ff7da697e89181e64e04152c7f96fe4fd`；normalization payload SHA256：`3a0273bb92b8c060df8a214b1e0e7dd0e4b5df6bece86b7dea15197ca56ed0db`；subset/support manifest SHA256 分别为 `e7196651…61558` 和 `bb1261f5…a8017`。
- receipt 状态为 `PASS_COMPLETE_768_128`，solver 为 CPU hybrid solver，mesh `101×101×56`，`rtol=1e-10`；此次没有新生成 labels，也没有创建本地 2 GB 副本。

## 运行与上游资产

- 已有 Heat3D-on-v1 e600 三 seed 均为 `COMPLETE`（seed0 best epoch 574、seed1 593、seed2 581）；没有启动新训练或覆盖输出。
- devbox 当前没有活动 G2 tmux/训练进程；非交互 shell 没有 `nvidia-smi` 命令，但这不改变既有 GPU receipt。
- 未发现 Therm-FM selective model_T/config/stats 或 HCP native bundle；HF cache 仅见 `thuerey-group/pde-transformer`。Therm-FM/HCP 仍为 `BLOCKED_BY_UPSTREAM_ASSETS`，不下载 24 GB archive。

完整机器可读 receipt：[`v7_g2_p20_devbox_recheck.json`](v7_g2_p20_devbox_recheck.json)。
