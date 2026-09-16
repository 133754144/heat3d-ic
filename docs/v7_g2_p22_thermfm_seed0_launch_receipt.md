# P22 Therm-FM seed0 launch receipt

seed0 已按冻结 P22 native-65 contract 在 devbox 独立 `PDEFormer` 环境串行启动，状态为
`RUNNING_VALID_ONLY`。正式输出目录为空预检后创建，脚本拒绝覆盖已有目录；训练使用
train 768 / valid 128、batch 40、200 epochs、valid normalized p=2 lower-is-better 选模，
不构造 test loader。

首个启动命令因 full-field 路径拼写错误在任何数据/model 访问前失败，日志保留为
`g2_p22_thermfm_seed0.log`；retry1 使用修正后的只读路径，科学配置未变。retry1 的
tmux session 是 `g2_p22_thermfm_seed0_retry1`，日志和 output 路径见 JSON receipt。

已观察到 epoch 1：20 train batches、768 samples，epoch walltime 29.24 s，train loss
1.010656，valid normalized loss 0.989627，finite。后续 seed1/seed2 必须等待 seed0
完成并审计后再串行启动；test_iid、sealed、DeepOHeat official100 均保持锁定。
