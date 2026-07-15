# Gate 6E optimization roadmap

本路线只按现有 train/valid 证据排序。test/hard 不参与排序；sealed IID 保持关闭。
本轮只准备第一项配置，不实现新架构，也不新增其他训练配置。

| priority | route | evidence and gate |
|---:|---|---|
| 1 | branch-only missing cell | L2 同时改变四个 loss 权重，无法分离 shape/scale branch rebalance 与 field/tail reweighting。先运行唯一缺失单元 `1.5/0.5/1/1`，是最小且可证伪的补全。|
| 2 | staged branch-loss schedule | 仅当 missing cell 显示早期稳定但后期回退时再考虑；用于区分 branch shaping 与长期 field calibration，不在本轮配置。|
| 3 | scale-loss gradient decoupling / scale-head independent LR | Gate 6A 显示 log-scale 与 relative-field global gradient cosine 为 `-0.581`，且 valid log-scale gradient 约为 shape gradient的 `6.48x`；这是直接的梯度冲突证据，但需要 runner/optimizer 语义变化。|
| 4 | EMA/SWA or prediction ensemble | 固定-alpha valid-only ensemble 可直接测量 checkpoint/模型误差互补性；只有观察到稳健改善后，才值得把 EMA/SWA 纳入训练合同。|
| 5 | graph augmentation and regularization | 当前没有证据表明 Gate 6 的主要误差由 graph 随机性或过拟合单独主导；需要独立 graph-seed/coverage 诊断，优先级低于已观察到的 loss 冲突。|
| 6 | coverage-targeted data expansion | 24D coverage distance 与 N3/L2 sample-relative error 有中等相关，但与 L2-N3 delta 相关较弱；可作为后续数据策略，不能用现有 valid target 反向选择样本。|
| 7 | capacity increase | 当前 width/steps 并未被证明是主瓶颈；增大容量会同时改变计算预算与正则化行为，故仅作为低优先级受控消融。|

推进约束：每次只开放一个科学变量；候选完全冻结前不得开启 sealed IID；test/hard
始终只允许冻结后的描述性报告。
