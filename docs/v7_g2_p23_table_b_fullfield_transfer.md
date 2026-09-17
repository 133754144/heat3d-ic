# Table B — P1i full-field transfer comparison

Table B 的域与指标已冻结，但本轮不填入数字。目标域为同一 `valid_iid=128`
与 `65×65×57=240,825` physical nodes，hotspot 定义为每个样本真实温度最高的
1% 节点。统一 evaluator 已实现于
[`scripts/evaluate_v7_g2_p23_common_fullfield.py`](../scripts/evaluate_v7_g2_p23_common_fullfield.py)。

当前 `BLOCKED_COMMON_EVALUATOR_NOT_EXECUTED`：V6 seed1/2 per-sample predictions
无法从当前工作树取得，Therm-FM 也没有逐样本 valid-only prediction archive；历史
aggregate 不能替代 paired rows。此外，混合角色旧 CSV 的意外读取已单独记录，故
不得把本表称作已闭合 publication comparison。

Therm-FM 的 Poseidon-T external PDE pretraining 必须单列，不能声称与 Heat3D
具有相同 information/training budget。
