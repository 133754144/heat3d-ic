# V7 G2-P14.1：Heat3D e200 U-v2 dense evaluation

本协议在 valid-only 推理开始前冻结。对象是已经完成的
Heat3D-on-DeepOHeat-v1 e200 三个 checkpoint；不重新训练、不覆盖 checkpoint，且只读取
冻结的 768 train / 128 valid_iid label cache。P1i test_iid、sealed 与 DeepOHeat
official100 均保持关闭。

## 三个表示

* native：1024 个 physics-layout-aware support 点，使用现有 native evaluator；
* IDW dense：复用 `layer_interface_knn_inverse_distance_v1`，输出
  `101×101×56=571256` 点；
* U-v2 dense：复用 V6 `UHighNRuntime` 的 asymmetric direct-query graph，输入仍为
  1024 support，query 为完整 DeepOHeat 网格，新增参数数为 0。query graph、model
  forward 和必要的 geometry preparation 均计入 U-v2 端到端推理时间。

三者在相同 valid128 physical cases 上，均使用 `EvaluationCore` 的 deltaT_K 温度空间
指标。primary 为 sample-first relative RMSE；同时报告 point-global relative RMSE、
RMSE[K]、MAE[K]、peak error 及已有 source/background/interface 分区统计。

## Oracle 边界

GT support → 冻结 IDW map 是合法的 label-independent reconstruction floor。V6 U-v2
实现是利用 native latent、physics context 与 query graph 的神经 direct-query path，
不是仅接收 1024 个温度值的插值器。因此不存在不改变算法就能定义的“GT support → U-v2
value-only” oracle；receipt 将其记录为 `NOT_DEFINED_FOR_DIRECT_QUERY`，不伪造数值、
不做 model/reconstruction 误差相减分解。

## 不可越界项

评估脚本拒绝 test/sealed 文件名，验证 source/subset/label/normalization SHA，逐样本
确认 split=`valid_iid`，不写入 Git 数据目录。输出 receipt 只含指标、hash 与运行元数据，
临时预测和大数组留在 `/tmp/g2_p14_1`。
