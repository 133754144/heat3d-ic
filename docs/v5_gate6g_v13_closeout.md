# Gate 6E V13 closeout

状态：`completed`。本 closeout 只读取 WSL2 已有 V13 工件和 `valid_iid=128`；未读取 test/hard/sealed，未训练或运行模型推理。

## 运行合同

- run commit: `21a1d9e`
- config SHA256: `38ec7070a27cd71739ba71cbbcc198e08ab688cc3092babe2f394588d84d6452`
- split: train=672, valid_iid=128, test_iid=128; nodes/sample=1024
- Global Context standardizer: `train_only`, samples=672

## V13 指标

| artifact | epoch | legacy base MSE | point-global relative RMSE | sample-first CV-relative RMSE | raw CV RMSE K |
|---|---:|---:|---:|---:|---:|
| base-MSE best | 318 | 0.03655965 | 23.700678% | 20.316459% | 0.16798237 |
| final | 600 | 0.03730812 | 23.942054% | 20.183978% | 0.17040920 |

Point-global trajectory best 为 epoch 318 / 23.700609%；sample-first trajectory best 为 epoch 395 / 19.950435%。两者均未保存对应参数，因此严格标记 `trajectory_only=true`，不称为 checkpoint。

## Paired 与 ensemble

JSON 中记录 N3/L2/V13 的逐模型 valid-only 聚合、true-CV-RMS Q1-Q4 point SSE，以及 N3/V13、L2/V13 固定 alpha ensemble；所有比较严格区分 point-global、sample-first 和 legacy base MSE。

大型 checkpoint、prediction 与 output 文件均未纳入 Git。
