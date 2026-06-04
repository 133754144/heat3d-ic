# Heat3D v2 startup group-build progress update

## 背景

M1 mini-batch e50 的 startup group build 会构建 train、valid、all 三组 batch graph。旧行为在 `progress_detail=basic` 下对每个 group 打印两行：

- `arrays+graph start`
- `arrays+graph built`

在 `train=192`、`valid=32`、`all=256` 时，这会产生接近千行 startup 日志，SSH `train.log` 很难阅读。

## 新行为

本次只修改进度显示，不改变训练、graph construction、profile timing、metrics 或 output 文件格式。

- `progress_detail=basic`: 使用 stdlib 轻量进度条/限频输出。TTY 下单行刷新；非 TTY 或 tee 日志中按时间或约 5% 进度输出一行，并在结束时打印 completed summary。
- `progress_detail=verbose` 或 `progress_detail=full`: 保留旧的逐 group start/built 详细输出。
- `progress_detail=off` 或 `progress_detail=quiet`: 不输出 group-build 细节。

示例：

```text
[startup] group build train [############------------] 96/192 elapsed=18.4s avg=0.19s eta=18.2s
[startup] group build train completed groups=192 elapsed=36.9s avg=0.19s
```

## 影响

SSH 手动训练仍可以看见 startup 是否卡住，但默认 `basic` 日志不再逐 group 刷屏。需要排查单个 group 构建时，使用 `--progress-detail full`。
