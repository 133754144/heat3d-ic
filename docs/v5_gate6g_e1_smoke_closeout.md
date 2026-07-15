# Gate 6G e1 smoke closeout

状态：`completed`。六组配置均在 devbox 独立 worktree 上使用正式 P5 数据完成 e1 execution smoke；输入为 train=672、valid_iid=128、1024 nodes/sample、B28。没有启动 e200/e600。

| config | isolated path | params | delta vs control | peak RSS GiB | live device GiB | reserved device GiB | allocator pool GiB | checkpoint reload |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `V4P5_22_gate6g_control_constlr` | control | 853,927 | +0 | 4.197 | 6.600 | 0.000 | 7.998 | 5/5 |
| `V4P5_23_gate6g_stopgrad_constlr` | stop-gradient | 853,927 | +0 | 4.166 | 6.413 | 0.000 | 7.998 | 5/5 |
| `V4P5_24_gate6g_shape_attention_constlr` | shape attention | 884,520 | +30,593 | 4.440 | 6.495 | 0.000 | 7.998 | 5/5 |
| `V4P5_25_gate6g_scale_attention_constlr` | scale attention + stop-gradient | 893,736 | +39,809 | 4.457 | 6.709 | 0.000 | 7.998 | 5/5 |
| `V4P5_26_gate6g_shape_attention_stopgrad_constlr` | shape attention + stop-gradient | 884,520 | +30,593 | 4.427 | 6.473 | 0.000 | 7.998 | 5/5 |
| `V4P5_27_gate6g_deep_scale_head_constlr` | deep scale head | 862,247 | +8,320 | 4.196 | 6.469 | 0.000 | 7.998 | 5/5 |

所有配置均满足：

- `status=passed`、`grad_finite=true`；
- 五类 checkpoint/prediction reload audit 全部通过；
- Global Context standardizer 仅由 train=672 拟合；
- 仅访问 `train` 与 `valid_iid`；
- 未访问 test、hard roles 或 sealed IID；
- `long_training_started=false`。

显存字段按采集器原样分开报告：live device、reserved device 与 allocator pool；本机后端报告的 reserved device 为 0，pool 为 8.000 GiB。e1 指标仅用于执行验证，不作为性能结论。可执行的 e200 手动命令和顺序见 `docs/v5_gate6g_attention_preflight.md`。
