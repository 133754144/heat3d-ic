# P1i graph policy E screen closeout

E 在 8192 的 accuracy 与 coverage 均通过冻结门；fresh/warm 分别改善 17.04%/34.32%。但峰值 VRAM 比为 1.2539，超过预注册 1.10，因此 **E NO-GO**，未运行 16384。

确认协议冻结为 A/B，使用剩余 valid96、三 seed、8192/16384；不重算冻结 valid32。
