# P22 Therm-FM reconstructed environment

Therm-FM 的真实 target qualification 与 seed0 formal 使用 devbox 独立 `PDEFormer` 环境，
不是仓库声称的官方锁定依赖环境。实测为 Python 3.13.9、PyTorch 2.9.0+cu128、CUDA 12.8、
Transformers 4.57.1、Accelerate 1.11.0、h5py 3.15.1、NumPy 2.3.4；GPU 是 RTX 5070
（sm_120，12,820,480,000 bytes）。CUDA availability 与设备信息由 `torch.cuda` 核验；主机
PATH 没有 `nvidia-smi`，不影响模型运行。

Therm-FM commit、model/trainer/config SHA 与完整机器可读字段见
`docs/v7_g2_p22_thermfm_reconstructed_environment.json`。该环境标签必须在正式结果中保留，
不得写成 official dependency environment。
