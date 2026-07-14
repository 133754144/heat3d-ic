# V5 Gate 5 error attribution

## N1 to N3

归因：`joint_path`。下表为 best clean valid/test 均值。

| model | joint % | shape CV-RMSE | scale log-RMSE | amplitude | oracle-scale % | oracle-shape % |
|---|---:|---:|---:|---:|---:|---:|
| N1 | 25.0174 | 0.1800 | 0.2331 | 0.9851 | 17.9994 | 15.3070 |
| N3 | 21.1236 | 0.1514 | 0.1778 | 0.9952 | 15.1362 | 13.0778 |

N3 FiLM 幅值：valid gamma/beta mean-abs=0.2624/0.2114；test=0.2650/0.2158。

## High DeltaT tail

| model | top-5 error share | top-10 error share | point minus sample-first pp | high-scale quartile error share |
|---|---:|---:|---:|---:|
| B0 | 0.2549 | 0.4047 | 0.3966 | 0.8338 |
| N0 | 0.2714 | 0.4387 | 3.7943 | n/a |
| N1 | 0.2541 | 0.4107 | 4.3545 | 0.8539 |
| N3 | 0.3161 | 0.4643 | 3.1698 | 0.8501 |

分箱、top-10 样本及 power/source/conductivity/top-h/anisotropy 相关系数见 diagnostic JSON。

point-global 与 sample-first：point-global 偏高主要来自高温升样本的误差集中；这些样本在 true-energy 加权的 point-global 中权重大于不加权的 sample-first 均值。

| model | strongest squared-error relation | Spearman rho |
|---|---|---:|
| B0 | total_power_W | 0.4986 |
| N0 | total_power_W | 0.4717 |
| N1 | q_weighted_inverse_kz_mK_W | 0.5000 |
| N3 | total_power_W | 0.4839 |
