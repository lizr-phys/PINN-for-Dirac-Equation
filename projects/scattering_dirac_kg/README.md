# 一维 Dirac / KG 定态散射 PINN

固定自然单位、m=1、S=0、V(x)=−0.5 exp(−x²/8)，单侧单位通量入射。E 为给定散射参数，区别于原束缚态实验。本目录独立，不改写已有四项目。

- [中文结果报告](report.md)
- [理论、自由旋量及边界](docs/theory.md)
- [方法、损失、先验和成本口径](docs/methods.md)
- [核实的原始论文与 DOI](docs/references.md)
- [冻结配置](config.json)、[执行协议](docs/plan.md)、[图表约定](docs/figure_contract.md)
- [运行清单](campaign_manifest.json)、[交付审计](results/audit.json)

主方法是随机特征 RBF 物理网络（PIELM 类）：复输出层由方程残差最小二乘训练，隐藏层固定；另完整保留常规全权重 Adam/L-BFGS MLP 对照。两者均只用方程与入射/出射条件，不读参考波形、R/T 或相位标签。

## 环境与复现

沿用已验证 Python 3.13.7、PyTorch 2.12.0 CPU、NumPy 2.3.5、SciPy 1.16.3、pandas 2.3.3、Matplotlib 3.10.8、mpmath 1.3.0。单线程，float64/complex128；任意精度参考用 60/70 位。

```powershell
Set-Location 'E:\PINN for Quantum Physics\projects'
python -m pytest scattering_dirac_kg/tests -q
python -m scattering_dirac_kg.run --stage all
```

阶段：`reference` → `fixed` → `baseline` → `scan` → `figures` → `report` → `audit`。参考与固定验证门须通过才允许正式条件网络。完成且源码/配置一致的训练会复用；改变训练定义时拒绝静默覆盖。

现有交付包含所有结果。只重新生成图表和报告：

```powershell
python -m scattering_dirac_kg.run --stage figures
python -m scattering_dirac_kg.run --stage report
```

在新运行组重训固定实验，不改写原始训练：

```powershell
python -c "from scattering_dirac_kg.campaign import fixed_campaign; fixed_campaign('reproduce_fixed')"
python -c "from scattering_dirac_kg.campaign import baseline_campaign; baseline_campaign('reproduce_mlp')"
python -c "from scattering_dirac_kg.campaign import conditional_campaign; conditional_campaign('reproduce_conditional')"
```

命令生成新组 CSV，不自动替代主报告的 `formal_v1`/`conditional_v1` 数据源。全新从头复现可复制 `frozen_source/formal_v1` 的源文件和配置到同级新包，保留旧目录，重新运行参考/训练；审计须为新项目建立原文件保留基线。若修改物理参数，重新验证参考和所有门控，不沿用旧参考。

## 原始数据

| 位置 | 内容 |
|---|---|
| `results/reference` | Jost/BVP、无势/解析方阱、扩域/容差/网格、101 点及 1001 点参考扫描 |
| `results/reference/high_precision` | Taylor 60/70 位振幅、阶数/步长/区间收敛与十进制原始值 |
| `results/formal_v1` | 两模型×四能量×三种子的主网络、配置、环境、训练点、模型、历史、奇异值、评价 |
| `results/mlp_baseline` | 同样 24 次常规 MLP，对失败不作筛选 |
| `results/conditional_v1` | 两模型各三条件网络、训练能量、每个扫描能量的复波形与指标 |
| `results/retraining_scan_v1` | 每模型 101 点独立重训，种子 0，实际测量成本 |
| `results/grid_refined_v1`、`results/domain_extended_v1` | 配点与区间敏感性 |
| `results/pilot_*`、`results/conditional_pilot` | 完整开发试验和失败，不混入正式统计 |
| `results/archived_reference_resolution_gate` | 高精度振幅参考加入前的失败门与原始指标 |
| `results/*summary*.csv` | 全部逐次指标、每能量三种子均值/样本 SD/范围 |
| `figures` | PNG/SVG/PDF、逐图数据源与图注清单 |
| `results/preservation_*.json` | 原四项目全部非缓存文件的 SHA256 保留核验 |
| `results/artifact_sha256.json` | 当前交付完整性哈希 |

NPZ 使用 `np.load`，不需要 pickle；主模型权重为独立复数组。MLP 的 `model.pt` 用 `torch.load(..., weights_only=True)`。Dirac 评价 state 的两列为真实复上/下分量，KG 两列为 (φ,φ')；current 为对应模型约定的流，density 为 Dirac 密度或 KG 电荷密度。

反射幅度小于 10⁻⁵ 时相位不解释，指标为 null/空 CSV。相位误差按模 2π 距离；所有波形不作事后相位拟合。参考数值一致性是经验收敛证据，不能当作严格误差上界。
