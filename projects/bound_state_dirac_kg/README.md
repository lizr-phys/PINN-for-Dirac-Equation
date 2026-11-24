# 一维 Dirac / KG 定态束缚态 PINN

独立于已有时间演化实验的基础验证。固定时间分量矢量高斯势，先验证传统参考解，再训练每个方程的最低正能态。

- [中文结果报告](REPORT.md)：数值结果、三种子误差、衰减、收敛、局限与图表。
- [理论核对](THEORY.md)：定态分离、实分量符号、耦合区别、KG 电荷与正交关系。
- [数值方法](METHODS.md)：参考算法、损失、硬/软约束、先验披露、统计口径。
- [核实来源](REFERENCES.md)：原始文献及 DOI。

- [束缚态改进轮](upgrade/README.md)：硬约束、多初值/参数、激发态与带权正交；[改进结果报告](upgrade/REPORT.md)。原基础数据保留。

## 环境

沿用现有 Python 3.13.7、PyTorch 2.12.0 CPU、NumPy 2.3.5、SciPy 1.16.3、pandas 2.3.3、Matplotlib 3.10.8。
正式运行使用 float64、单线程 CPU；可并行运行互不相同的种子，但墙钟时间会受资源竞争影响。
`requirements.txt` 固定核心版本。若使用其他硬件或库版本，末尾数字可能变化。

```powershell
Set-Location 'E:\PINN for Quantum Physics'
python -m pip install -r bound_state_dirac_kg/requirements.txt
python -m pytest bound_state_dirac_kg/tests -q
python -m bound_state_dirac_kg.run --stage all
```

`all` 依次运行参考解、六次正式训练、独立评价、四次敏感性训练、图表、报告和审计。
已完成且配置相同的训练会跳过；配置不同时拒绝静默覆盖，应使用新 tag。
本目录已包含本轮全部完成结果，直接执行评价、图表和报告阶段无需重训。

```powershell
python -m bound_state_dirac_kg.run --stage reference
python -m bound_state_dirac_kg.train --kind dirac --seed 0
python -m bound_state_dirac_kg.train --kind kg --seed 0
python -m bound_state_dirac_kg.run --stage train
python -m bound_state_dirac_kg.run --stage evaluate
python -m bound_state_dirac_kg.run --stage sensitivity
python -m bound_state_dirac_kg.run --stage figures
python -m bound_state_dirac_kg.run --stage report
python -m bound_state_dirac_kg.run --stage audit
```

完整重训可在新 tag 下进行，示例：

```powershell
python -m bound_state_dirac_kg.train --kind dirac --seed 0 --tag reproduce
python -m bound_state_dirac_kg.evaluate --tag reproduce
```

冻结配置为 m=1、V0=0.5、a=2、L=20，能区 [0.5,0.8] 来自独立参考谱。
当前参考驱动程序明确只支持该物理参数组合，修改势阱应另作参考验证；不要仅修改训练参数而沿用旧参考。
训练函数不导入参考求解器、不读取参考波形，参考结果只在评价阶段加载。

## 数据目录

| 位置 | 内容 |
| --- | --- |
| `config.json` | 冻结物理/训练配置和能区先验声明 |
| `results/reference` | BVP 波形和插值多项式、射击核验、谱、网格/区间收敛及差分倍增反例 |
| `results/formal/{dirac,kg}/seed_{0,1,2}` | 配置/环境、检查点、训练曲线、8001 点预测和残差、各项指标 |
| `results/pilot_initial` | 首轮未缩放 L-BFGS 的种子 0 原始试验，不混入正式统计 |
| `results/grid_refined` | 种子 0：残差配点 768、归一化求积 512 |
| `results/domain_extended` | 种子 0：区间扩大至 [−24,24] |
| `results/*_summary.csv` | 对应运行组的逐次原始指标 |
| `results/formal_aggregate.csv` | 三种子均值、样本标准差、最小和最大值 |
| `results/figures` | 六组 PNG/SVG/PDF 及数据来源清单 |
| `results/audit.json` | 完整性、数值阈值、参考误差分离及测试记录 |
| `results/artifact_sha256.json` | 交付文件完整性哈希（不含缓存和清单自身） |

NPZ 可用 `numpy.load` 读取；CSV 为可直接检查的原始浮点值；`model.pt` 是 `state_dict`，用 `weights_only=True` 加载。
首轮试验可通过 `train(kind,0,'new_pilot',{'lbfgs_objective_scale':1.0})` 重现。
所有临时输出、训练数据和新代码限制在本目录，不向旧实验写入文件。
