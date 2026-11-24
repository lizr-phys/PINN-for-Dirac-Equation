# 束缚态改进轮

本目录沿用静态高斯时间分量矢量势（S=0），保留原基础实验。改进包括积分归一化与 Robin 硬结构、Dirac 精确下分量重建、物理正能隙搜索、多种子/参数/激发态验证、合法 KG 带权正交和数值恢复。

- [中文结果报告](REPORT.md)
- [方法、推导与先验披露](METHODS.md)
- [当前配置](config.json)、[版本 6 快照](frozen_sources_v6.json)、[版本 7 快照](frozen_sources_v7.json)、[逐运行协议清单](campaign_manifest.json)
- 原始运行汇总（本地档案）、[图表与图注](figures/manifest.json)、[最终审计](results/audit.json)

## 复现

沿用原项目固定 Python 依赖；float64、单线程 CPU 串行计时。

```powershell
Set-Location 'E:\PINN for Quantum Physics'
python -m pip install -r bound_state_dirac_kg/requirements.txt
python -m bound_state_dirac_kg.upgrade.run --stage all
```

阶段顺序：reference→train→evaluate→diagnostics→figures→report→audit。共 50 次正式、9 次对照和 4 次敏感性训练。相同配置的完成运行跳过；更改科学源码或配置必须另设版本和运行组，禁止静默混合协议。最终审计运行原 11 项测试和本轮新增测试。

只重新生成统计/图表时：

```powershell
python -m bound_state_dirac_kg.upgrade.run --stage evaluate
python -m bound_state_dirac_kg.upgrade.run --stage diagnostics
python -m bound_state_dirac_kg.upgrade.run --stage figures
python -m bound_state_dirac_kg.upgrade.run --stage report
python -m bound_state_dirac_kg.upgrade.run --stage audit
```

每运行的 config.json 保存实际环境、初始化与源文件 SHA256；model.pt、训练求积点、history.csv、检查点、独立测试网格和 evaluation.npz 可重新检查。原始三种子基础实验通过 original_snapshot.json 保留审计；pilot_v1–v7 与版本 5 的部分正式组用于开发/失败披露，不进入最终统计。保留 38 次独立验证通过的版本 6 最低/第一激发态，剩余 12 次正式和 13 次辅助运行使用版本 7；明列协议，不静默混合。版本 6 的第二激发态尺度溢出失败保留，版本 7 作等价稳定缩放和真实范数检查。阈值恢复和一次中点重启不使用参考标签，报告重启与累计耗时。

修改配置后需新的 formal_group、version，并为对照/敏感性组使用新目录；旧目录有配置冲突时主动报错。参考解与训练可分开运行。四个离散势阱分别重训，不属于参数化神经算子或全谱盲发现。
