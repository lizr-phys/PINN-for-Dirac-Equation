# R11 复现与审计入口

所有调用在实验项目根目录进行。Python为本机现有运行时；BLAS和Torch按单线程执行。科学冻结后不得改写预注册或科学模块。

1. 正式前：运行 src.run_r11_ablation 的 prepare，完成参考复核和历史SHA保护；运行 tests/test_r11_ablation.py 全部测试，记录 pre_freeze_test_result.json；运行 smoke，只作工程验证；运行 freeze。
2. 正式训练：运行 src.run_r11_ablation 的 formal。调度五个新种子，按固定拉丁轮转串行五组；启动前核查五worker及控制核互斥。中断只能按恢复状态继续原预算，不能重置或换种子。cached终点不得再次训练。
3. 全部25run完成后：运行 src.aggregate_r11_ablation，随后 src.audit_r11_ablation（数据审计）。
4. 作图：运行 src.plot_r11_ablation，随后 src.qa_r11_ablation_figures 生成独立PDF预览；实际查看所有PNG和PDF预览后记录可视QA。若需修正图面，只改非科学绘图代码并保留已有导出记录，不能改数据。
5. 报告：运行 src.build_r11_ablation_report。运行整个项目pytest，保存日志、JUnit XML和pytest_final_result.json。
6. 最终：运行 src.audit_r11_ablation --final，核查全部数据、历史/冻结SHA、预算、配对、完整产物、图面及报告链接。全部产物写入 artifact_sha256.json。

图面实际检查通过、完整报告落盘后，可运行 `python -m src.verify_r11_deliverables` 一次执行完整pytest、保存日志/XML/实际结果、补入报告并完成最终审计。该入口拒绝覆盖已有完整测试证据；如有失败，先保留证据并核查原因。

正式优化预算与固定评价由冻结的protocol.json决定。源代码和数据索引分别在科学冻结、历史保护及全产物SHA清单中。中间点与终点全部保留；A/E的final和20000耦合快照是相同实际检查点，不计为额外重复。

训练恢复包含模型、Adam、独立自由/耦合随机生成器完整状态和采样迹、Python/NumPy/Torch RNG、阶段、步数、历史及累计阶段成本。评价发生在固定检查点落盘后；不会改变训练随机流。E是显式解析先验例外，主学习方法比较仅包括A–D。
