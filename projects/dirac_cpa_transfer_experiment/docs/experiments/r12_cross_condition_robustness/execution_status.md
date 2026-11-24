# R12 执行状态说明

科学配置已冻结，参考四算例均合格，29项工程测试及16个组合两步落盘smoke通过。历史9485文件已保护。

正式运行按 T10→m10→V04→two_harmonics 四批执行，每批五个单核seed工作进程，80次固定运行已由主调度启动。预计203核小时、五核约41小时，评价/I/O另计；这是基于R11实测的估计，不保证完成时刻。

当前状态以 results/raw/r12_cross_condition_robustness/runtime_plan.json、progress_snapshot.json 及各worker日志为准。本文件不记录尚未完成的科学结果，不能当作R12最终报告。

全部训练完成后，已启动的后处理进程依次汇总、审计、导出PNG/SVG/PDF及source CSV、独立渲染PDF，生成R12_ROBUSTNESS_REPORT.md，并停在实际视觉QA阶段。视觉检查通过后运行完整pytest、最终审计和链接检查。FIGURE_QA不得在尚未查看真实图像时记为通过。

如主机或进程中断，src.resume_r12_robustness检查是否仍有正式调度或worker存活，禁止重复启动；确实中断时先保留旧运行计划，再从Adam/RNG/时间采样恢复状态续训。预算和终点不变，不根据精度重跑。已完成的run由完整六件套验证缓存，不重复训练。

完整训练尚未完成前，不判断A/C的跨条件成败，也不将解析E的结果归为学习载波成果。
