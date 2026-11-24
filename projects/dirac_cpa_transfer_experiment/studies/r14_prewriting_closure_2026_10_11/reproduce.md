# R14 复现与恢复入口（工作进行中）

所有操作在本目录执行。历史 R1–R13 结果不参与新训练，不需再次运行历史脚本。Python版本和包版本见 [环境冻结](frozen/environment.json)，源码及 SHA 见 [源码清单](frozen/source_manifest.json)；历史证据保护见 历史哈希清单（本地档案）。

首次复现的顺序：冻结设计 `python plan.py`；保留历史内容哈希并建立本轮隔离的依赖快照；运行 `python -m closure.reference`、`python -m closure.tests`、`python -m closure.spectral --test`、`python smoke_pipeline.py`；全部门槛通过后 `python freeze.py`；运行 `python driver.py`。原档案已完成这些步骤，续办时不得重置已有正式目录。

这段首次建立顺序只适用于新的复现实验目录。`freeze.py`还会写环境与资源估算，历史审计程序会生成保护基线，不能在原交付档案中以“复现”为由重跑这些建立步骤。只读复核现有R14结果需要本目录的plan、closure/vendor、冻结源码/设计、references及原始结果；历史内容复核另需要完整原工程与docs/reviews，维持原workspace/projects/项目/studies/轮次层级。历史清单中的路径相对workspace，清单本身不包含约179GB的历史原文件，不能只复制清单便声称可重验历史内容。

驱动器先执行固定24个开发任务，冻结基线选择，然后依次T10、m10、m5T5及三扩展。运行 `python -m closure.spectral` 完成谱积分基准。新结果路径为 development、formal、spectral_benchmark，所有 job 记录在 runtime/jobs。训练从独立的新初始化开始，每组500步保存Adam、权重、全RNG和私有采样流。

工程恢复：先查看 progress.json 和日志，并核验实际进程；已有驱动器仍活跃时不得启动第二个。确认工程中断后，以原 driver.py 重启，自动读取 recovery_state.pt，从相同权重、优化器和下一个时间样本继续。不得删除 failure.json、换种子、重建Adam或按精度改变预算。丢失的尚未落盘片段按同一随机流重放并披露其费用限制。

正式源码冻结后不改模型、损失、成功阈值或科学选择；工程调度修复另存记录。未完成状态由文件和执行步数核验，不能仅凭模型文件存在判为完成。

训练结束仍需：所有固定终点和数值失败清点；指标独立重算；载波复用与零样本45项评价；费用核验及配对统计；查新对应表；图形输出和实际目视检查；全部链接及复现清单；基于最终结果的总报告、主张—证据表和投稿定位。此说明不会替代这些实际工作。

本目录已有正在运行的冻结实验。完整重新训练的复现应在独立副本中执行；先核实实际驱动器和工作进程，不能在此处再次启动相同任务。已有结果的只读重算不重新训练：依次执行`python -m analysis.audit_new_metrics`、`python -m analysis.completeness`、`python -m analysis.aggregate`、`python -m analysis.cost_audit`、`python -m analysis.cost_summary`、`python -m analysis.learning_advantage`。旧终点复核分别为`analysis.audit_early_physical_metrics`和`analysis.audit_historical_spectral_metrics`，其覆盖范围见审计记录。

主/终点推理和内存测量由`python -m analysis.inference_benchmark`增量执行；源检查点和测量源码的SHA不一致时停止，不能静默重测覆盖。预定255个端点并不等于255个独立训练。零样本程序`python -m analysis.zero_shot`在主训练后做45项相同完整模型的未修正评价；该程序不能用于修正初态或训练权重。

后台后处理已在等待预定训练终点，进度见runtime/postprocessing_progress.json（本地档案）。其中的25项操作队列包含独立精度、JVP残差/势卷积、成本、实际权重张量计数、载波初始控制、配对、相位、推理、推理成本来源/中位数、独立统计、完整组首次判定归档、谱基准重算、45项零样本及直接空间重算、13张正式图生成、向量渲染和历史内容重验。先核实该后台进程，避免重复后处理。队列结束仍需实际视觉检查和最终报告验收。

残差只读核验入口为`python -m analysis.residual_checks`及`python -m analysis.residual_provenance`；保存201个残差时刻的JVP独立导数、内部/外部势卷积与完整4001/8001点表的标量重算。原始文件SHA和该测量源码SHA均核对。指标的频率/相位诊断使用参考值只在独立评价中进行；范数方向残差投影近零不等于动力学正确。

历史528行规范表的来源及检查点身份见[逐行追溯](history/canonical_row_traceability.csv)。最终完整历史内容重验使用`analysis.verify_history_integrity`，单独保存带时间戳的校验结果；最初保护清单保持不变。全套正式图入口`analysis.plot_formal`只在完整预定尝试和独立复核齐全时出图，随后实际打开图检查，生成代码本身不表示图已生成或验收。

实际已检查的图及源数据见[视觉记录](visual_QA.md)，目前全套正式图未齐。最终链接检查需重新运行`python -m analysis.link_check`；当前仅覆盖已有文档。科学及成本边界见[独立复核补充](verification_notes_2026_10_11.md)。

[理论闭环核验](theory_closure_audit.md)逐式检查原相位门的初始虚部切向、复谱范数投影、近似载波自由缺陷、谱外残差和连续误差关系的成立条件。[推导来源记录](engineering/theory_closure_audit.json)绑定实际冻结源码及说明文件SHA。它不替代实验，不将离散残差或硬范数恒等式称为动力学证书。

统计表独立核验入口为`analysis.audit_statistics`，直接读原始指标/成本和已重算的FFT指标，不调用汇总实现。缺五种子的均值/区间必须空置；n=5配对区间、32种符号翻转和2×2交互重新计算。`analysis.audit_inference`独立重算七次观测中位数并查检查点/测量源码SHA；它不重测或覆盖观测。`analysis.audit_spectral_benchmark`用全谱双线性卷积的Parseval关系重算108个积分配置的密度/流误差，同时检验原五次成本中位数。积分器的原OS峰值RSS是累积基准进程的峰值，不能当独立配置的隔离内存峰值。

`analysis.parameter_audit`直接计数原始初态文件里的线性权重/偏置张量，不重建网络。实际保存的总/活跃/冻结参数与每份成本记录逐项核对；初态、Fourier B等固定buffer单列，不充当可训练参数，张量字节不充当整个进程内存峰值。

`analysis.pairing_audit`比较七个机制组实际初始权重字节，并以声明的920000+seed私有随机流独立重放20000批32个均匀时间，核对已保存的完整归一/物理时间前缀与检查点哈希。它只打开已终止作业的恢复文件，活跃作业的20000前缀保持待核；没有七组完整原始证据不能标整块通过。

载波初始控制由`analysis.carrier_initialization_audit`读取原/静态自由载波实际未训权重，用NumPy逐层前向比较t=0物理时间斜率，并检查联合组从对应静态未训载波出发；不能只凭相同seed声明匹配。

`analysis.archive_cohort_receipts`在当前汇总表和独立统计输入SHA一致时，归档已有完整五种子的主/终点首次判定、独立逐种子行及当时审计副本；失败判定同样保存，不覆盖先前收据。它不为缺失组生成完成证明，也不追加训练。

最终`claims_evidence.csv`的`claim_id`必须唯一，`evidence_paths`使用非空JSON路径数组；最终`code_data_manifest.json`的`files`逐项记录`path`、`sha256`、`bytes`，历史大清单等可另列`linked_inventories`并绑定SHA。`analysis.link_check`实际核对这些CSV/JSON来源及哈希，同时核查Markdown/文献外链。自引用文件与仍变动的运行状态须明确排除，不能虚构其稳定哈希；链接齐全仍不是科学验收。
