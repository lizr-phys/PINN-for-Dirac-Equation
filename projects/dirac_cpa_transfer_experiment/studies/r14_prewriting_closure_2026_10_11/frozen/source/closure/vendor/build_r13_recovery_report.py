from __future__ import annotations

import json
from datetime import datetime,timezone,timedelta
import pandas as pd
from .r13_config import ROOT,RAW,DOCS,FIGURES,THRESHOLDS

def load(name):return json.loads((RAW/name).read_text(encoding='utf-8'))

def table(frame,columns):
    return frame[[k for k in columns if k in frame]].to_markdown(index=False,floatfmt='.6g')

def build():
    decision=load('decision.json');summary=pd.read_csv(RAW/'summary_metrics.csv')
    seeds=pd.read_csv(RAW/'per_seed_metrics.csv');free=pd.read_csv(RAW/'carrier_summary_metrics.csv')
    pairs=pd.read_csv(RAW/'paired_summary.csv');cost=load('cost_estimate.json')
    selection=decision['combination_selection']
    qa=json.loads((FIGURES/'FIGURE_QA.json').read_text(encoding='utf-8')) if (FIGURES/'FIGURE_QA.json').exists() else {'status':'PENDING'}
    lines=['# R13 长时间 Dirac 演化的失效归因与修复',
        f"\n更新时间（北京时间）：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。状态：{decision['status']}；图表核查：{qa['status']}。",
        '\n## 主要结果',
        f"\n完整终点任务 {decision['complete_jobs']}/{decision['unique_carrier_jobs']+decision['unique_coupled_jobs']}；数值/依赖失败 {len(decision['failed_jobs'])}；未达终点 {len(decision['incomplete_jobs'])}。E始终是解析先验诊断，不能算作学习载波成果。"]
    for case,groups in decision['per_case_independent_validation'].items():
        for group,result in groups.items():
            if not result['complete']:
                lines.append(f"\n- {case} / {group}：独立终点不完整，不能宣称成功。")
            else:
                m=result['means'];passed='通过' if result['pass_check'] else '未通过'
                lines.append(f"\n- {case} / {group}：{passed}；spinor={m['spinor_rel_l2_gauge_aligned']:.6g}、current={m['current_rel_l2']:.6g}、fidelity={m['fidelity_mean']:.6g}、modal={m['modal_population_rel_l2']:.6g}、max norm drift均值={m['max_norm_drift']:.6g}，联合通过{result['joint_pass_count']}/5。")
    if not selection['selected']:
        lines.append('\n未找到同时满足预注册组件与组合资格的方案，未向独立验证追加其他修复。A/E新种子结果仍完整保留。')
    else:
        independent=decision['per_case_independent_validation']
        recovered=[c for c in ('T10','m10') if independent[c].get('selected_combination',{}).get('pass_check')]
        failed=[c for c in ('T10','m10') if c not in recovered]
        lines.append(f"\n所选组合的独立绝对成功条件：{', '.join(recovered) or '无'}；未通过/不完整条件：{', '.join(failed) or '无'}。载波单项的归因仍限于600–602消融；新种子验证的是组合整体。")
    matched=pairs[(pairs.stage=='T10_factorial')&(pairs.target=='combined')&(pairs.baseline=='E_repaired_envelope_alias')]
    precision=matched[matched.metric=='spinor_rel_l2_gauge_aligned']
    expense=matched[matched.metric=='full_attributed_pipeline_seconds']
    if len(precision) and len(expense):
        ratio=float(precision.ratio_of_means.iloc[0]);cost_ratio=float(expense.ratio_of_means.iloc[0])
        lines.append(f'\n严格同包络的探索诊断：组合/E修复包络的旋量误差比={ratio:.6g}，归属总成本比={cost_ratio:.6g}。')
        if ratio>=1 and cost_ratio>=1:
            lines.append('\n精确载波在这个同包络诊断中同时更精确且更经济，学习载波的实际价值尚未建立。')
        else:
            lines.append('\n此诊断存在精度或成本权衡；只记录实测比值，不能仅凭三个探索种子宣布学习载波具有普遍经济优势。')
    lines.extend(['\n## 历史证据、物理与桥接',
        '\nR9原A的旋量/流平均误差约.0343/.0464，部分种子范数超标；R10沿原种子追加预算，不能当独立复现。R11的新种子A通过4/5，C从头联合训练通过5/5，不能预设预训练必要。R12的T10中A/C显著失败，而E保持较小动力学误差但仍有范数失败；m10的E通过，学习组失败。本轮保护历史结论，主检验载波与守恒两类问题。',
        '\n固定161模态[-64,96]、k=n/8、特征(k-2)/10、x∈[-8π,8π)、V=.2cos(x)。桥接m5T5、主条件m5T10、交叉m10T5。初始谱是R5 n0…32的原归一高斯与等权±支，扩展只零填充；m10重算初始旋量。桥接同时使用新网格与尺度，且只有3个探索种子，不能单独隔离模态数、尺度或种子影响，也不能冒称5种子复现。',
        '\n[桥接、全部探索及验证均值表](results/raw/r13_long_time_recovery/summary_metrics.csv)；[参考校准](results/raw/r13_long_time_recovery/reference_validation.json)。',
        '\n'+table(summary[summary.stage=='bridge'],['case','group','n','spinor_rel_l2_gauge_aligned_mean','current_rel_l2_mean','fidelity_mean_mean','modal_population_rel_l2_mean','max_norm_drift_mean','joint_pass_count']),
        '\n参考固定161对193的零填充全复轨迹差、两端各16层占据、厄米性、范数、逆演化、独立矩阵指数及PDE均在训练前通过；原始33点高斯未重归一。参考不进入优化，物理参数和截断不由PINN精度决定。Nx1024,Nt201；另测Nx2048积分差，平方双线性误差最大Fourier索引320，网格不混叠。',
        '\n## 冻结设计与预算',
        '\n[预注册](docs/experiments/r13_long_time_recovery/preregistration.md)、[完整协议](results/raw/r13_long_time_recovery/protocol.json)、[科学代码SHA-256](results/raw/r13_long_time_recovery/frozen_protocol_and_code.json)、[理论与导数核查](docs/experiments/r13_long_time_recovery/theory_check.md)、[分批恢复计划](docs/experiments/r13_long_time_recovery/batch_recovery_plan.md)。',
        '\n网络4×64 tanh、双精度、Adam lr=.001、32新时间×161模态，完整未加权旋量PDE。A及修复A类6000自由预训练+20000耦合；E只有20000耦合。所有耦合包络逐元素配对、采样流独立于初始化并共享标准化时间前缀。冻结权重仍保留载波时间导数与缺陷，手写精确链式导数对照独立自动微分及全部参数梯度。原A的前向和残差与R12逐元素一致。',
        '\n相位缩放使用已知mT，不提供逐k精确频率；初始化末层除mT使初始相位函数匹配原型。静态频率仅输入k，从完整自由残差学习，并按原网络τ=0路径匹配初始斜率；减少64参数，不能宣称载波全部权重配对。缩放和静态分别检验。',
        '\n总范数正则固定λ=.01，没有搜索。硬范数d=r/√Σ|r|²涵盖全谱及两支，允许交换占据；完整求导包含归一化分母的跨模态项，训练/评价同前向，未后处理归一化。它结构上保证总概率，仍须通过其余动力学指标。参数化在r=0处未定义，禁止幅值阈值或静默夹断，遇到则保留失败。',
        '\n只用600–602的固定6000/20000终点选择组件及最多一个组合。若不满足预注册资格，仍完成原A/E的700–704独立对照，不另选组合或延长预算。探索通过规则是五项均值+2/3，正式绝对成功始终是五种子五项均值+至少4/5。',
        '\n## 自由载波：初占据相位与全谱缺陷',
        '\n'+table(free,['stage','case','group','n','initial_weighted_unwrapped_phase_rms_mean','initial_weighted_unwrapped_phase_rms_population_sd','initial_weighted_unit_spinor_residual_rms_mean','carrier_free_pde_residual_rms_mean']),
        '\n[载波逐种子数据](results/raw/r13_long_time_recovery/carrier_per_seed_metrics.csv)；[载波选择及证据哈希](results/raw/r13_long_time_recovery/carrier_selection.json)。相位以t=0为锚点，未作逐模态/逐时间对齐；初始权重严格来自原高斯。全谱单位缺陷不能代替实际支持上的相位误差；耦合评价另保存参考瞬时占据加权的自由相位诊断，该未来占据不参与选择或训练。',
        '\n## E固定：包络修复',
        '\n'+table(summary[summary.stage=='exact_envelopes'],['case','group','n','spinor_rel_l2_gauge_aligned_mean','current_rel_l2_mean','fidelity_mean_mean','active_coherence_rel_l2_mean','max_norm_drift_mean','joint_pass_count']),
        '\n[包络选择](results/raw/r13_long_time_recovery/envelope_selection.json)。这一对照固定精确载波，因此误差差异支持包络结构/目标对本预算优化的作用。硬范数漂移很小是结构恒等式，不是独立学得守恒的证据；应同时看相位相干、概率流与占据。',
        '\n## T10四组消融及组合选择',
        '\n'+table(summary[summary.stage=='T10_factorial'],['group','n','alias','spinor_rel_l2_gauge_aligned_mean','current_rel_l2_mean','active_coherence_rel_l2_mean','max_norm_drift_mean','joint_pass_count']),
        f"\n载波候选：{selection['carrier_kind']}；包络候选：{selection['envelope_kind']}；组合进入独立验证：{selection['selected']}。选择JSON保留全部候选的资格、固定排序与来源哈希。",
        '\n同case/seed自由来源复用，所有耦合包络和Adam重新初始化。E alias直接引用预定E包络实验，不能当作新增重复或另计训练成本。另列同包络修复的E alias，可区分载波效果与范数结构收益；新种子验证中的E仍是原PDE-only，因此组合对原E的收益不能全归于载波。',
        '\n## 新种子独立验证与实际误差比',
        '\n'+table(summary[summary.stage=='independent_validation'],['case','group','n','spinor_rel_l2_gauge_aligned_mean','spinor_rel_l2_gauge_aligned_population_sd','current_rel_l2_mean','fidelity_mean_mean','modal_population_rel_l2_mean','max_norm_drift_mean','joint_pass_count','absolute_success']),
        '\n'+table(pairs[(pairs.stage.isin(['T10_factorial','independent_validation'])) & pairs.metric.isin(['spinor_rel_l2_gauge_aligned','current_rel_l2','max_norm_drift','full_attributed_pipeline_seconds'])],
            ['stage','case','target','baseline','metric','n','ratio_of_means','paired_difference_mean','paired_difference_population_sd','favorable_pair_count']),
        '\n[逐种子配对差](results/raw/r13_long_time_recovery/paired_differences.csv)；[实际比值与population SD](results/raw/r13_long_time_recovery/paired_summary.csv)。比值是实际结果，未增加事后优势倍数门槛。不同条件逐个判定，不取跨算例平均掩盖失败。',
        '\n## 成本、残差工作量与恢复',
        f"\n预估累计绑定单核训练{cost['estimated_total_accumulated_bound_core_hours']:.2f}小时、批次墙钟约{cost['estimated_batch_wall_hours']:.2f}小时，评价与I/O另计。实际唯一任务记录训练{decision['unique_incurred_training_hours']:.3f}小时、包含setup/评价/I/O的已计管线费用{decision['unique_incurred_pipeline_hours']:.3f}小时。这些并发绑定核心的计时不能当作隔离硬件加速证据。",
        '\n[唯一成本账本](results/raw/r13_long_time_recovery/unique_cost_ledger.csv)、[全部预定检查点](results/raw/r13_long_time_recovery/snapshot_index.csv)、[调度与实际核分配](results/raw/r13_long_time_recovery/runtime_plan.json)。每run保存活动/冻结/总参数、预训练/耦合/评价/setup/I/O耗时、两支单位及耦合旋量残差点、范数正则额外前向点和peak RSS。固定恢复间隔500步。来源复用在实际实验成本计一次，每个算法的独立流程仍归属完整6000预训练成本。',
        '\n原/缩放载波12802参数，静态12738，包络12932；A总25734、静态修复总25670、E总12932。耦合阶段学习组载波冻结；E解析先验不计作可训练参数，不能据此省略其先验属性。RSS含长期worker的缓存分配器，不是每模型隔离内存。未落盘时被外部终止的重复片段不在已恢复成本累计内，runtime历史与日志保留该限制。',
        '\n成本观察边界：终点使用单独execution_cost收据，覆盖最终快照CSV/NPZ导出；不可变metrics.json是在该导出前写入，两者原值都保留。中间曲线点计时至该次评价结束，当前点导出费用进入后续点，未估算为已测值。根目录最终复制、模块启动和控制进程部分开销未完全纳入每run加总；批次墙钟包含这些开销及空闲等待。此限制不改变训练预算、精度或选择规则。',
        '\n'+table(summary[summary.stage=='independent_validation'],['case','group','n',
            'carrier_training_seconds_mean','coupled_training_seconds_mean','evaluation_seconds_mean',
            'full_attributed_pipeline_seconds_mean','full_attributed_pipeline_seconds_population_sd',
            'training_spinor_residual_points_mean','peak_rss_mib_mean']),
        '\n## 机制证据与解释边界',
        '\n已证实的恒等式：单位模载波与正交旋量不直接改变总谱概率；载波缺陷及实势的直接全局范数投影为零。包络投影控制Ṅ=2 Im〈s,R〉，但载波仍可能间接影响其优化。保存norm balance、载波/包络残差分解与谱外负耦合源，不能把这些数值积分当作严格连续误差认证。',
        '\n相位与自由缺陷改善是载波修复的直接证据；只有耦合carrier-only及新种子结果同时改善，才支持其恢复本问题长时精度。E包络对照可检验守恒改善是否保留流与相干；硬范数不能凭N一项声明动力学成功。组件消融量是固定预算的描述性证据，不排除优化、参数化先验和容量差异共同影响；失败不证明数学不可表示。',
        '\n若同包络修复下的E仍更精确且更经济，应报告学习载波实际价值尚未建立。若组合仅胜过原E的范数漂移、却未胜过修复E的动力学/成本，不能把守恒结构贡献归于学习载波。本轮未用700–704再增加“修复E”run，因此该严格同结构的新种子成本比较仍是边界。',
        '\nR11从头联合训练成功表明预训练不是已证明的必要条件。本轮只检验所列预训练修复，不能据此否定联合训练。缩放、自主频率、范数正则与径向约束作为已知思路的修复对照，原创性未核查，不作新方法首创声明。范围仅已知有限Fourier高斯初谱、所定弱周期势与重训条件，不宣称未知连续谱发现、一般外势或通用Dirac求解器。',
        '\n## 全部种子终点数据',
        '\n'+table(seeds,['stage','case','group','seed','alias','spinor_rel_l2_gauge_aligned','current_rel_l2','fidelity_mean','modal_population_rel_l2','max_norm_drift','active_coherence_rel_l2','predicted_max_occupation_outside_initial_support','residual_integral_fine','training_seconds','full_attributed_pipeline_seconds','peak_rss_mib']),
        '\n[全run六件套与恢复状态索引](results/raw/r13_long_time_recovery/attempt_index.json)。失败列表如下；未达到固定终点不伪造指标。',
        '\n```json\n'+json.dumps(decision['failed_jobs']+decision['incomplete_jobs'],ensure_ascii=False,indent=2)+'\n```',
        '\n## 图表与审计'])
    if (RAW/'remaining_resource_gate.json').exists():
        resource=load('remaining_resource_gate.json');compaction=load('storage_compaction.json')
        position=lines.index('\n## 机制证据与解释边界')
        lines.insert(position,
            '\n2026-10-10电脑重启后按原模型、Adam及采样/RNG断点精确恢复。整轮最初25 GiB存储预留阻止启动时，仅对已完成R13根目录与最终快照的SHA-256完全相同文件改用NTFS硬链接，全部路径和内容保留、R1–R12不动，'
            f"释放{compaction['recovered_bytes']/2**30:.3f} GiB。剩余6个m10 run按实测快照体积、1.25倍快照余量、2倍导出余量及4 GiB后处理余量重算，要求{resource['required_tail_free_bytes']/2**30:.3f} GiB、当时可用{resource['observed_free_bytes']/2**30:.3f} GiB并通过。"
            '这是一项公开记录的运行资源修订，未改变科学SHA、物理、选择规则、损失、种子、预定检查点或优化步数。'
            '\n[资源修订与断点哈希](results/raw/r13_long_time_recovery/remaining_resource_gate.json)；[逐文件内容核查及存储处理](results/raw/r13_long_time_recovery/storage_compaction.json)。'
            '\n自动训练收尾曾因图表核查的路径拼接错误而停止；只修订未冻结的后处理模块，没有重新训练或筛选结果。')
    if (RAW/'final_test_result.json').exists():
        tests=load('final_test_result.json');audited=load('audit.json')
        lines.append(f"\n完整pytest：{tests['status']}（{tests.get('passed_tests','见XML')}个通过）；最终审计：{audited['status']}，核查{audited.get('checked_count',0)}项，失败{audited.get('failed_count',0)}项。")
        lines.append('\n[完整测试日志](results/logs/r13_long_time_recovery/full_pytest_final.log)；[测试XML](results/logs/r13_long_time_recovery/full_pytest_final.xml)；[链接检查](results/raw/r13_long_time_recovery/link_check.json)。')
    registry=json.loads((FIGURES/'figure_registry.json').read_text(encoding='utf-8')) if (FIGURES/'figure_registry.json').exists() else {'figures':[]}
    for r in registry['figures']:
        lines.append(f"\n- [{r['name']} PNG]({r['exports']['png']['path']}) / [SVG]({r['exports']['svg']['path']}) / [PDF]({r['exports']['pdf']['path']}) / [source CSV]({r['source_csv']})。{r['contract']}")
    lines.append('\n[FIGURE_QA](results/figures/r13_long_time_recovery/FIGURE_QA.json)、[审计](results/raw/r13_long_time_recovery/audit.json)。图需实际PNG及PDF渲染检查后才能签视觉PASS；自动脚本不会代签。完整pytest、历史/冻结SHA核查、链接检查的结果保存在本轮logs和audit中。')
    path=ROOT/'R13_LONG_TIME_RECOVERY_REPORT.md'
    content='\n'.join(lines)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=content:
        import shutil
        history=DOCS/'report_history';history.mkdir(exist_ok=True)
        shutil.copyfile(path,history/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.md'))
    path.write_text(content,encoding='utf-8');return path

if __name__=='__main__':print(build())
