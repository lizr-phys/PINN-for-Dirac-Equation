from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .r12_config import ROOT,RAW,DOCS,FIGURES,CASES,GROUPS,SEEDS,THRESHOLDS,CASE_LABELS,configuration
from .freeze_r7_phase_release import sha256

SHORT=dict(zip(GROUPS,('A 预训练冻结','C 联合训练','D 双复网络','E 解析诊断')))
METRICS=list(THRESHOLDS)

def link(path,label):return f'[{label}](<{Path(path).resolve().as_posix()}>)'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def number(value):return f'{value:.7g}'

def mean_sd(part,key):return number(part[key].mean())+' ± '+number(part[key].std(ddof=0))

def build():
    frame=pd.read_csv(RAW/'per_seed_metrics.csv',float_precision='round_trip')
    costs=pd.read_csv(RAW/'cost_and_residual_work.csv',float_precision='round_trip')
    comparison=pd.read_csv(RAW/'comparison_summary.csv',float_precision='round_trip')
    budget=pd.read_csv(RAW/'budget_change_summary.csv',float_precision='round_trip')
    dec=json.loads((RAW/'decision.json').read_text(encoding='utf-8'))
    refs=json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
    qa=json.loads((FIGURES/'FIGURE_QA.json').read_text(encoding='utf-8'))
    estimates=json.loads((RAW/'cost_estimate.json').read_text(encoding='utf-8'))
    masks=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    lines=['# R12：载波—复包络跨条件稳健性实验','',
        '实验ID：r12_cross_condition_robustness。四个新条件重新训练，seeds500–504，共80次正式运行；固定最终总26000步，没有精度驱动追加或选择性重跑。',
        '', '## 1. 结论先读','',
        'A与C并列检验。R11的A通过4/5、C通过5/5，已否定“预训练已被证明必不可少”的预设。E使用解析自由相位，单独标记，不计作学习载波成果。','']
    lines += ['绝对成功要求五种子均值同时满足：旋量相对L2≤0.10、概率流相对L2≤0.15、平均保真度≥0.98、模态占据相对L2≤0.10、测试网格最大范数漂移≤0.02，并且至少4/5种子同时通过全部五项。以下按算例独立判定，任何一项失败均不算完整动力学成功。','']
    rows=[]
    for case in CASES:
        rows.append([CASE_LABELS[case]]+[f'{dec["per_case"][case]["final"][g]["joint_pass_count"]}/5 '+
            ('PASS' if dec['per_case'][case]['final'][g]['pass_check'] else 'FAIL') for g in GROUPS])
    lines += [table(['条件']+[SHORT[g] for g in GROUPS],rows),'']
    passed={g:[c for c in CASES if dec['per_case'][c]['final'][g]['pass_check']] for g in GROUPS}
    for g in GROUPS[:2]:
        lines += [SHORT[g]+'有效的条件：'+('、'.join(CASE_LABELS[c] for c in passed[g]) if passed[g] else '本轮四个条件均未建立绝对成功')+'。','']
    cheaper=all(costs[(costs.case==c)&(costs.model==GROUPS[-1])].training_seconds.mean() <
        costs[(costs.case==c)&(costs.model==g)].training_seconds.mean() for c in CASES for g in GROUPS[:2])
    learned_conditions=set(passed[GROUPS[0]])|set(passed[GROUPS[1]])
    e_conditions=set(passed[GROUPS[-1]])
    if cheaper and learned_conditions and learned_conditions<=e_conditions:
        value_statement='E在四个算例的实际训练成本均低于A和C，且在所有学习组通过的条件也满足同一绝对规则。结合逐算例连续误差，当前尚未建立学习载波相对可用解析载波的实际价值；这不是统计等效证明，也不把解析成果算为学习成果。'
    elif not learned_conditions:
        value_statement='A和C在本轮四个条件均未建立完整动力学成功；学习载波的跨条件实用价值尚未建立。E的解析诊断和成本另列，不能把其成功转记为学习成果。'
    elif learned_conditions-e_conditions:
        value_statement='学习组在'+ '、'.join(CASE_LABELS[c] for c in CASES if c in learned_conditions-e_conditions)+'通过而E未通过同一规则，提供这些条件下的诊断差异证据。误差与成本仍需逐条件读取；不能将其唯一归因于载波近似或泛化为普遍实用优势。'
    else:
        value_statement='E与学习组的实际成本及连续误差见各算例表。当前不作无条件经济性或统计等效主张；学习载波的实际价值须由明确应用中的可用解析先验及成本共同判断。'
    lines += [value_statement,'']

    for case in CASES:
        data=frame[(frame.case==case)&(frame.endpoint=='final')]
        parts={g:data[data.model==g] for g in GROUPS}
        ratios=[parts[g].spinor_rel_l2_gauge_aligned.mean()/parts[GROUPS[2]].spinor_rel_l2_gauge_aligned.mean() for g in GROUPS[:2]]
        lines += [f'- {CASE_LABELS[case]}：A/C相对D的旋量均值比分别为{number(ratios[0])}/{number(ratios[1])}；'
            f'A/C/E的旋量均值为{number(parts[GROUPS[0]].spinor_rel_l2_gauge_aligned.mean())}/'
            f'{number(parts[GROUPS[1]].spinor_rel_l2_gauge_aligned.mean())}/{number(parts[GROUPS[3]].spinor_rel_l2_gauge_aligned.mean())}。']
    lines += ['', '每算例独立判定；优势与绝对成功不同，不能跨算例平均覆盖失败。五项门槛及至少4/5联合规则均未改变。', '',
        link(RAW/'per_seed_metrics.csv','全部逐种子指标')+' · '+link(RAW/'decision.json','逐算例判定')+' · '+link(RAW/'comparison_summary.csv','全部配对统计'),'','## 2. 大实验思路与公平边界','',
        'R1–R5研究已知自由能支上的表示与训练优势；R6/R7表明空通道可优化不保证完整耦合。R8成功的是可学习自由载波＋复包络方案，不能与原始纯实振幅LTPB混同。R9推广到宽高斯初谱，A旋量/流均值约.0343/.0464，但两个范数项超标，仅3/5；R10同祖先加预算达到5/5。R11新种子复现A=4/5，C=5/5，支持同时研究预训练冻结与从头联合更新。R12检验指定条件重训的稳健性，优先厘清有效范围与实际成本。','',
        '四算例仅改变T、m、势幅度或谐波结构；均重新初始化训练，不是零样本泛化或时间外推。四算例内部同161模式及同输入尺度；相对R11的129模式，k特征尺度按既有公式由8变10，故与R11跨轮差异不能完全归为单一物理变化。','',
        'A/C改变了辅助任务、预算分配及载波更新策略，不能单独识别冻结的因果作用。A/D与C/D匹配约总容量，却不匹配活动参数、表示或辅助工作。E/A的20000耦合快照匹配包络步数，仍有解析先验和预训练成本差异；E/C最终匹配耦合步数而活动参数/解析先验不同。E从自由算符取得精确相位，其效果只作诊断。','',
        '## 3. 物理、初谱和独立参考','',
        r'$$i\partial_t\psi=[-i\sigma_x\partial_x+m\sigma_z+V(x)I_2]\psi,\quad x\in[-8\pi,8\pi).$$','',
        r'$$g_n=C\exp[-(n/8-2)^2/(4\times0.5^2)],\quad n=0,\ldots,32,\quad\sum_n|g_n|^2=1,\quad c_{n,\pm}(0)=g_n/\sqrt2.$$','',
        '原始33个高斯系数保持不变，截断扩大只零填充，不重新归一化。m10的初始本征旋量与自由算符均按m=10计算，两支系数仍等权。','',
        table(['算例','m','T','V(x)','空间RMS'],[[c,configuration(c).mass,configuration(c).t_max,
            {'T10':'.2 cos(x)','m10':'.2 cos(x)','V04':'.4 cos(x)','two_harmonics':'.2[cos(x)+cos(2x)]/√2'}[c],number(configuration(c).as_dict()['potential_rms'])] for c in CASES]),'',
        r'$$R_n=i\dot s_n-(k_n\sigma_x+m\sigma_z)s_n-\sum_q\frac{a_q}{2}(s_{n-8q}+s_{n+8q}).$$','',
        'q=1和2分别对应n±8和n±16。双谐波保持势RMS，但不保持最大幅度、导数或耦合图；不能从RMS相同推断优化难度相同。','',
        '参考政策先于学习保存，按预定相邻截断检查，选第一个所有算例均合格的较小共同截断。完整复轨迹零填充差<10⁻⁶、两端各16层最大总占据<10⁻⁸。最终[-64,96]共161模式；较小129模式不足以覆盖全部四条件。Hermiticity、范数、逆演化、独立矩阵指数与原旋量PDE均独立验证。','',
        table(['算例','较小截断','较大截断','复轨迹相对差','下16层','上16层','合格'],[[r['case'],str(r['smaller']),str(r['larger']),number(r['full_complex_zero_padded_relative_difference']),number(r['lower_sixteen_layers_max_total_occupation']),number(r['upper_sixteen_layers_max_total_occupation']),r['pass_check']] for r in refs['comparisons']]),'',
        link(RAW/'reference_validation.json','完整参考校准')+' · '+link(RAW/'reference_selection_policy.json','参考选择政策')+' · '+link(RAW/'reference_convergence.csv','校准CSV'),'']
    rows=[]
    for c in CASES:
        m=json.loads((RAW/'references'/c/'diagnostic_ignoring_potential.json').read_text(encoding='utf-8'))
        rows.append([c,number(m['spinor_rel_l2_gauge_aligned']),number(m['current_rel_l2']),number(m['reference_max_occupation_outside_initial_support'])])
    lines += ['忽略外势的自由解作为独立物理影响诊断，不作为学习模型：','',table(['算例','自由解旋量误差','自由解流误差','耦合参考支持外最大占据'],rows),'',
        '## 4. 模型、导数与预注册','',
        r'$$p=e^{i\tau q_\theta},\quad d=c_0+\tau z,\quad c=pd,\quad \dot c=\dot p\,d+p(z/T+\tau\dot z).$$','',
        'A在每保留k的±本征旋量上作非零单位初态自由PDE预训练6000步，再冻结载波耦合20000步；C载波与包络从头完整PDE联合26000步；D双独立四实输出网络的和/√2训练26000步；E解析自由载波的复包络训练26000步。','',
        '全为4层64宽tanh、双精度、Adam .001，每步32新排序均匀时间×全部161模态。主loss沿用R11 mean_t sum_n mean_component |R|²/L，无额外模态权重或补救结构。冻结权重仍保留载波时间导数和近似缺陷；精确Linear/tanh链式内核与独立自动微分核查全部参数梯度。','',
        table(['组','耦合活动参数','冻结参数','总参数','辅助活动参数'],[['A',12932,12802,25734,12802],['C',25734,0,25734,0],['D',25864,0,25864,0],['E*',12932,0,12932,0]]),'',
        'D比A/C总参数多130，即约.505%；E的零载波参数不代表无物理先验。载波两输出网络不与四输出包络网络声称权重逐元素配对。','',
        '初始化与时间流独立，同seed同形网络逐位相同；四组及四算例共享归一时间随机数，T10物理时间加倍。A与其他组共享20000耦合前缀，C/D/E全部26000相同；自由单位IVP流独立。评价在固定权重保存后进行，训练RNG不被评价改变。矩阵参考未来从不进入训练，解析自由相位仅E允许。','',
        '科学冻结SHA-256：`'+sha256(RAW/'frozen_protocol_and_code.json')+'`。正式前29项工程测试及16组两步落盘smoke通过，无精度调参。固定5000/10000/15000/20000/26000耦合快照，不选择最好checkpoint。','',
        link(DOCS/'preregistration.md','预注册')+' · '+link(DOCS/'theory_check.md','理论核查')+' · '+link(RAW/'frozen_protocol_and_code.json','冻结清单'),'']
    for endpoint in ('final','coupled_20000'):
        lines += ['## '+('5. 固定最终结果' if endpoint=='final' else '6. 同20000耦合步数结果'),'']
        for c in CASES:
            data=frame[(frame.case==c)&(frame.endpoint==endpoint)]
            lines += ['### '+CASE_LABELS[c],'',table(['组','Spinor','Current','Fidelity','Modal','Max norm drift','联合通过','判定'],[
                [SHORT[g]]+[mean_sd(data[data.model==g],k) for k in METRICS]+[str(dec['per_case'][c][endpoint][g]['joint_pass_count'])+'/5',
                'PASS' if dec['per_case'][c][endpoint][g]['pass_check'] else 'FAIL'] for g in GROUPS]),'']
            rows=[]
            for _,r in data.iterrows():
                ok=all(r[k]>=v if k=='fidelity_mean' else r[k]<=v for k,v in THRESHOLDS.items())
                rows.append([SHORT[r.model],int(r.seed)]+[number(r[k]) for k in METRICS]+['PASS' if ok else 'FAIL'])
            lines += [table(['组','Seed','Spinor','Current','Fidelity','Modal','Max norm drift','联合'],rows),'']
        lines += ['均值±population SD，ddof0，n=5。A的final与coupled20000引用同一快照，不能作两次独立重复；C/D/E两个终点来自同一训练祖先。','']
    lines += ['## 7. 配对优势及预算证据','',
        '以下只列两项主场误差的真实比值与配对差；所有五门槛及辅助量的逐seed差均保存在CSV，未新增优势门槛。差为target−baseline。','',
        table(['算例','终点','Target/Baseline','指标','均值比','配对差均值±SD','有利配对'],[
            [r.case,r.endpoint,SHORT[r.target]+'/'+SHORT[r.baseline],r.metric,number(r.ratio_of_means),number(r.paired_difference_mean)+' ± '+number(r.paired_difference_population_sd),str(int(r.favorable_pair_count))+'/5']
            for _,r in comparison[comparison.metric.isin(METRICS[:2])].iterrows()]),'',
        'C/D/E在同一祖先20000→26000耦合步数的预定变化：','',
        table(['算例','组','指标','26000/20000均值比','配对差均值±SD','改善配对'],[[r.case,SHORT[r.model],r.metric,number(r.ratio_of_means),number(r.paired_difference_mean)+' ± '+number(r.paired_difference_population_sd),str(int(r.favorable_pair_count))+'/5'] for _,r in budget[budget.metric.isin(METRICS[:2]+['max_norm_drift'])].iterrows()]),'',
        link(RAW/'paired_comparisons.csv','全部逐seed配对差')+' · '+link(RAW/'budget_changes.csv','逐seed固定追加预算变化'),'','## 8. 相干、支持与残差诊断','',
        '评价Nx1024、Nt201，Fourier双线性误差平方最高指数320<1024，并核验Parseval与2048点积分。一项全时空常数相位用于旋量/复系数，raw误差另外保留，不逐时或逐模对齐。相干mask仅参考支占据≥10⁻⁶；预测塌缩仍计入，相位定义覆盖率单独保存。支持外为n<0或n>32，中心外原本宽高斯系数不叫新侧带。','',
        table(['算例','组','相干误差','支复系数误差','支持外最大占据','保留残差积分','外残差积分','载波单位缺陷RMS'],[
            [c,SHORT[g]]+[mean_sd(frame[(frame.case==c)&(frame.model==g)&(frame.endpoint=='final')],k) if frame[(frame.case==c)&(frame.model==g)&(frame.endpoint=='final')][k].notna().any() else 'N/A' for k in
            ('active_coherence_rel_l2','branch_complex_coefficient_rel_l2','predicted_max_occupation_outside_initial_support','residual_integral_fine','exterior_residual_integral','carrier_free_pde_residual_rms')] for c in CASES for g in GROUPS]),'',
        '残差积分T5为4001点、T10为8001点，间距均.00125；粗网格每隔一点。外残差为零延拓处−Vψ；双谐波重叠源项先求和再平方。逐模式未加权复残差、载波缺陷、支中心/宽度、范数投影恒等式及粗细积分差全部归档。','',
        r'$$\dot N=2\operatorname{Im}\langle\psi,R\rangle,\qquad \|e(t)\|\le\|e(0)\|+\int_0^t\|R(s)\|\,ds.$$','',
        '该连续恒等式的数值梯形积分不是严格连续或无限谱认证；误差界对应raw误差。单位模载波使自由缺陷的直接全局范数投影为零，但仍间接影响包络优化，不能把范数漂移唯一归因载波。C存在内部相位规范自由度，完整场准确不等于内部载波唯一恢复自由色散，全谱单位缺陷也不能单独否定C。','',
        '## 9. 全阶段成本','',
        f'正式前根据R11实测和161/129模式比估计约{estimates["estimated_total_core_hours"]:.2f}核小时、五worker约{estimates["estimated_five_worker_wall_hours"]:.2f}小时，评价/I/O另计。实际成本如下，秒，均值±population SD：','',
        table(['算例','组','总训练','预训练','耦合训练','全部评价','落盘I/O','RSS MiB','旋量残差点'],[
            [c,SHORT[g]]+[mean_sd(costs[(costs.case==c)&(costs.model==g)],k) for k in ('training_seconds','carrier_training_seconds','coupled_training_seconds','evaluation_seconds','checkpoint_and_artifact_io_seconds','peak_rss_mib')]+
            [int(costs[(costs.case==c)&(costs.model==g)].training_spinor_residual_points.iloc[0])] for c in CASES for g in GROUPS]),'',
        f'实际80次已保存优化轨迹累计训练计时为{costs.training_seconds.sum()/3600:.3f}小时（各进程单核绑定）。共2080000已保存优化步、11334400000旋量残差点、22668800000复标量分量评价。A辅助阶段两个单位支IVP单独计数。380预定成本点全部保留。该累计计时不等于整机日历墙钟或直接测量的CPU忙碌时间。','',
        '五worker实际mask1/2/4/8/16，控制核128；启动前互斥核查，初次OS实测及各worker进程结束记录均核验实际分配。各batch新worker，内部同seed四组持久进程，RSS含分配器缓存；共享内存、缓存、功耗与系统负载。实际并发墙钟/单核阶段耗时不作为隔离硬件算法加速证据。训练计时包含优化、全部预训练；评价/I/O/会话耗时另列，未覆盖的小量初始化/复制开销归会话elapsed。','',
        '发生两次已观察到的进程中断，原因未确定；完成结果不重训，未完成运行从权重、Adam及独立采样/RNG状态精确恢复，仍在原定优化步数停止。日志、原调度状态及恢复核验均保存在runtime_history和recovery_observation文件。若中断时存在恢复间隔内尚未落盘的计算，本表不覆盖该部分，故记录成本和唯一优化轨迹工作量不视为严格无遗漏的整机实际工作账本。','',
        link(RAW/'cost_and_residual_work.csv','80次完整成本')+' · '+link(RAW/'error_cost_curves.csv','380固定误差成本点')+' · '+link(RAW/'runtime_plan.json','分批CPU调度'),'','## 10. 归因与解释范围','']
    lines += ['本轮实际结果：T10中A/C不仅范数失败，旋量、概率流、保真度和模态占据均未达到均值门槛；m10中A另外四项均值达标，主要失败项为范数，而C有多项场误差超标。强单谐波与等RMS双谐波中A/C的另外四项均值达标，但范数漂移和种子联合通过数未达要求。四个算例的D均出现严重范数衰减和场失真。学习载波流程的连续场误差在每个算例均优于D，但这些比较优势没有转化为跨条件绝对成功。','',
        '预训练并非必要性的结论不能由本轮A/C比较重建：A在m10和两个外势算例的旋量均值优于C，T10中则C优于A；双谐波概率流均值也是C略优。流程排序随条件和指标变化，且A/C同时改变辅助任务、耦合预算与载波冻结策略。','',
        '解析E仅m10达到5/5完整通过，在其余三个条件也出现范数门槛失败。因此，学习载波误差不能独自解释全部失败。E在四条件的记录训练计时均低于A/C，T10和m10场误差明显较小，两个外势条件与学习组的误差排序不同；这些具体比值见配对表，未建立统计等效或无条件优越。','',
        '预定20000→26000检查点没有使任何C或D算例转为完整通过；E在m10的两个终点均5/5通过，T10和双谐波的联合通过数反而分别从3降到1、从3降到2。额外步数可改善部分场指标，但不保证范数或完整成功单调改善。本轮没有检验26000步以外预算，不能断言再增加预算一定解决问题，也不能断言有限网络无法表示。','']
    unit_defects={c:frame[(frame.case==c)&(frame.model==GROUPS[0])&(frame.endpoint=='final')].carrier_free_pde_residual_rms.mean() for c in CASES}
    lines += ['A冻结载波的全保留谱单位自由PDE缺陷RMS均值（独立评价）依次为：'+
        '；'.join(c+'='+number(unit_defects[c]) for c in CASES)+'。V04与双谐波的自由阶段同m、T、初始网络及采样流，所得单位缺陷相同；外势没有进入该预训练阶段。T10和m10的缺陷明显较大，支持不同自由振荡条件下固定6000步预训练的近似质量并不稳健。该量包含初态零填充的远端单位通道，不等同高斯加权完整场误差，也不能唯一归因范数漂移；两个m=5、T=5外势条件载波缺陷较小仍未完整通过，E也有范数失败。证据更支持流程优化及有限预算敏感性，尚不能把失败归为数学不可表示或单一载波误差。','']
    for c in CASES:
        final=dec['per_case'][c]['final'];early=dec['per_case'][c]['coupled_20000']
        transitions=[SHORT[g] for g in GROUPS[1:] if final[g]['pass_check'] and not early[g]['pass_check']]
        lines += [f'- {CASE_LABELS[c]}：A/C/D/E最终判定依次为'+ '/'.join('PASS' if final[g]['pass_check'] else 'FAIL' for g in GROUPS)+
            ('；'+ '、'.join(transitions)+'在预定额外6000耦合步后由FAIL变PASS，直接支持该范围内预算敏感性。' if transitions else '；预算变化的连续误差见第7节，未用终点精度追加训练。')]
    lines += ['',
        '证据：参考截断在学习前独立合格；失败不能直接归为参考校准问题。若A/C优于D，说明仅总容量不足以解释结果。E与学习组的差异同时改变解析先验、表示坐标、活动参数及成本，不能把差值当作可加分离的纯载波误差。','',
        '推测：长时/质量增加提高快速振荡难度；强势或新增谐波增加包络耦合要求。但当前固定结构与预算没有隔离所有优化因素，无法唯一判定失败来自预算、耦合复杂性还是有限网络逼近。固定预算失败不证明数学不可表示，不追加补救模型。','',
        value_statement+' 不另设相当性的事后倍数门槛；同一绝对判据通过不等于两方法误差统计等效。','',
        '结论只覆盖已知有限Fourier高斯初谱在四个指定条件下的重新训练；不宣称未知连续谱发现、一般外势、零样本泛化、时间外推或通用Dirac求解器。','',
        '## 11. 图表与完整产物','']
    for f in qa['figures']:
        name=f['figure'];lines += ['### '+name,'',f['claim'],'',
            link(FIGURES/(name+'.png'),'PNG')+' · '+link(FIGURES/(name+'.svg'),'SVG')+' · '+link(FIGURES/(name+'.pdf'),'PDF')+' · '+
            ' · '.join(link(ROOT/s['path'],'source CSV') for s in f['source_data']),'']
    lines += [link(FIGURES/'FIGURE_QA.md','FIGURE_QA')+' · '+link(DOCS/'figure_contract.md','图表契约')+' · '+link(RAW/'run_artifact_index.csv','每run六件套与恢复索引')+' · '+link(RAW/'snapshot_index.csv','所有预定快照'),'']
    for c in CASES:lines += [link(RAW/'case_statistics'/c/'per_seed_metrics.csv',c+'逐seed数据')+' · '+link(RAW/'references'/c/'selected_reference.json',c+'参考路径/哈希')]
    lines += ['','## 12. 最终验证','']
    for filename in ('pytest_final_result.json','audit.json','link_check.json'):
        p=RAW/filename
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'));lines += [filename+': '+r.get('status','UNKNOWN')+'；'+link(p,'证据')+'。']
        else:lines += [filename+': PENDING；完整训练、视觉QA及最终检查完成后更新。']
    lines += ['', '后处理曾因primary_groups元组经JSON序列化变为列表而触发两项一致性审计失败。将审计比较对象规范为JSON后，原始JSON与CSV重算的所有数值及判定逐项精确一致，没有浮点容差放宽或科学规则修改；原失败审计和修复前后处理源文件保留在postprocessing_history。科学冻结文件未改变。','',
        '全部R1–R11报告、科学代码和历史产物受9485文件SHA-256清单保护。R12使用新增源文件和独立raw/figures/logs/docs目录，不改旧结论。科学成功与工程完整性分别报告。','']
    report=ROOT/'R12_ROBUSTNESS_REPORT.md';report.write_text('\n'.join(lines),encoding='utf-8')
    print('Built',report.name,flush=True);return report

if __name__=='__main__':build()
