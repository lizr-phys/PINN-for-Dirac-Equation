from __future__ import annotations
import json
import math
from datetime import datetime, timezone
import pandas as pd
from .r11_config import (ROOT,RAW,DOCS,FIGURES,LOGS,GROUPS,SEEDS,THRESHOLDS,
    ARTIFACTS,run_directory,budget)
from .freeze_r7_phase_release import sha256

def link(label,path):
    return f"[{label}](<{path.resolve().as_posix()}>)"

def fmt(value):
    if value is None or isinstance(value,float) and not math.isfinite(value):
        return "N/A"
    if isinstance(value,bool): return "PASS" if value else "FAIL"
    if isinstance(value,(int,float)): return f"{value:.8g}"
    return str(value)

def table(frame,columns):
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.to_dict("records"):
        lines.append("| "+" | ".join(fmt(row.get(c)) for c in columns)+" |")
    return "\n".join(lines)

def build():
    frame=pd.read_csv(RAW/"per_seed_metrics.csv",float_precision="round_trip")
    summary=pd.read_csv(RAW/"summary_metrics.csv",float_precision="round_trip")
    comparisons=pd.read_csv(RAW/"comparison_summary.csv",float_precision="round_trip")
    costs=pd.read_csv(RAW/"cost_and_residual_work.csv",float_precision="round_trip")
    decision=json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    reference=json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    frozen=sha256(RAW/"frozen_protocol_and_code.json")
    def cohort(g,endpoint="final"):
        return decision["absolute_success"][endpoint][g]
    def mean(g,key,endpoint="final"):
        return cohort(g,endpoint)["means"][key]
    def contrast(g,b,key,endpoint="final"):
        return comparisons[(comparisons.endpoint==endpoint)&(comparisons.target==g)&
            (comparisons.baseline==b)&(comparisons.metric==key)].iloc[0]
    def mean_sd(g,key,endpoint="final"):
        p=frame[(frame.model==g)&(frame.endpoint==endpoint)][key]
        return f"{p.mean():.8g} ± {p.std(ddof=0):.8g}"
    apass=cohort(GROUPS[0])["pass_check"]
    lines=["# R11：高斯波包外势问题的独立复现与载波预训练消融","",
        "实验ID：r11_independent_replication_ablation。日期："+
        datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")+"（本机时间，Asia/Shanghai）。",
        "独立新种子400–404；沿用同一科学实现和参考，不冒充独立代码库、硬件或未知谱发现。","",
        "## 1. 主要结果与五个问题","",
        "**A "+("通过" if apass else "未通过")+"预注册独立复现标准，"+
        str(cohort(GROUPS[0])["joint_pass_count"])+"/5种子联合通过。** "+
        "五项均值和至少4/5联合规则均须满足；比较优势与绝对成功分别判断。E始终为精确载波解析诊断，不计作学习方法成果。",""]
    main=[]
    for g in GROUPS:
        c=cohort(g)
        main.append(dict(Group=g.split("_")[0]+("*诊断" if g==GROUPS[4] else ""),
            Spinor=mean_sd(g,"spinor_rel_l2_gauge_aligned"),
            Current=mean_sd(g,"current_rel_l2"),Fidelity=mean_sd(g,"fidelity_mean"),
            Modal=mean_sd(g,"modal_population_rel_l2"),Norm=mean_sd(g,"max_norm_drift"),
            Joint=f"{c['joint_pass_count']}/5",Criterion="PASS" if c["pass_check"] else "FAIL"))
    lines += [table(pd.DataFrame(main),["Group","Spinor","Current","Fidelity","Modal","Norm","Joint","Criterion"]),"",
        "均值±population SD（ddof=0），n=5。Norm先取每seed在统一201点网格的最大漂移，再跨seed统计；4001点更细诊断另报。A–D主终点总26000步，E为20000步耦合诊断。",""]
    ab=contrast(GROUPS[0],GROUPS[1],"spinor_rel_l2_gauge_aligned","coupled_20000")
    abj=contrast(GROUPS[0],GROUPS[1],"current_rel_l2","coupled_20000")
    ac=contrast(GROUPS[0],GROUPS[2],"spinor_rel_l2_gauge_aligned")
    acj=contrast(GROUPS[0],GROUPS[2],"current_rel_l2")
    cb=contrast(GROUPS[2],GROUPS[1],"spinor_rel_l2_gauge_aligned")
    cbj=contrast(GROUPS[2],GROUPS[1],"current_rel_l2")
    ad=contrast(GROUPS[0],GROUPS[3],"spinor_rel_l2_gauge_aligned")
    adj=contrast(GROUPS[0],GROUPS[3],"current_rel_l2")
    ea=contrast(GROUPS[4],GROUPS[0],"spinor_rel_l2_gauge_aligned")
    eaj=contrast(GROUPS[4],GROUPS[0],"current_rel_l2")
    lines += [
        "1. **A是否独立复现：** "+("本题给定预算下五新种子达到冻结的绝对规则。" if apass else
        "这批新种子在给定预算下没有满足完整规则；保留所有失败，不能以低平均旋量误差替代联合标准。")+
        f"平均spinor/current={mean(GROUPS[0],'spinor_rel_l2_gauge_aligned'):.8g}/{mean(GROUPS[0],'current_rel_l2'):.8g}。",
        f"2. **学到的载波相对随机冻结载波：** 同20000耦合步数 A/B spinor/current均值比={ab.ratio_of_means:.8g}/{abj.ratio_of_means:.8g}；"+
        f"旋量/流有利配对数={int(ab.favorable_pair_count)}/5、{int(abj.favorable_pair_count)}/5。"+
        ("两项平均误差均支持辅助预训练得到的载波在这个冻结流程中有益。" if ab.ratio_of_means<1 and abj.ratio_of_means<1
         else "结果不能概括为预训练在两项平均误差上均有优势。")+
        "同耦合比较仍需计入A的辅助成本；主终点A/B也改变了耦合预算。",
        f"3. **预训练后冻结与联合训练：** 主终点 A/C spinor/current均值比={ac.ratio_of_means:.8g}/{acj.ratio_of_means:.8g}。"+
        f"C联合通过{cohort(GROUPS[2])['joint_pass_count']}/5，绝对规则"+
        ("通过" if cohort(GROUPS[2])["pass_check"] else "未通过")+"。"+
        f"同一次C轨迹在20000耦合步数为{cohort(GROUPS[2],'coupled_20000')['joint_pass_count']}/5，"+
        f"26000步为{cohort(GROUPS[2])['joint_pass_count']}/5；不能忽略额外预算的作用。"+
        ("因此本题在这批新种子和26000步预算下，辅助预训练并非达到完整动力学成功的必要条件。"
         if cohort(GROUPS[2])["pass_check"] else "联合训练是否成功仍按其全部绝对指标判断。")+
        f"A较低的平均误差与A/C联合通过数{cohort(GROUPS[0])['joint_pass_count']}/5、"+
        f"{cohort(GROUPS[2])['joint_pass_count']}/5须分开解释。"+
        "这比较的是两个完整训练流程，同时改变预训练、冻结和耦合步数，不能声称单因素冻结因果效应。"+
        f"从相同随机载波起点的C/B spinor/current均值比={cb.ratio_of_means:.8g}/{cbj.ratio_of_means:.8g}，"+
        f"有利配对数={int(cb.favorable_pair_count)}/5、{int(cbj.favorable_pair_count)}/5。"+
        "B/C只考察从头训练时是否允许载波更新，活动参数也不同，不能外推为预训练载波冻结的单因素效应。",
        f"4. **是否超过容量匹配控制：** A/D spinor/current均值比={ad.ratio_of_means:.8g}/{adj.ratio_of_means:.8g}，"+
        ("两项平均误差均较低，支持A的流程优势不能仅用总参数量解释。" if ad.ratio_of_means<1 and adj.ratio_of_means<1
         else "未建立两项平均误差均超过容量控制的结论。")+
        "D总参数比A多约0.505%，但A的活动参数、单位辅助任务及残差工作量不同，不等同等实际计算成本。",
        f"5. **精确载波的诊断：** E/A同20000耦合步数spinor/current均值比={ea.ratio_of_means:.8g}/{eaj.ratio_of_means:.8g}，"+
        f"E联合通过{cohort(GROUPS[4])['joint_pass_count']}/5。"+
        ("更准确的自由坐标在这两项平均误差上仍有收益，提示近似载波和包络优化有剩余改进空间。" if ea.ratio_of_means<1 and eaj.ratio_of_means<1
         else "精确自由载波没有在两项平均误差上同时改善A，故不能把剩余误差唯一归为载波近似。")+
        "E改变了解析先验及优化坐标，误差差值不是可加分离的纯载波误差；其效果不算学习成果。",""]
    for group in (GROUPS[0],GROUPS[4]):
        failures=[]
        for record in frame[(frame.model==group)&(frame.endpoint=="final")].to_dict("records"):
            missed=[key+"="+fmt(record[key]) for key,value in THRESHOLDS.items()
                if not (record[key]>=value if key=="fidelity_mean" else record[key]<=value)]
            if missed:
                failures.append("seed"+str(record["seed"])+"（"+"，".join(missed)+"）")
        lines += [group.split("_")[0]+"未联合通过的终点："+
            ("；".join(failures) if failures else "无")+"。全部保留，未替换或追加训练。",""]
    lines += ["## 2. 大实验脉络与本轮定位","",
        "R1–R5建立已知自由Fourier能支上的表示/训练优势，R6/R7表明空通道局部可优化不保证完整外势耦合。R8成功的是可学习自由载波＋复包络方案A，不能与原始纯实振幅LTPB混同。R9迁移到宽高斯初谱后A平均误差低于容量控制，但两个种子范数项超标，严格联合3/5。R10原五个祖先条件续训达到5/5，仍不属于新随机种子重复。","",
        "R11优先检验独立种子复现与预训练/更新策略归因，没有改变物理、损失、网络宽深或添加补救结构。没有时间窗、守恒正则、动态切换、幅值阈值、模态加权或预测归一化。","",
        " · ".join([link("R8报告",ROOT/"R8_COUPLING_RECOVERY_REPORT.md"),
            link("R9报告",ROOT/"R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md"),
            link("R10报告",ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md"),
            link("历史原始表核查",RAW/"history_review.json")]),"",
        "## 3. 物理、初态与参考","",
        "$$i\\partial_t\\psi=[-i\\sigma_x\\partial_x+5\\sigma_z+0.2\\cos(x)I_2]\\psi,"
        "\\quad x\\in[-8\\pi,8\\pi),\\quad T=5.$$","",
        "$$k_n=n/8,\\quad g_n=C\\exp[-(k_n-2)^2/(4\\times0.5^2)],\\quad n=0,\\ldots,32,"
        "\\quad\\sum_n|g_n|^2=1,\\quad c_{n,\\pm}(0)=g_n/\\sqrt2.$$","",
        "所选n=-48,…,80，129模态。扩大谱只对原33个高斯系数零填充，不重新归一化。q=1为n±8，完整残差为","",
        "$$R_n=i\\partial_t\\widetilde\\psi_n-(k_n\\sigma_x+5\\sigma_z)\\widetilde\\psi_n"
        "-0.1(\\widetilde\\psi_{n-8}+\\widetilde\\psi_{n+8}).$$","",
        "沿用R9预先验证的首个合格截断，R11正式前重验[-48,80]/[-64,96]。复轨迹差="+
        fmt(reference["revalidation"]["full_complex_zero_padded_relative_difference"])+
        "，下/上8层最大占据="+fmt(reference["revalidation"]["lower_eight_layers_max_total_occupation"])+"/"+
        fmt(reference["revalidation"]["upper_eight_layers_max_total_occupation"])+
        "。相对R9原轨迹差="+fmt(reference["relative_difference_from_r9"])+
        "。Hermiticity、范数、可逆性及独立expm仍通过；没有根据PINN调整截断或势强。","",
        "R9忽略外势自由解的spinor/current约0.300671/0.209617，远高于参考截断误差；本轮外势影响可测。预测边界占据及外残差是预测诊断，不重新选择参考。","",
        link("参考核查和轨迹",RAW/"reference_validation.json")+" · "+
        link("201点参考",RAW/"reference_test_grid.npz")+" · "+link("4001点参考",RAW/"reference_fine_grid.npz"),"",
        "## 4. 流程、理论和公平边界","",
        "$$p=e^{i\\tau q_\\theta},\\quad d=c_0+\\tau z,\\quad c=pd,"
        "\\quad\\dot c=\\dot p\\,d+p(z/T+\\tau\\dot z).$$","",
        "$$F_n=[u_+p_+,u_-p_-],\\quad"
        "R_n=iF_n\\dot d_n+(i\\dot F_n-H_nF_n)d_n-0.1(F_{n-8}d_{n-8}+F_{n+8}d_{n+8}).$$","",
        "A辅助单位自由IVP与R9原流程相同；冻结只停止参数更新，完整时间导数和缺陷仍进入loss。B载波与A预训练前逐位相同；C载波和包络同时求完整参数梯度。D双独立4×64四实输出网络按和/√2；E自由相位由已知算符解析计算，独立模块明示先验。四层64宽tanh、双精度、Adam lr=.001、每步32新时间×全部129模态和未加权完整PDE保持一致。","",
        "时间导数是同一Linear/tanh前向的精确链式实现，已逐参数对照自动微分；无时间差分近似。矩阵参考不进入任何组训练。A–D无显式精确频率；E是用户指定的解析例外。E检查点不存矩阵未来轨迹或频率数组。",""]
    design=[]
    for g in GROUPS:
        pre,coupled=budget(g); row=costs[costs.model==g].iloc[0]
        design.append(dict(Group=g,Pre=pre,Coupled=coupled,Total=pre+coupled,
            Active=int(row.active_parameters),Frozen=int(row.frozen_parameters),Parameters=int(row.total_parameters),
            SpinorPoints=int(row.training_spinor_residual_points)))
    lines += ["",table(pd.DataFrame(design),["Group","Pre","Coupled","Total","Active","Frozen","Parameters","SpinorPoints"]),"",
        "A自由阶段活动12802、闲置包络12932；耦合阶段活动12932、冻结12802。C活动25734；D活动25864；E活动12932但含解析信息，不能按可学习参数数目当作无先验公平基线。总步数相同不等于计算成本相同。","",
        "初始化流：carrier900000+seed、第一复网络800000+seed、D第二网络830000+seed。采样流：自由910000+seed、耦合920000+seed。A/B/C初始载波逐元素相同；各组同形第一复网络相同，载波两输出不与四输出网络声明配对。所有组耦合20000步公共前缀相同，B/C/D全部26000步相同；独立评价不改变训练随机状态。","",
        "五worker实际核mask1/2/4/8/16与控制核128互斥，单线程，按seed串行并固定拉丁轮转组顺序。仍共享缓存、内存、功耗，未隔离系统后台负载；实际耗时不作为隔离硬件算法加速证据。","",
        "冻结manifest SHA-256："+frozen+"。正式前30测试及五组工程smoke通过；只验证维度、初值、梯度、NaN与落盘，不按预测精度调参。","",
        " · ".join([link("预注册",DOCS/"preregistration.md"),link("机器协议",RAW/"protocol.json"),
            link("理论核查",DOCS/"theory_check.md"),link("冻结清单",RAW/"frozen_protocol_and_code.json"),
            link("CPU执行计划",RAW/"runtime_plan.json")]),"",
        "## 5. 评价规则和全部种子","",
        "统一Nx=1024、Nt=201，端点不重复；Fourier双线性误差平方最高指数256<1024，Parseval及2048点验证。只消去一个全时空常数相位，不逐模态或逐时间对齐。活跃mask只按参考支占据≥10⁻⁶，相干塌缩预测仍计入；相位定义覆盖率独立报告。初始支持外仅指n<0或n>32。","",
        "branch中心/宽度由圆周矩及展开定义，沿用R9并对同一耦合参考计算；低预测支范数下轨迹只作失败诊断。复系数、密度、流、fidelity、总及±支模态占据、相干和轨迹均保存。",""]
    for endpoint in ("final","coupled_20000"):
        rows=[]
        for record in frame[frame.endpoint==endpoint].to_dict("records"):
            passed=all(record[k]>=v if k=="fidelity_mean" else record[k]<=v for k,v in THRESHOLDS.items())
            rows.append(dict(Group=record["model"].split("_")[0],Seed=record["seed"],Total=record["steps"],
                Coupled=record["coupled_steps"],Spinor=record["spinor_rel_l2_gauge_aligned"],
                Current=record["current_rel_l2"],Fidelity=record["fidelity_mean"],Modal=record["modal_population_rel_l2"],
                Norm=record["max_norm_drift"],Joint="PASS" if passed else "FAIL"))
        lines += ["","### "+endpoint,"",table(pd.DataFrame(rows),
            ["Group","Seed","Total","Coupled","Spinor","Current","Fidelity","Modal","Norm","Joint"])]
    lines += ["","A/E的final与coupled_20000引用同一实际快照，不能视作两次独立重复。B/C/D的20000/26000来自同一次轨迹。","",
        "## 6. 实际配对优势、冻结与容量","",
        "差为target−baseline，误差负差有利、fidelity正差有利；比值均值与均值比均列，无事后倍数门槛或显著性筛选。","",
        table(comparisons[comparisons.metric.isin(["spinor_rel_l2_gauge_aligned","current_rel_l2"])],
            ["endpoint","target","baseline","metric","paired_difference_mean",
             "paired_difference_population_sd","ratio_of_means","mean_of_paired_ratios","favorable_pair_count"]),"",
        "A/B同耦合步数控制冻结策略、初始化及时间序列，检验预训练获得的载波权重；主终点还混合6000步预算分配。A/C不是冻结单因素消融。B/C才从相同随机载波出发改变是否更新，但也改变活动参数。A/D控制总参数，未控制表示及辅助工作量。E/A同耦合预算但解析先验不同，效果只作诊断。","",
        "载波的单位自由IVP误差不是完整场误差：C并未被要求在全部空模态上解单位自由任务，且包络可补偿载波。不能用其全谱单位缺陷单独判定联合训练失败，必须以完整动力学为准。联合训练还允许载波与包络之间再分配相位：p→p exp(if)、d→d exp(−if)，f(k,0)=0时同初态和完整场不变。因此C的好场误差也不能证明其内部载波唯一恢复了自由色散。","",
        link("逐seed所有配对差",RAW/"paired_comparisons.csv")+" · "+
        link("比较汇总",RAW/"comparison_summary.csv"),"",
        "### 同一轨迹的额外6000步预算作用","",
        "B/C/D在20000→26000耦合步数之间保持同一训练祖先、损失及优化器状态，以下为真实配对变化；不能把这三个流程的变化当作A额外预算的反事实。A/E两个终点标签指同一快照，无新增步数。","",
        table(pd.read_csv(RAW/"budget_change_summary.csv",float_precision="round_trip").query(
            "metric in ['spinor_rel_l2_gauge_aligned','current_rel_l2','max_norm_drift']"),
            ["model","metric","paired_difference_mean","paired_difference_population_sd",
             "ratio_of_means","mean_of_paired_ratios","favorable_pair_count"]),"",
        link("逐seed预算变化",RAW/"budget_changes.csv")+" · "+
        link("预算变化汇总",RAW/"budget_change_summary.csv"),"",
        "## 7. 辅助物理与残差诊断",""]
    aux=frame[frame.endpoint=="final"].copy()
    lines += ["",table(aux,["model","seed","density_rel_l2","branch_population_rel_l2",
        "branch_complex_coefficient_rel_l2","active_coherence_rel_l2",
        "plus_center_rmse","minus_center_rmse","plus_width_relative_l2","minus_width_relative_l2",
        "phase_defined_reference_active_coverage"]),"",
        table(aux,["model","seed","plus_min_predicted_norm","minus_min_predicted_norm",
            "predicted_max_occupation_outside_initial_support","lower_boundary_max_occupation",
            "upper_boundary_max_occupation"]),"",
        table(aux,["model","seed","carrier_free_pde_residual_rms","weighted_carrier_defect_spinor_rms",
        "residual_integral_fine","full_physical_residual_integral","exterior_residual_integral",
        "fine_grid_max_norm_drift","norm_balance_integration_max_abs_discrepancy"]),"",
        "D无载波的对应量为N/A。保存129截断内和两端各8个外模态共145模式的完整复残差，外部R=−0.1边界源项。未加权分模态功率、4001/2001积分差、raw误差和残差积分估计均可复查。数值积分不是严格连续误差认证，也不认证无限谱。","",
        "$$\\dot N=2\\operatorname{Im}\\langle\\psi,R\\rangle,\\qquad"
        "\\|e(t)\\|\\le\\|e(0)\\|+\\int_0^t\\|R(s)\\|\\,ds.$$","",
        "单位模载波使F†G为Hermitian，载波缺陷与Hermitian势耦合的直接全局范数投影为零；载波仍会间接影响包络优化。不能把范数漂移唯一归因载波，也不能把范数良好当作相干正确。积分界对应有限Hermitian系统和raw误差；常数规范对齐误差不替代它。","",
        "## 8. 全阶段成本","",
        "以下均为每run真实观察值，秒。训练优化墙钟含全部阶段，排除独立评价与主要落盘；会话elapsed还包含日志、监测及其余开销。主要落盘I/O为独立计时，未覆盖的小量初始化/根目录复制开销包含在会话elapsed或执行事件中，不声称理想单线程benchmark。发生恢复时elapsed仅对应本次会话，累计阶段时间在恢复状态内。RSS是20ms native实际进程采样峰值，含持久seed worker的分配器缓存。","",
        table(costs,["model","seed","carrier_training_seconds","coupled_training_seconds","training_seconds",
            "evaluation_seconds","checkpoint_and_artifact_io_seconds","elapsed_seconds_this_session",
            "peak_rss_mib","training_spinor_residual_points","evaluation_coupled_spinor_residual_points",
            "evaluation_carrier_unit_spinor_residual_points"]),"",
        "115个预定误差—成本点全部保留；A的累计残差工作量按两个单位辅助IVP计数，不能只按26000优化步当作与B/C/D同成本。E的解析先验成本结构独立。成本图同时以实际训练墙钟和旋量残差点为横轴，评价/落盘另列。","",
        link("成本和全部残差工作量",RAW/"cost_and_residual_work.csv")+" · "+
        link("全部115成本曲线点",RAW/"error_cost_curves.csv"),"",
        "## 9. 图表与source data",""]
    cost_summary=[]
    for group in GROUPS:
        part=costs[costs.model==group]
        cost_summary.append(dict(Group=group.split("_")[0],
            PreSeconds=f"{part.carrier_training_seconds.mean():.8g} ± {part.carrier_training_seconds.std(ddof=0):.8g}",
            CoupledSeconds=f"{part.coupled_training_seconds.mean():.8g} ± {part.coupled_training_seconds.std(ddof=0):.8g}",
            TrainSeconds=f"{part.training_seconds.mean():.8g} ± {part.training_seconds.std(ddof=0):.8g}",
            EvalSeconds=f"{part.evaluation_seconds.mean():.8g} ± {part.evaluation_seconds.std(ddof=0):.8g}",
            PeakRSSMiB=f"{part.peak_rss_mib.mean():.8g} ± {part.peak_rss_mib.std(ddof=0):.8g}"))
    # Keep the group cost summary with the cost evidence before the figure section.
    lines[-2:-2]=["各组观察成本的均值±population SD（n=5）：","",
        table(pd.DataFrame(cost_summary),["Group","PreSeconds","CoupledSeconds","TrainSeconds","EvalSeconds","PeakRSSMiB"]),""]
    qa=json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    for item in qa["figures"]:
        name=item["figure"]
        lines += ["","### "+name,"",item["claim"],"",
            f"![{name}](<{(FIGURES/(name+'.png')).resolve().as_posix()}>)","",
            " · ".join(link(s.upper(),FIGURES/f"{name}.{s}") for s in ("png","svg","pdf")),"",
            " · ".join(link("Source CSV",FIGURES/p) for p in item["source_csv"])]
    lines += ["",link("FIGURE_QA",FIGURES/"FIGURE_QA.md"),"",
        "## 10. 产物、测试与审计","",
        "25个正式run全部保留最终六件套、Adam/RNG恢复、初始权重、事件和成本曲线；40个完整评价快照各有六件套与恢复状态，75个中间物理评价快照保存权重/密网格及指标。没有最佳检查点、失败seed替换或追加补救模型。",""]
    for label,name in (("完整pytest","pytest_final_result.json"),("最终审计","audit.json"),("链接检查","link_check.json")):
        path=RAW/name
        if path.exists():
            result=json.loads(path.read_text(encoding="utf-8"))
            lines += [f"{label}：{result.get('status','未完成')}；详见 "+link(name,path)+"。",""]
    for g in GROUPS:
        for seed in SEEDS:
            directory=run_directory(g,seed)
            lines += ["",f"- {g}/seed{seed}: "+" · ".join(link(p,directory/p)
                for p in ARTIFACTS+("recovery_state.pt","execution_cost.json"))]
    lines += [""," · ".join(link(label,RAW/path) for label,path in (
        ("逐seed完整CSV","per_seed_metrics.csv"),("全部指标统计","summary_metrics.csv"),
        ("decision","decision.json"),("数据审计","data_audit.json"),
        ("最终审计","audit.json"),("产物索引","run_artifact_index.csv"),
        ("固定快照索引","snapshot_index.csv"),("历史SHA保护","protected_prior_results.json"),
        ("产物SHA","artifact_sha256.json"),("完整pytest","pytest_final_result.json"),
        ("链接检查","link_check.json"))),"",
        "## 11. 结论范围","",
        "本轮严格区分完整动力学成功、容量影响、辅助预训练、冻结/联合流程差异、预算和优化失败。参考已独立收敛；某模型失败不能直接解释成谱截断失败或数学不可表示。五新种子的可重复性只在这套已知问题、实现和固定预算中成立，不是一般优化成功保证。","",
        "即使通过，也只证明已知有限Fourier高斯初谱在该弱周期外势下有效，不能宣称未知连续谱发现、一般外势或通用Dirac求解器。E仅诊断解析自由坐标的影响，不能将其收益归入可学习方案。本轮不追加补救结构；后续物理稳健性应另行预注册。",""]
    text="\n".join(lines)
    (ROOT/"R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md").write_text(text,encoding="utf-8")
    (DOCS/"report.md").write_text(text,encoding="utf-8")
    return len(text)

if __name__=="__main__":
    print("Report characters: "+str(build()))
