from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from .r8_config import ROOT, RAW, DOCS, FIGURES, LOGS, THRESHOLDS
from .r8_decision import read_stage, joint_mask
from .freeze_r8_coupling_recovery import verify_freeze, verify_history
from .freeze_r7_phase_release import write_json, sha256
from .audit_r8_coupling_recovery import expected_runs
from .plot_r8_coupling_recovery import LABELS


def link(path, label=None):
    path = Path(path)
    return f"[{label or path.name}](<{path.resolve().as_posix()}>)"


def number(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.6g}"


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def summary(frame, metrics):
    rows = []
    for (case, name, balanced), group in frame.groupby(["case", "model", "balanced"], sort=False):
        rows.append([case, LABELS[name]+(" ·残差平衡" if balanced else ""), len(group)]+
            [number(group[m].mean())+" ± "+number(group[m].std(ddof=0)) for m in metrics]+
            [f"{int(joint_mask(group).sum())}/{len(group)}"])
    return table(["算例", "配置", "n"]+list(metrics)+["五项联合合格数"], rows)


def per_seed(frame, metrics):
    return table(["算例", "配置", "seed"]+list(metrics),
        [[r.case, LABELS[r.model]+(" ·平衡" if r.balanced else ""), int(r.seed)]+[number(r[m]) for m in metrics]
         for _, r in frame.sort_values(["case", "model", "balanced", "seed"]).iterrows()])


def tests_note():
    path = LOGS/"final_pytest.xml"
    if not path.exists():
        path = LOGS/"prefreeze_full_pytest.xml"
    node = ET.parse(path).getroot()
    suites = [node] if node.tag == "testsuite" else list(node.findall("testsuite"))
    count = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped"))
    return f"全项目旧/新pytest：{count}项，失败/错误/跳过合计{failures}。"+link(path, "测试记录")


def build_report():
    exploration, validation = read_stage("exploration"), read_stage("validation")
    if len(exploration) != 18 or len(validation) != 20:
        raise ValueError("All 18 exploratory and 20 independent validation runs required")
    all_runs = pd.concat([read_stage(s) for s in ("smoke", "exploration", "validation", "ablation", "extension", "rescue")], ignore_index=True)
    if len(all_runs) != len(expected_runs()):
        raise ValueError("Every preregistered conditional run must finish before final report")
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
    reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    frozen, history = verify_freeze(), verify_history()
    audit = json.loads((RAW/"audit.json").read_text(encoding="utf-8")) if (RAW/"audit.json").exists() else {"status": "PENDING"}
    runtime = json.loads((RAW/"runtime_affinity_audit.json").read_text(encoding="utf-8"))
    ablation, extension, rescue = read_stage("ablation"), read_stage("extension"), read_stage("rescue")
    successes = decision["successful_candidates"]
    primary = list(THRESHOLDS)
    mechanism = json.loads((RAW/"mechanism_summary.json").read_text(encoding="utf-8"))
    interpretation = []
    for item in mechanism["models"]:
        if item["model"] not in selection["selected_candidates"]:
            continue
        group = validation[validation.model==item["model"]]
        interpretation.append(LABELS[item["model"]]+"：五新seed的spinor/current/fidelity均值="+
            "/".join(number(group[key].mean()) for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"))+
            "；联合合格"+str(item["joint_absolute_pass_count"])+"/5；第一侧带复系数误差="+
            number(item["mean_first_sideband_complex_error"])+"，最大中心外占据均值="+
            number(item["mean_max_outside_center_occupation"])+"（reference="+
            number(item["reference_max_outside_center_occupation"])+"）。"+
            ("通过的是本轮完整模型及训练流程，不等于原R6/R7极坐标模型已自动修复。" if item["independently_validated_absolute_success"] else
             "该配置未达联合绝对条件，局部梯度或个别改善不能改称完整成功。"))
    if len(ablation):
        mapping = {"A_identity_carrier": "A_carrier_complex", "B_static_free": "B_free_correction",
            "window_complex": "C_release_windows", "D_identity_carrier": "D_carrier_windows",
            "A_carrier_complex": "D_carrier_windows"}
        for name, group in ablation.groupby("model"):
            target = validation[validation.model==mapping[name]]
            interpretation.append("机制配对："+LABELS[mapping[name]]+"相对"+LABELS[name]+"的平均spinor/current误差比="+
                number(target.spinor_rel_l2_gauge_aligned.mean()/group.spinor_rel_l2_gauge_aligned.mean())+"/"+
                number(target.current_rel_l2.mean()/group.current_rel_l2.mean())+
                "；该消融联合合格"+str(int(joint_mask(group).sum()))+"/5。数值贡献以完整配对差与SD判断，不设事后倍数成功门槛。")
    not_validated = [name for name in ("A_carrier_complex", "B_free_correction", "C_release_windows", "D_carrier_windows") if name not in selection["selected_candidates"]]
    interpretation.append("未进入五seed验证的"+"、".join(LABELS[name] for name in not_validated)+
        "只完成三seed探索；不能把探索失败提升为其在20,000步预算下一定失败，也不能据此排除所有窗口协同可能。")
    sections = ["# R8：弱周期外势耦合动力学恢复实验报告", "",
        "实验ID：r8_coupling_recovery。执行始于2026-10-01；具体事件时间以各run的execution_events.jsonl为准。", ""]

    def section(title, *content):
        sections.extend(["## "+title, ""]+[item for value in content for item in (value, "")])

    result = "、".join(LABELS[n] for n in successes)+"通过五新种子的完整动力学绝对判据。" if successes else "两个预定候选均未通过五新种子的完整动力学绝对判据。"
    section("1. 结论先读", result, "自动分类：**"+decision["classification"]+"**。工程完整性不替代科学成功；全部探索、验证与条件run保留，没有最佳seed或最佳checkpoint替换固定终点。",
        "本轮论证：在已知有限自由Dirac基底及弱周期动量耦合中，检验学习自由载波、复包络、自由解加修正与时间窗是否恢复真实侧带；结论由冻结预算下全种子旋量、概率流、占据、范数和fidelity共同决定。")
    if "A_carrier_complex" in successes:
        main_a = validation[validation.model == "A_carrier_complex"]
        section("1.1 关键数值与成功边界",
            "A五个新seed平均spinor/current/fidelity="+
            "/".join(number(main_a[key].mean()) for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"))+
            "。seed100最大范数漂移="+number(main_a[main_a.seed==100].max_norm_drift.iloc[0])+
            "，超过.02，故不是5/5全成功；仍按冻结的均值及4/5联合规则通过。动力学图固定使用这个未联合通过的seed100，不换最好seed。",
            "通过的是可学习载波＋复包络方案A，不是原始全实振幅LTPB或R7相位释放模型本身恢复了耦合。")
    section("2. 历史证据与研究问题",
        "R1公共单相位具有一般数学表达能力，但混合支载波预条件优势退化；总范数/能量/连续性无法充分识别相位错误。R2–R5在已知自由能支与Fourier表示下有效，其中共享高斯谱五seed spinor≈0.04104、current≈0.04806，不等于动态空通道也容易学习。",
        "R6原LTPB约0.36670/0.45612；R7空支相位释放普通组仍约0.36783/0.45628。R7证明局部虚部切向可优化，但侧带几乎未生成。两共享谱模型均失败，不能单独归因于极坐标数学不可表示。",
        link(ROOT/"docs/comprehensive/report.md", "综合报告")+"；"+link(ROOT/"docs/experiments/periodic_potential_coupled_ltpb/report.md", "R6")+"；"+link(ROOT/"R7_PHASE_RELEASE_EXPERIMENT_REPORT.md", "R7")+"。旧文件受SHA-256保护。")
    section("3. 方程、初态、参考与隔离",
        r"$$i\partial_t\psi=[-i\sigma_x\partial_x+m\sigma_z+V_0\cos(qx)I_2]\psi,$$",
        r"$$R_j=i\partial_t\widehat{\widetilde\psi}_j-H(k_j)\widehat{\widetilde\psi}_j-\frac{V_0}{2}(\widehat{\widetilde\psi}_{j-1}+\widehat{\widetilde\psi}_{j+1}),\quad H(k)=k\sigma_x+m\sigma_z.$$",
        "L=16π,m=5,k0=2,q=1,V0=.2,T=5,J=6；13个动量−4到8。初态仅中心为(u+ + u−)/√2。Fourier展开带1/√L。自由基底并非外势完整Hamiltonian本征态。",
        "参考由块三对角Hermitian矩阵本征演化，独立矩阵指数核验。主J6/J8继承授权条件；所有扩展截断先于训练、按固定J梯级选择，不依PINN精度择优。",
        table(["算例", "训练J", "比较J", "零填充复轨迹相对差", "外边缘占据", "参考范数漂移"],
            [[case, info["selected_half_width"], info["selected_half_width"]+2,
              number(info["trials"][-1]["comparison"]["padded_full_trajectory_spinor_relative_l2"]),
              number(info["trials"][-1]["observables"]["max_outer_layer_total_occupation"]),
              number(info["trials"][-1]["checks"]["norm_max_drift"])]
             for case, info in reference["cases"].items() if case != "free_packet"]),
        "自由回归保持R5的33个k=n/8、k0=2、σk=.5、两支等权高斯谱；更宽矩阵只对同一33初谱零填充，不改变归一化。"+link(RAW/"reference_validation.json", "完整参考核验"),
        "精确频率和未来参考轨迹不进入learned forward/loss/optimizer/config/checkpoint。自由载波由单位初态原旋量PDE学得，频率只在独立评价计算。H(k)、u±、已知动量、初态与预测交接均为先验，不等于无先验发现色散。")
    section("4. 模型与理论核查",
        "载波p=exp[i(t/T)qθ(k,t/T)]，同一4×64 tanh网络输出两实相位，p(0)=1、|p|=1。单位模自由演化约束防止预训练零解塌缩，也必须计入方法物理先验。",
        "每个k、每个正/负本征旋量分别作单位非零初态，预训练i∂t(pu)−H(k)(pu)。不改写成含E的对角方程。冻结仅停止权重更新，保留p的时间图。",
        "这些是已知自由算符的辅助单位初态IVP，不是把目标耦合初态的空通道改成非零，也不是未来目标标签；主初态始终仅占据中心。辅助任务对每个自由支提供强物理预训练，需作为额外先验与训练成本明示。数学上，在固定自由本征支中该PDE约束当然包含其色散信息；‘没有E输入’不等于算符本身没有频率信息。",
        r"单位模$p=e^{iS}$使自由残差$R_{free}=p[-\dot S\,u-H(k)u]$，故$\|R_{free}\|^2=\|\dot S\,u+H(k)u\|^2$。这在辅助任务中去除了快速指数对残差模长的影响，把自由载波学习转为相位导数约束；不等于MLP参数优化已成为凸问题。实现仍计算原旋量PDE，而非向模型输入解析频率。",
        r"$$c_A=p[c_0+\tau(a+ib)],\quad\widehat{\widetilde\psi}_B=\widetilde\psi_{free}^{learned}+V_0t\chi,$$",
        r"$$c_A'=p'[c_0+\tau z]+p[z/T+\tau z'],\quad z=a+ib.$$",
        r"令$F_j=[u_+(k_j)p_{j,+},u_-(k_j)p_{j,-}]$、$d_j=c_{j,0}+\tau z_j$，则原旋量残差恒等于",
        r"$$R_j=iF_j\dot d_j+(i\dot F_j-H_jF_j)d_j-\frac{V_0}{2}(F_{j-1}d_{j-1}+F_{j+1}d_{j+1}).$$",
        "这解释近似自由载波预条件：若自由缺陷小，包络主要承担耦合；缺陷不为零时仍完整计入。实现直接计算原旋量式，不用含精确能量的对角训练方程。单位模载波与正交自由基底形成逐模态幺正坐标变换，不暗中按模态重加权主残差。",
        "完整残差保留载波缺陷；A空支导数z(0)/T、B为V0χ(0)，虚方向可用。B的χ是原始双分量复旋量，主问题V0T=1使V0t=τ，所以本轮不能把该因素解释为新的数值缩放收益。",
        "C窗内已占据复初态用(|c_s|+τwr)exp[i(arg c_s+τwq)]，严格空初态用τwr exp(iq)。D窗内用p(t)/p(s)与c_s+τwz。前窗预测末态投影、detach后作为下一窗初值，不用参考重置、不归一化、不按阈值切换。",
        "连续分片预测的导数可以跳变，连续场不产生delta项。保存交接误差。可用虚切向不等于耦合已学准；相位只在参考活跃区域具有物理定义。",
        link(DOCS/"theory_check.md", "理论核查")+"；"+link(RAW/"theory_validation.json", "局部梯度验证")+"；"+link(ROOT/"src/r8_models.py", "模型")+"；"+link(ROOT/"src/r8_loss.py", "完整残差"))
    section("5. 公平协议、成本与冻结",
        "探索六配置×seeds0–2，10000总步；A/B/D=3000自由+7000耦合，C每窗1000、D每窗700。最多两候选独立验证加两个重训控制，新seeds100–104、20000总步；载波6000+14000，C每窗2000、D每窗1400。全部双精度、Adam1e−3，不早停、不调预算、不选择最好检查点。",
        "每步32新排序均匀时间×全部k，载波涉及两个独立单位分支IVP。初始化与时间流独立；四实输出MLP同seed逐元素相同，窗口共享归一随机数而映射不同物理时间。不同载波预算共享耦合公共前缀，不能宣称所有配置全部物理点一致。",
        "单窗12932、载波12802；A/B共25734，十窗C129320、D142122。当前优化参数与全部推理存储参数分列。时间窗收益混合短区间、顺序训练、优化器重置与更大分片容量，不能完全归为同容量训练策略。",
        "训练时间是全部优化步骤的实际墙钟累计，包含预训练、窗口交接和更新，不含检查点落盘与后评价；评价时间另列。run_elapsed_seconds_this_session另存当前进程会话时间，发生恢复时不冒充整个实验总耗时。peak RSS是训练监测区间进程常驻内存峰值，单位MiB。步骤、旋量残差点和复标量分量均计数。",
        "训练残差次数由总步数、虚拟分支数和RNG记录复核；后评价另分完整耦合残差、载波单位残差及仅导数诊断。metrics中继承的evaluation_coupled_spinor_points为旧估算口径，不冒充全部实际残差调用计数；自由回归跳过不适用导数后的实际工作量明确分列。"+
        link(RAW/"residual_evaluation_work.csv", "逐run残差评价工作量"),
        "资源执行偏差：原计划三独立核，但冻结的调度公式将各任务seed与slot相加，跨seed批次可能落在同一核。"+
        str(runtime["non_smoke_runs_in_shared_core_cohorts"])+"个非smoke运行位于有核竞争的批次。没有修改冻结代码或选择性重跑；样本、总步数和固定终点更新未变。耗时为真实观察到的运行成本，但不是公平的算法速度比较，不能据此声称加速。"+
        link(RAW/"runtime_affinity_audit.json", "偏差明细")+"；"+link(RAW/"cost_runtime_context.csv", "逐run资源上下文"),
        "冻结manifest SHA-256："+frozen+"。"+link(RAW/"frozen_protocol_and_code.json", "逐文件清单")+"；"+link(DOCS/"preregistration.md", "预注册")+"；"+link(DOCS/"figure_contract.md", "图表契约"))
    section("5.1 冻结后评价兼容修复的明确披露",
        "自由高斯回归A/seed200已完成20000步并保存最终权重，但旧评价器将只适用于中心空侧带的导数误差除以零参考。原R8代码虽计划在free_packet中删除这项诊断，删除时点晚于报错。",
        "新增评价适配器仅对V0=0的预注册free_packet扩展提前跳过该不适用诊断。所有主指标、完整残差、参考、配置、预算、优化器和冻结科学源文件保持原样。其余自由回归统一使用适配器，不能称为冻结后完全没有工程变更。",
        "seed200从已经完成的终点恢复评价，新增优化步为0；原报错日志与首次最终权重保留。独立审计核验恢复前后所有模型张量SHA-256、总步数和训练耗时不变。每个自由回归保存evaluation_adapter_record.json。"+
        link(DOCS/"evaluation_adapter_note.md", "修复边界与证据")+"；"+link(ROOT/"src/r8_free_packet_evaluation_adapter.py", "独立适配器"))
    section("6. 探索阶段与预定筛选",
        "Q=mean_seed[(spinor/.10 + current/.15 + 第一侧带复系数误差/.10)/3]，从A–D选最低两项。侧带只沿用主常数规范，不独立对齐。Q低不代表成功，不以单个seed或训练loss筛选。",
        table(["候选", "Q", "探索联合合格", "进入验证"],
            [[LABELS[r["model"]], number(r["score"]), str(r["exploration_absolute"]["joint_pass_count"])+"/3",
              "是" if r["model"] in selection["selected_candidates"] else "否"] for r in selection["ranking"]]),
        summary(exploration, primary), per_seed(exploration, primary+["sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2"]),
        link(RAW/"exploration_selection.json", "冻结公式的筛选记录")+"；"+link(RAW/"exploration_runs.csv", "全部18次原生指标"))
    section("7. 独立验证与绝对成功判定",
        "五seed均值spinor≤.10,current≤.15,fidelity≥.98,modal-population≤.10,norm-drift≤.02；至少4/5 seeds联合满足五项。相对失败控制更好、单项守恒良好或探索表现好，均不能代替。",
        summary(validation, primary), per_seed(validation, primary),
        table(["配置", "绝对成功", "逐seed联合通过"],
            [[LABELS[n], "PASS" if info["pass_check"] else "FAIL", str(info["joint_pass_count"])+"/5"] for n, info in decision["groups"].items()]),
        "完整辅助精度：", per_seed(validation, ["spinor_rel_l2_raw", "final_spinor_rel_l2_gauge_aligned", "density_rel_l2", "fidelity_min", "branch_population_rel_l2", "sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2"]),
        "侧带、局部与残差诊断：", per_seed(validation, ["lower_sideband_occupation_rel_l2", "upper_sideband_occupation_rel_l2", "max_outside_center_occupation", "initial_sideband_time_derivative_rel_l2", "initial_sideband_imaginary_tangent_parameter_sensitivity", "fine_unweighted_pde_loss", "residual_integral_fine"]),
        "只消去一个全时空常数相位，raw误差另存。参考占据≥10⁻⁶评价相位/相干，预测相位另需≥10⁻¹²。塌缩预测不从相干误差删除；有效mask保存在dense_grid。")
    comparisons = pd.read_csv(RAW/"comparison_summary.csv")
    quadrature = pd.read_csv(RAW/"physical_quadrature_check.csv")
    quadrature_formal = quadrature[quadrature.stage != "smoke"]
    free = json.loads((RAW/"diagnostic_uncoupled_free.json").read_text(encoding="utf-8"))
    free_distances = pd.read_csv(RAW/"validation_distance_to_free.csv")
    section("8. 侧带、配对贡献与成本",
        "参考最大中心外占据0.3209004147。真正生成需联合核对中心向侧带转移、复系数与相干，不把小非零噪声当成成功。",
        "明确标记的忽略外势自由oracle只作训练后诊断，不进入learned排名。它相对耦合参考的spinor/current/fidelity="+
        "/".join(number(free[key]) for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"))+
        "，侧带占据为零，完整耦合残差范数均值="+number(free["coupled_residual_l2_mean"])+"。"+link(RAW/"diagnostic_uncoupled_free.json"),
        table(["配置", "n", "到自由oracle的spinor距离 ± SD", "到耦合参考的spinor误差 ± SD"],
            [[LABELS[name], len(group), number(group.spinor_distance_to_uncoupled_free.mean())+" ± "+number(group.spinor_distance_to_uncoupled_free.std(ddof=0)),
              number(group.spinor_distance_to_coupled_reference.mean())+" ± "+number(group.spinor_distance_to_coupled_reference.std(ddof=0))]
             for name, group in free_distances.groupby("model")]),
        summary(validation, ["sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2", "lower_sideband_occupation_rel_l2", "upper_sideband_occupation_rel_l2", "max_outside_center_occupation"]),
        table(["比较", "n", "spinor差 ± SD", "current差 ± SD", "侧带复系数差 ± SD", "配对spinor比均值", "配对current比均值", "两项联合胜出"],
            [[r.comparison, int(r.n_seeds), number(r.spinor_rel_l2_gauge_aligned_difference_mean)+" ± "+number(r.spinor_rel_l2_gauge_aligned_difference_population_sd),
              number(r.current_rel_l2_difference_mean)+" ± "+number(r.current_rel_l2_difference_population_sd),
              number(r.sideband_complex_coefficient_rel_l2_difference_mean)+" ± "+number(r.sideband_complex_coefficient_rel_l2_difference_population_sd),
              number(r.spinor_rel_l2_gauge_aligned_ratio_mean), number(r.current_rel_l2_ratio_mean),
              f"{int(r.spinor_current_joint_wins)}/{int(r.n_seeds)}"] for _, r in comparisons.iterrows()]),
        "差为target−baseline，误差负差有利。所有比值/逐seed差见"+link(RAW/"paired_comparisons.csv")+"及"+link(RAW/"comparison_summary.csv")+"。population SD(ddof=0)不是SEM或置信区间，不作小样本普遍显著性声明。",
        "固定Nx=256评价口径未变。另以有限谱双线性量的完整Fourier系数复算空间L2，检查扩展J8时最高双线性频率到达Nyquist的数值影响；这只作积分诊断，不替换预定成功指标。全部非smoke run密度/流误差最大绝对差分别为"+
        number(quadrature_formal.density_absolute_metric_difference.max())+"/"+number(quadrature_formal.current_absolute_metric_difference.max())+"。"+link(RAW/"physical_quadrature_check.csv"),
        per_seed(validation, ["parameter_count", "active_trainable_parameter_count", "carrier_steps", "coupled_steps", "residual_training_batches", "residual_training_spinor_points", "training_seconds", "peak_rss_mb", "evaluation_seconds"]))
    carrier = validation[validation.carrier_steps > 0]
    carrier_summary = summary(carrier, ["carrier_free_pde_residual_rms", "carrier_phase_slope_mean_relative_error", "carrier_phase_slope_max_relative_error"]) if len(carrier) else "正式候选未用载波，斜率不适用。"
    section("9. 载波缺陷与未加权残差积分",
        "载波p仍可能不准；后评价取实际p′相位斜率和原旋量自由PDE残差，不以内部q直接代替物理解。复包络可修正载波缺陷，不能称learned p为精确oracle。", carrier_summary,
        "载波斜率属于模恒为1的辅助自由IVP，在所有k上有定义；它不是目标耦合初态中空通道的物理相位。目标旋量的物理相位与相干指标只使用总预测场的自由能支投影及保存的参考占据有效mask。",
        r"$$\|e(t)\|_2\le\|e(0)\|_2+\int_0^t\|R(s)\|_2ds.$$",
        "raw场和未加权全部模态残差用于Duhamel数值估计，完整物理残差另计边缘外耦合。4001/2001积分差只是数值核验，不是严格连续认证，也不把随机末批loss作为误差证书。",
        "该积分界针对所选有限Hermitian梯级与同一截断参考；外层泄漏单独报告，不能据有限梯级积分界认证无限谱或连续物理系统。",
        per_seed(validation, ["raw_final_spinor_error_norm", "residual_integral_fine", "residual_integral_coarse", "residual_integral_quadrature_relative_difference", "residual_estimate_max_numerical_violation", "exterior_residual_leakage_max_l2_norm"]),
        "逐模态/时段未加权残差、初始导数与输出梯度全部落盘，阶段/窗口切换不删除。")
    conditioning = pd.read_csv(RAW/"window_coordinate_summary.csv")
    section("9.1 时间窗交接的坐标条件诊断",
        r"对非零交接$c_s=\rho e^{i\alpha}$，$\dot c(s)=e^{i\alpha}(r+i\rho q)/\Delta t$。相位输出方向尺度为$\rho/\Delta t$，振幅方向为$1/\Delta t$；Complex两方向则都为$1/\Delta t$。",
        r"由预测交接态的原耦合PDE计算所需导数$v$，对应$q_{required}=\Delta t\operatorname{Im}(e^{-i\alpha}v)/\rho$。当预测振幅很小且源方向不与它共线时，局部相位坐标可要求很大数值。",
        "只在参考占据≥10⁻⁶且预测交接严格非零的通道列出该q约束；它是局部输出/导数坐标诊断，不是零振幅处的物理相位误差，不作训练标签。所需v来自预测态与已知H，而非参考未来轨迹。条件数仅涉及两个局部输出方向，不等于完整MLP Jacobian或Hessian条件数。",
        table(["配置", "seed", "有效第一侧带交接数", "坐标条件比中位数", "最大所需|q|", "最大实际|q|", "平均交接初导数绝对误差"],
            [[LABELS[r.model], int(r.seed), int(r.valid_channel_handoffs), number(r.median_local_velocity_condition),
              number(r.maximum_required_absolute_q), number(r.maximum_actual_absolute_q), number(r.mean_initial_derivative_abs_error)] for _,r in conditioning.iterrows()]),
        "这些数据支持检查近零振幅的优化条件，但不能单独证明C的失败由极坐标造成；还混有短窗预算、顺序误差和容量差异。相同窗口Complex训练对照仅在C正式成功时触发。"+
        link(RAW/"window_coordinate_conditioning.csv", "全部通道及有效性标记")+"；"+link(RAW/"window_coordinate_summary.csv", "逐seed汇总"))
    if len(ablation):
        section("10. 条件最小机制消融",
            "主验证成功后才启动，五配对seed、20000总步。移除p的组仍完整预训练但不在forward使用，保留闲置开销。移除载波也减少有效表示结构，消融不能独自排除所有容量解释。",
            "本轮实际触发的消融："+"、".join(LABELS[name] for name in ablation.model.unique())+
            "。其余成功条件未满足的消融没有执行，未训练不能写成失败或通过。",
            summary(ablation, primary), per_seed(ablation, primary+["sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2"]),
            "配对效应见第8节。Window Complex只在C成功时才触发；D与A同时改变时间范围和分片容量，不能把差异全部归为优化顺序。失败救援未触发。")
    else:
        section("10. 唯一探索性救援",
            "主候选均失败，成功消融与范围扩展不触发。按冻结规则只对最低正式Q候选追加一次EMA模态残差平衡，原探索seeds0–2、10000总步。唯一改变为detach的权重，结构、预训练、采样、Adam均不变；对照复用探索同配置同seed同预算。",
            summary(rescue, primary), per_seed(rescue, primary+["sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2"]),
            "救援即使改善也不是五个全新种子的独立成功。保留未加权残差与全部权重，不追加其他补救；若继续采用需下一轮重新独立验证。")
    if len(extension):
        extension_judgments = json.loads((RAW/"extension_decision.json").read_text(encoding="utf-8"))["results"]
        section("11. 条件扩展与成功范围",
            "成功候选与Complex用新seeds200–202，每次20000总步，完成T8、V0=.1/.4及自由高斯谱回归。n=3范围诊断用同五项均值/≥2/3联合通过，不代替n=5主验证。固定参考J与失败全部保留。",
            "每个扩展算例从头进行固定预算的自由预训练及耦合训练，是同一方法的重新训练范围验证；不是把主问题checkpoint直接零样本外推到新T或新势强。",
            summary(extension, primary), per_seed(extension, primary+["sideband_complex_coefficient_rel_l2", "active_coherence_rel_l2"]),
            table(["算例", "配置", "三seed范围判据", "联合合格数", "均值条件未通过项"],
                [[row["case"], LABELS[row["model"]],
                  "PASS" if row["absolute_scope_diagnostic"]["pass_check"] else "FAIL",
                  str(row["absolute_scope_diagnostic"]["joint_pass_count"])+"/3",
                  "、".join(key for key, passed in row["absolute_scope_diagnostic"]["mean_checks"].items() if not passed) or "无"]
                 for row in extension_judgments]),
            link(RAW/"extension_decision.json", "范围诊断分类")+"。自由回归仅检验已知33初谱；未重复R5的未训练k中点，不据此宣布连续谱泛化。")
        packet = pd.read_csv(RAW/"free_packet_regression_metrics.csv")
        section("11.1 自由波包的分裂传播回归",
            "沿用R5的圆周中心、圆周宽度及全时间线性速度定义，直接从总预测场的自由能支投影重建分支密度，不以内部载波相位替代。该附加物理回归诊断不更改预注册成功门槛。低占据塌缩控制的中心/宽度只作失败诊断，不能解释为正常波包。",
            "自由回归V0=0，33个高斯模态在初态中均已非零。通用原始指标名中的sideband指中心相邻k的系数探针，outside_center是原有多模态占据，均不表示自由系统产生了新的耦合侧带；只有主周期外势实验的初始空侧带可支持动态生成的结论。",
            table(["配置", "seed", "最小分支范数", "+支速度/参考", "−支速度/参考", "+支中心RMSE", "−支中心RMSE", "+支宽度L2", "−支宽度L2"],
                [[LABELS[r.model], int(r.seed), number(r.branch_minimum_norm),
                  number(r.branch_plus_velocity)+"/"+number(r.branch_plus_exact_velocity),
                  number(r.branch_minus_velocity)+"/"+number(r.branch_minus_exact_velocity),
                  number(r.branch_plus_center_rmse), number(r.branch_minus_center_rmse),
                  number(r.branch_plus_width_rel_l2), number(r.branch_minus_width_rel_l2)] for _, r in packet.iterrows()]),
            link(RAW/"free_packet_regression_metrics.csv", "回归附加指标")+"；"+link(RAW/"free_packet_regression_series.csv", "分支轨迹")+"。")
    else:
        section("11. 扩展未触发", "NOT_TRIGGERED：没有候选通过主绝对条件，故不训练T8/外势幅值/自由波包扩展。参考预核验通过不等于learned验证成功；不能把未训练说成扩展失败或通过。")
    section("12. 归因、局限与术语", "\n\n".join(interpretation),
        "表达能力：普通Complex并未被证明数学不可表示。局部相位切向可用与全程求解成功分开，有限预算失败不是表示不可能定理。",
        "载波预条件：A把已知自由能支旋转交给非监督单位PDE预训练p，让复包络处理耦合与载波误差。若A优于B且通过消融，支持对侧带也作预条件，而非只分离中心自由部分；须按实际数据判断，不能先验宣称因果。",
        "A与B同时改变修正项的载波预条件及输出基底（分支系数/原始旋量），不是严格单因素消融；其比较只能提示机制。单因素证据优先来自预定的移除载波或相同窗口Complex组。",
        "本轮最小消融检验的是是否使用已学载波。它没有独立训练‘同一载波＋纯实极坐标包络’或把辅助单位IVP迁给普通Complex的对照，因此不能把全部贡献唯一分配给复包络、辅助任务或函数类容量。支持的是本预算下完整载波预条件流程，以及移除载波后的退化，不是一般必要性定理。",
        "训练/容量：窗口同时缩短时间、顺序传递初态、重置Adam并增加存储参数。D与A不是完全同容量消融，不能把时间窗效果全算在相位释放上。",
        "守恒量：范数或密度正确可伴随错误相干与流；本轮五项联合门槛与复侧带避免此误判。零振幅处相位未定义，不报告无物理含义的普通相位误差。",
        "先验：k梯级、自由u±、H(k)、已知初谱、单位模载波与预测交接均是先验；外势完整Hamiltonian的通道仍耦合。",
        "边界：有限已知谱、特定初态、周期电势与预算；不涵盖未知谱发现、一般外势、非周期散射、场量子化或通用Dirac求解器。扩展失败不事后调参。")
    sections.extend(["外势主验证及T8/V0扩展仍采用只占据中心的平面波初态。高斯波包回归的V0严格为0；两类结果合在一起，也不能推出‘外势作用下的连续高斯波包已求解成功’。该交叉推广需要下一轮独立预注册。", ""])
    sections.extend(["本问题已有可靠的直接矩阵演化解。本轮针对PINN的表示和优化机制，不证明神经方法比矩阵指数/本征分解更快或更适合这一小型线性系统。", ""])
    originality = DOCS/"originality_assessment.md"
    section("13. 完成后原创性评估", originality.read_text(encoding="utf-8") if originality.exists() else
        "文献核验尚待完成，不能据本地结果宣称首次。常见预条件、学习载波、复包络、预训练和时间窗均按已有工具处理。")
    trained = all_runs[all_runs.stage != "smoke"]
    section("14. 工程核验与产物",
        tests_note(), f"工程审计：{audit['status']}。历史保护{history['checked_files']}文件，变化/缺失{len(history['changed_or_missing'])}。科学分类单独判断。",
        f"非smoke训练{len(trained)} runs，总优化{int(trained.steps.sum())}步（自由预训练{int(trained.carrier_steps.sum())}）；smoke另6×40步。训练旋量残差点{int(trained.residual_training_spinor_points.sum())}，不含后评价/参考。",
        " · ".join(link(RAW/n) for n in ("all_runs.csv", "all_summary.csv", "run_artifact_index.csv", "decision.json", "mechanism_summary.json", "data_audit.json", "audit.json", "protected_history.json")))
    sections.extend(["## 15. 图表证据", ""])
    manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
    for i, figure in enumerate(manifest["figures"], 1):
        name = figure["figure_id"]
        sections.extend([f"### 图{i}：{name}", "", figure["caption"]+" Seed范围："+figure["seed_scope"]+"。", "",
            f"![图{i}](<{(FIGURES/(name+'.png')).as_posix()}>)", "",
            link(FIGURES/(name+".svg"), "SVG")+" · "+link(FIGURES/(name+".pdf"), "PDF")+" · "+
            " · ".join(link(ROOT/r["path"], "Source "+str(j+1)) for j, r in enumerate(figure["source_data"])), ""])
    sections.extend([link(FIGURES/"FIGURE_QA.md", "最终QA")+"；"+link(FIGURES/"visual_review.md", "实际目视记录"), "",
                     "## 附录A：全部原始运行产物", "",
                     "不删失败、探索与条件run；checkpoint均为固定终点。恢复、预训练载波、交接与额外诊断保存在同run目录。", ""])
    artifact_rows = []
    for _, r in all_runs.sort_values(["stage", "case", "model", "seed"]).iterrows():
        folder = RAW/r.run_path
        artifact_rows.append([r.stage, r.case, LABELS[r.model]+(" ·平衡" if r.balanced else ""), int(r.seed)]+
            [link(folder/n, n) for n in ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")])
    sections.extend([table(["stage", "case", "配置", "seed", "config", "metrics", "loss", "series", "dense", "checkpoint"], artifact_rows), "",
        "## 附录B：下一轮原则", "",
        "优先按独立验证与消融决策，不按最好图片。成功者先解决失败边界、载波成本和容量混杂；全部失败时依据梯度与未加权模态残差开展新的单因素验证。不把局部修复或小样本救援提高为完整动力学成功。"])
    report = "\n".join(sections)+"\n"
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/"report.md").write_text(report, encoding="utf-8")
    (ROOT/"R8_COUPLING_RECOVERY_REPORT.md").write_text(report, encoding="utf-8")
    files = [ROOT/"src"/name for name in ("aggregate_r8_coupling_recovery.py", "audit_r8_coupling_recovery.py",
        "plot_r8_coupling_recovery.py", "build_r8_coupling_recovery_report.py", "r8_report_qa.py",
        "r8_free_packet_evaluation_adapter.py")]
    files += list((ROOT/"tests").glob("test_r8_reporting.py"))
    write_json(RAW/"reporting_code_sha256.json", dict(files={p.relative_to(ROOT).as_posix(): sha256(p) for p in files}))
    print(f"Report built: {len(all_runs)} runs, {len(manifest['figures'])} figure groups", flush=True)


if __name__ == "__main__":
    build_report()
