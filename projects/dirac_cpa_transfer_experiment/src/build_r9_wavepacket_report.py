from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .r9_config import ROOT, RAW, DOCS, FIGURES, MODELS, SEEDS, ARTIFACTS, run_directory, THRESHOLDS
from .freeze_r7_phase_release import sha256, write_json

LABELS = {MODELS[0]: "A：载波＋复包络", MODELS[1]: "B：Complex", MODELS[2]: "C：双网络Complex"}
BT = chr(96)


def link(label, path):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def table(frame, columns, labels=None):
    labels = labels or columns
    lines = ["| "+" | ".join(labels)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.7g}"
            values.append(str(value))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def build_report():
    frame = pd.read_csv(RAW/"per_seed_metrics.csv")
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((RAW/"diagnostic_ignoring_potential.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RAW/"summary_metrics.csv")
    comparison = pd.read_csv(RAW/"comparison_summary.csv")
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    test_path = RAW/"pytest_final_result.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else dict(status="PENDING")
    a_pass = decision["absolute_success"][MODELS[0]]["pass_check"]
    matched = decision["A_advantages"][MODELS[2]]
    conclusion = "**A通过预注册完整动力学标准**" if a_pass else "**A未通过预注册完整动力学标准**"
    advantage = "A的平均旋量与概率流误差仍低于总参数匹配C" if matched["both_mean_errors_lower"] else "A未同时在平均旋量与概率流误差上优于总参数匹配C"
    lines = []
    def add(text=""):
        lines.extend([text, ""])
    def stat(model, key):
        row = summary[(summary.model == model)&(summary.metric == key)].iloc[0]
        return f"{row['mean']:.7g} ± {row.population_sd:.7g}"
    add("# R9：弱周期外势下高斯波包的可学习载波＋复包络验证")
    add(f"实验ID：{BT}r9_wavepacket_periodic_potential{BT}。日期：2026-10-03（Asia/Shanghai）。独立R9目录，历史结果保持原貌。")
    add("## 1. 主要结论")
    add(conclusion+"。"+advantage+"。绝对成功与相对优势分别判断；全部15个正式run保留，不追加补救模型。")
    rows = []
    for model in MODELS:
        row = dict(model=LABELS[model])
        row.update({key: stat(model, key) for key in THRESHOLDS})
        row["joint"] = f"{decision['absolute_success'][model]['joint_pass_count']}/5"
        row["passed"] = "PASS" if decision["absolute_success"][model]["pass_check"] else "FAIL"
        rows.append(row)
    add(table(pd.DataFrame(rows), ["model", *THRESHOLDS, "joint", "passed"],
              ["模型", "Spinor", "Current", "Fidelity", "Modal population", "Max norm drift", "联合通过", "绝对标准"]))
    add("均值±population SD（ddof=0），n=5，非SEM/CI。Norm项先对每seed取全时间最大漂移，再在五seed间统计；至少4/5种子五项联合通过才成功。")
    add(f"A/C实际均值误差比：spinor={matched['spinor_ratio_of_means']:.8g}，current={matched['current_ratio_of_means']:.8g}；逐seed两项联合胜出{matched['paired_joint_spinor_current_wins']}/5。未事后添加优势倍数阈值。")
    add("## 2. 历史依据与本轮范围")
    add("六轮综合报告区分自由载频优势与原始实振幅LTPB外势失败；R5通过自由高斯已知初谱验证，R6/R7没有完整恢复空通道耦合。R8成功的是辅助自由载波预训练＋复包络A，不是原始纯实振幅LTPB。")
    add("本轮只检验R8 A迁移到同一R5高斯初谱的给定外势问题，以及总参数匹配控制。没有时间窗、守恒正则、动态切换、幅值阈值或模态权重。")
    add(" · ".join(link(label, ROOT/name) for label, name in [
        ("六轮综合报告", "DIRAC_LTPB_COMPREHENSIVE_EXPERIMENT_REPORT.md"),
        ("R5", "WAVEPACKET_SHARED_SPECTRAL_LTPB_REPORT.md"),
        ("R6", "PERIODIC_POTENTIAL_COUPLED_LTPB_REPORT.md"),
        ("R7", "R7_PHASE_RELEASE_EXPERIMENT_REPORT.md"), ("R8", "R8_COUPLING_RECOVERY_REPORT.md")]))
    add("## 3. 固定物理、初态与参考")
    add(r"$$i\partial_t\psi=[-i\sigma_x\partial_x+5\sigma_z+0.2\cos(x)I_2]\psi,\quad x\in[-8\pi,8\pi),\quad T=5.$$")
    add(r"$$k_n=n/8,\quad g_n=C\exp[-(k_n-2)^2/(4\times0.5^2)],\quad n=0,\ldots,32,\quad\sum_n|g_n|^2=1.$$")
    add(r"$$\widetilde\psi_n(0)=g_n(u_++u_-)/\sqrt2,\qquad\psi=L^{-1/2}\sum_n e^{ik_nx}\widetilde\psi_n.$$")
    add("扩大截断只零填充同一33系数，不重新归一化。q=1移动n±8；势乘I₂。自由能支不是外势完整Hamiltonian的独立能级。")
    add(r"$$R_n=i\partial_t\widehat{\widetilde\psi}_n-(k_n\sigma_x+5\sigma_z)\widehat{\widetilde\psi}_n-0.1(\widehat{\widetilde\psi}_{n-8}+\widehat{\widetilde\psi}_{n+8}).$$")
    add("块带状Hermitian Hamiltonian本征演化与独立矩阵指数交叉验证。第二组首个通过，选较小截断[-48,80]，129模态、258×258矩阵。第三组无需执行，不写成已验证。")
    add(table(pd.DataFrame(reference["comparisons"]), ["smaller", "larger", "full_complex_zero_padded_relative_difference",
        "lower_eight_layers_max_total_occupation", "upper_eight_layers_max_total_occupation", "pass_check"],
        ["较小截断", "较大截断", "全复轨迹零填充差", "下8层占据", "上8层占据", "通过"]))
    selected = reference["comparisons"][-1]
    add(f"选中参考Hermiticity误差={selected['hermiticity_max_abs']:.3g}，范数漂移={selected['norm_max_drift']:.6g}，时间逆转误差={selected['reversal_relative_error']:.6g}，独立expm最大相对差={selected['independent_expm_max_relative_error']:.6g}；4001点谱PDE残差RMS范数={selected['spectral_pde_residual_l2_rms']:.6g}。截断依据参考选定，先于任何正式PINN结果。")
    add("### 外势可测性")
    add(f"忽略外势的同初態自由解相对耦合参考：常数规范spinor={diagnostic['spinor_rel_l2_gauge_aligned']:.8g}（raw={diagnostic['spinor_rel_l2_raw']:.8g}），current={diagnostic['current_rel_l2']:.8g}，density={diagnostic['density_rel_l2']:.8g}，fidelity={diagnostic['fidelity_mean']:.8g}，modal population={diagnostic['modal_population_rel_l2']:.8g}。远大于参考截断差，外势影响确实可测；该自由诊断不进入learned排名。")
    add(f"参考初始支持外最大占据={diagnostic['reference_max_occupation_outside_initial_support']:.8g}。新增支持只指n<0或n>32，不能把中心n=16外都称为新侧带，因为初态已有宽谱。")
    add(" · ".join([link("参考校准", RAW/"reference_validation.json"), link("参考测试轨迹", RAW/"reference_test_grid.npz"),
        link("自由诊断", RAW/"diagnostic_ignoring_potential.json"), link("自由轨迹", RAW/"diagnostic_ignoring_potential.npz")]))
    add("## 4. 模型、理论与公平预算")
    add("A直接使用R8 LearnedCarrier，p=exp[iτqθ]、c=p(c₀+τz)；每保留k、正负支分别作非零单位本征旋量自由IVP。B为c=c₀+τz。C为c=c₀+τ(z₁+z₂)/√2。所有网络4×64 tanh、float64/complex128、Adam 10⁻³，输入沿用R8规则[(k−2)/8,t/5]。")
    add("A=6000自由＋14000耦合；B/C=20000耦合，每run共20000优化步。每步32个新时间×全部129模态；载波每样本两个独立单位旋量。主loss仅完整未加权旋量残差，不改为含E(k)的对角方程。")
    add(r"$$F_n=[u_+p_+,u_-p_-],\quad d_n=c_{n,0}+\tau z_n,\quad R_n=iF_n\dot d_n+(i\dot F_n-H_nF_n)d_n-0.1(F_{n-8}d_{n-8}+F_{n+8}d_{n+8}).$$")
    add("冻结权重仍保留p_t及载波近似缺陷。时间导数采用精确Linear/tanh链式内核，保留参数图；状态、时间导数、残差与全部参数梯度独立对照原自动微分。这只是同一前向和损失的计算实现，无时间差分、无频率标签。")
    budget = []
    for name in MODELS:
        m = frame[frame.model == name].iloc[0]
        budget.append(dict(model=LABELS[name], active=int(m.active_parameters), frozen=int(m.frozen_parameters),
            total=int(m.total_parameters), steps=f"{int(m.carrier_steps)}/{int(m.coupled_steps)}",
            spinor_points=int(m.training_carrier_unit_spinor_residual_points+m.training_coupled_spinor_residual_points),
            scalar_points=int(m.training_complex_scalar_residual_evaluations)))
    add(table(pd.DataFrame(budget), ["model", "active", "frozen", "total", "steps", "spinor_points", "scalar_points"],
              ["模型", "耦合活动参数", "冻结参数", "总参数", "自由/耦合步数", "旋量残差评价点", "复标量残差评价数"]))
    add("A预训练活动12802、闲置包络12932；耦合活动12932、冻结12802。C比A总参数多130即0.5052%，但耦合活动参数约为A两倍。每run后评价4001×129完整耦合残差点；A另201×129×2单位载波旋量残差点。步数、活动参数、总存储容量及实际计算成本必须分别理解。")
    add("seeds300–304。A/B/C第一四输出网络逐元素配对，C第二网络独立；两输出载波不与四输出网络宣称逐元素配对。初始化、自由时间、耦合时间流独立；A/B/C共享14000步耦合公共前缀，B/C全部20000步相同。完整采样迹、Adam和RNG恢复状态保存。")
    add("固定五物理核slot，mask1/2/4/8/16；不使用seed＋slot索引。启动前互斥检查，worker检查实际OS affinity，一线程，同seed依次A/B/C。没有沿用R8调度缺陷。仍共享缓存、内存及功耗资源，报告真实成本，不作跨硬件或单一算法加速声明。")
    add(" · ".join([link("理论核查", DOCS/"theory_check.md"), link("模型", ROOT/"src/r9_models.py"),
        link("完整残差", ROOT/"src/r9_loss.py"), link("时间导数", ROOT/"src/r9_time_derivative.py"),
        link("核分配", RAW/"runtime_plan.json")]))
    add("## 5. 冻结与评价口径")
    add(f"冻结manifest SHA-256：{BT}{sha256(RAW/'frozen_protocol_and_code.json')}{BT}。正式前完成参考和维度/初值/梯度/NaN/落盘smoke、恢复与链式导数测试，没有按精度调参。固定第20000步最终检查点，不选最好checkpoint、不选择性重跑。")
    add("物理网格Nx=1024端点不重复、Nt=201。129模态跨度128，双线性误差平方最高整数频率256；Fourier求积不混叠，并核验Parseval与Nx=2048一致性。旋量、复系数只消去一个全时空常数相位，不逐时间或逐模态对齐；raw误差另存。")
    add("参考相位mask=支占据≥10⁻⁶；预测相位定义另需占据≥10⁻¹²并报覆盖率。相干C_ab=c_ac_b*按两端参考活跃非对角通道统计，塌缩预测仍计入。保存完整参考通道mask及复系数，pair mask由两端参考mask与非对角规则确定。")
    add("从总旋量投影重建正负自由支，圆周一阶矩z定义中心L unwrap(arg z)/(2π)、宽度L√(−2ln|z|)/(2π)。全轨迹相对同一耦合参考，不用自由直线或内部载波相位替代。低预测支范数另存，其中心/宽度只作失败诊断。")
    add("五seed均值spinor≤.10、current≤.15、fidelity≥.98、modal population≤.10、maximum norm drift≤.02，并至少4/5五项联合通过。优势另报实际比、配对差及population SD，不增加事后倍数阈值。")
    add(" · ".join([link("预注册", DOCS/"preregistration.md"), link("冻结清单", RAW/"frozen_protocol_and_code.json"),
        link("机器协议", RAW/"protocol.json"), link("独立评价", ROOT/"src/r9_evaluate.py")]))
    add("## 6. 全部逐seed结果")
    per_seed = frame.copy()
    per_seed["model"] = per_seed.model.map(LABELS)
    per_seed["joint"] = ["PASS" if all(row[key] >= bound if key == "fidelity_mean" else row[key] <= bound
        for key, bound in THRESHOLDS.items()) else "FAIL" for _, row in per_seed.iterrows()]
    add(table(per_seed, ["model", "seed", *THRESHOLDS, "joint"],
              ["模型", "Seed", "Spinor", "Current", "Fidelity", "Modal", "Max norm drift", "联合"]))
    add(table(per_seed, ["model", "seed", "density_rel_l2", "fidelity_min", "spinor_final_rel_l2",
        "branch_population_rel_l2", "branch_complex_coefficient_rel_l2", "active_coherence_rel_l2", "phase_defined_reference_active_coverage"],
        ["模型", "Seed", "Density", "Min fidelity", "Final spinor", "支占据", "复支系数", "活跃相干", "相位覆盖率"]))
    add("## 7. 真实配对优势与容量")
    comp = comparison[comparison.metric.isin(["spinor_rel_l2_gauge_aligned", "current_rel_l2",
                                             "modal_population_rel_l2", "active_coherence_rel_l2"])].copy()
    comp["target"], comp["baseline"] = comp.target.map(LABELS), comp.baseline.map(LABELS)
    add(table(comp, ["target", "baseline", "metric", "paired_difference_mean", "paired_difference_population_sd",
        "ratio_of_means", "mean_of_paired_ratios", "favorable_pair_count"],
        ["Target", "Baseline", "指标", "配对差均值", "配对差population SD", "均值误差比", "配对比值均值", "有利种子数"]))
    add("差为target−baseline，误差负差有利，fidelity正差有利；均值之比和配对比值均值分列。不以n=5作总体显著性声明。")
    add("A/C同时涉及表示、辅助单位IVP、活动参数与耦合步数。若A仍优于C，支持优势不能仅由总参数量解释；若没有优势，则A/B比较不足以排除容量解释。C/B真实结果单独保留。")
    add("本轮没有独立消融预训练、纯实包络或随机载波，不能唯一分配预训练的单独因果贡献。R8移除载波历史结果提供背景，不能冒充R9单因素证据。比较的是整个冻结载波＋复包络流程。")
    add(" · ".join([link("全部配对差", RAW/"paired_comparisons.csv"), link("比较汇总", RAW/"comparison_summary.csv")]))
    add("## 8. 波包、模态及分支轨迹")
    add(table(per_seed, ["model", "seed", "predicted_max_occupation_outside_initial_support", "plus_center_rmse",
        "minus_center_rmse", "plus_width_relative_l2", "minus_width_relative_l2", "plus_min_predicted_norm", "minus_min_predicted_norm"],
        ["模型", "Seed", "新增支持占据", "+中心RMSE", "−中心RMSE", "+宽度L2", "−宽度L2", "+最小支范数", "−最小支范数"]))
    add("所有种子的中心/宽度轨迹曲线与数据保存；动力学与模态图固定seed300，显示参考与三个模型。初始支持外占据才表示新增支持，不把中心外原有高斯占据误称新增侧带。")
    add("## 9. 载波缺陷、未加权残差及外泄漏")
    add(table(per_seed[per_seed.model == LABELS[MODELS[0]]], ["seed", "carrier_free_pde_residual_rms",
        "carrier_phase_slope_mean_relative_error", "carrier_phase_slope_max_relative_error"],
        ["Seed", "单位IVP载波缺陷RMS", "斜率平均相对误差", "斜率最大相对误差"]))
    add("辅助载波初态在所有k、每支非零，因此其全部斜率可诊断；这不是目标空通道物理相位。精确自由频率只在独立后评价出现，不进入forward/loss/optimizer/checkpoint。载波仍近似，缺陷完整进入耦合PDE。")
    add(r"$$\|e(t)\|\le\|e(0)\|+\int_0^t\|R(s)\|\,ds.$$")
    add("此理论针对所选有限Hermitian系统及raw场误差。保存4001点完整复残差、未加权分模态功率、两端各八个截断外残差、4001/2001梯形积分差。数值积分不是严格连续误差认证，也不认证无限谱。有限截断内估计与完整物理残差、外泄漏分别报告。")
    add(table(per_seed, ["model", "seed", "fine_unweighted_pde_loss", "residual_integral_fine",
        "residual_integral_quadrature_relative_difference", "residual_estimate_max_numerical_violation",
        "exterior_residual_leakage_max_l2_norm", "full_physical_residual_integral"],
        ["模型", "Seed", "Fine PDE loss", "数值残差积分", "积分相对差", "数值违例", "外泄漏范数", "完整物理积分"]))
    add("参考截断已经收敛；远大于参考差的learned误差不能归为谱截断失败。载波缺陷、包络耦合优化及控制塌缩/有限预算失败分开判断。低loss或范数接近1不代替复旋量与概率流动力学准确性。")
    add("### 冻结后的外侧复残差归档修正")
    add("冻结评价代码的截断外复数归档保存了正号耦合源；按R=i∂tψ−Hψ定义，零延拓场的外侧PDE残差应为该源的负值。这个符号不影响范数、残差积分或五项主指标。原六件套、评价数据和科学代码全部原样保留，另存full_physical_residual.npz，包含145模态完整未加权复残差及正确负号的外侧项。")
    add("此项只补齐诊断归档，未增加优化步、改变模型、参考、评价口径或成功判据。逐run记录原产物SHA-256、补档额外forward工作量与耗时；独立物理空间算子测试验证符号。额外后处理成本与原训练、评价成本分列。")
    add(link("全部15 run的归档修正记录", RAW/"full_physical_residual_archival_supplement.json"))
    add("## 10. 真实资源成本")
    costs = []
    for name in MODELS:
        costs.append(dict(model=LABELS[name], pre=stat(name, "carrier_training_seconds"), coupled=stat(name, "coupled_training_seconds"),
            total=stat(name, "training_seconds"), evaluate=stat(name, "evaluation_seconds"), rss=stat(name, "peak_rss_mib")))
    add(table(pd.DataFrame(costs), ["model", "pre", "coupled", "total", "evaluate", "rss"],
              ["模型", "预训练秒", "耦合秒", "总优化秒", "评价秒", "Peak RSS MiB"]))
    add("训练时间累计每步loss/梯度/Adam与阶段转换实际工作，不含恢复检查点、CSV落盘及后评价；评价另计，进程会话耗时另存。训练期RSS与含评价的peak RSS分别保存，native采样方法落盘。步数相同不等于成本相同。")
    add(table(per_seed, ["model", "seed", "carrier_training_seconds", "coupled_training_seconds", "evaluation_seconds",
        "training_peak_rss_mib", "peak_rss_mib", "cpu_affinity_mask"],
        ["模型", "Seed", "预训练秒", "耦合秒", "评价秒", "训练RSS MiB", "总RSS MiB", "核mask"]))
    add(link("分阶段成本及残差工作量", RAW/"cost_and_residual_work.csv"))
    add(link("归档符号补正的独立后处理成本", RAW/"postfreeze_archival_cost.csv"))
    add("## 11. 全部图表及source data")
    add("统计图全部五seed，误差棒population SD；动力学seed300。SVG文本可编辑、PDF TrueType、PNG300dpi，共享色标，无平滑、删除失败点或每模型单独色标归一化。")
    captions = {
        "reference_calibration": "参考截断校准及可测外势效应",
        "all_seed_primary_metrics": "全部种子五项绝对标准",
        "paired_comparisons": "实际配对差及误差比",
        "dynamics_seed300": "固定seed300密度与概率流",
        "modal_branch_seed300": "总模态与正负支占据，虚线为初始支持边界",
        "branch_center_width": "全部种子分支中心及宽度轨迹",
        "diagnostics_all_seeds": "相干、复谱、残差积分与外泄漏",
        "carrier_defects_all_seeds": "全部载波缺陷与评价斜率",
        "training_costs": "全部种子时间与RSS"}
    for i, figure in enumerate(qa["figures"], 1):
        name = figure["figure"]
        add(f"### 图{i}：{captions[name]}")
        add(f"![{name}](<{(FIGURES/(name+'.png')).as_posix()}>)")
        add(" · ".join([link("SVG", FIGURES/(name+".svg")), link("PDF", FIGURES/(name+".pdf"))]+
                      [link(f"Source CSV {j+1}", FIGURES/p) for j, p in enumerate(figure["source_csv"])]))
    add("## 12. 测试、审计和完整证据")
    add(f"完整pytest：{test['status']}。新增测试覆盖n±8、参考收敛、R5高斯归一化、硬初值、载波时间导数和缺陷、参数量、随机流、原步恢复、无频率泄漏、Fourier积分、相干mask及产物完整性。")
    add("工程完整性与科学成功独立。审计从每run原始复谱和物理网格独立重算五项主指标，并核验单一常数相位、模态占据和保存场的一致性。最终状态以audit.json、link_check和FIGURE_QA为准。历史保护清单覆盖7372文件；科学冻结与最终产物SHA-256分别保存。")
    add(" · ".join(link(label, path) for label, path in [
        ("Decision", RAW/"decision.json"), ("Audit", RAW/"audit.json"), ("链接", RAW/"link_check.json"),
        ("FIGURE_QA", FIGURES/"FIGURE_QA.md"), ("图QA JSON", FIGURES/"FIGURE_QA.json"),
        ("完整pytest", RAW/"pytest_final_result.json"), ("逐seed", RAW/"per_seed_metrics.csv"),
        ("汇总SD", RAW/"summary_metrics.csv"), ("产物索引", RAW/"run_artifact_index.csv"),
        ("历史保护", RAW/"protected_history.json"), ("产物SHA-256", RAW/"artifact_sha256.json")]))
    add("### 每run六件套及恢复状态")
    evidence = []
    for model in MODELS:
        for seed in SEEDS:
            directory = run_directory(model, seed)
            evidence.append(dict(model=LABELS[model], seed=seed, files=" · ".join(
                link(name, directory/name) for name in ARTIFACTS+("recovery_state.pt", "full_physical_residual.npz",
                                                               "full_physical_residual_archival_record.json"))))
    add(table(pd.DataFrame(evidence), ["model", "seed", "files"], ["模型", "Seed", "证据"]))
    add("### 可复现入口")
    add(BT*3+"powershell\n"+"\n".join("python -m "+cmd for cmd in [
        "src.run_r9_wavepacket prepare", "src.run_r9_wavepacket smoke", "src.run_r9_wavepacket freeze",
        "src.run_r9_wavepacket formal", "src.r9_full_residual_archive", "src.aggregate_r9_wavepacket", "src.plot_r9_wavepacket",
        "src.build_r9_wavepacket_report", "pytest", "src.audit_r9_wavepacket"])+"\n"+BT*3)
    add("已完成run不会重训，未完成仅从原模型、Adam、RNG原步续跑；科学文件变更会被拒绝。图表视觉核验记录须完成后执行最终审计。")
    add("## 13. 归因与解释边界")
    add("完整动力学以五项均值及4/5联合规则判断；容量作用以C/B和A/C数据判断；预训练单独因果未在本轮隔离。控制失败不证明Complex数学不可表示，A失败也不证明所有载波方法不可行。参考已收敛，不把优化失败写成谱截断失败。")
    add("即使通过，也仅证明已知有限Fourier高斯初谱在此弱周期外势、固定预算与所选有限截断下有效。不能宣称未知连续谱发现、一般外势或通用Dirac求解器，也未证明任意新问题零样本迁移。")
    report = "\n".join(lines)
    path = ROOT/"R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md"
    path.write_text(report, encoding="utf-8")
    (DOCS/"report.md").write_text(report, encoding="utf-8")
    files = [ROOT/"src"/name for name in ("aggregate_r9_wavepacket.py", "plot_r9_wavepacket.py", "qa_r9_wavepacket_figures.py",
              "build_r9_wavepacket_report.py", "audit_r9_wavepacket.py", "r9_full_residual_archive.py")]
    write_json(RAW/"reporting_code_sha256.json", dict(files={p.relative_to(ROOT).as_posix(): sha256(p) for p in files}))
    return path


if __name__ == "__main__":
    print(build_report())
