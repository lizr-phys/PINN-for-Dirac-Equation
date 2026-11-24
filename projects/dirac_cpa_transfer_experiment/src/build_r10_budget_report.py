from __future__ import annotations
import json
import pandas as pd
from .r10_config import ROOT, RAW, FIGURES, DOCS, SEEDS, run_directory, source_directory
from .r9_config import RAW as R9_RAW, THRESHOLDS, ARTIFACTS
from .aggregate_r10_budget import STAGES
from .freeze_r7_phase_release import write_json, sha256

TITLES = {"spinor_rel_l2_gauge_aligned": "Spinor", "current_rel_l2": "Current",
    "fidelity_mean": "Fidelity", "modal_population_rel_l2": "Modal population", "max_norm_drift": "Max norm drift"}


def link(label, path):
    return f"[{label}](<{path.as_posix()}>)"


def table(frame, columns, labels=None):
    names = labels or columns
    rows = ["| "+" | ".join(names)+" |", "| "+" | ".join("---" for _ in names)+" |"]
    for _, row in frame.iterrows():
        cells = []
        for name in columns:
            value = row[name]
            cells.append(f"{value:.8g}" if isinstance(value, float) else str(value))
        rows.append("| "+" | ".join(cells)+" |")
    return "\n".join(rows)


def build_report():
    frame = pd.read_csv(RAW/"per_seed_metrics.csv")
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    changes = pd.read_csv(RAW/"paired_change_summary.csv")
    costs = pd.read_csv(RAW/"cost_and_residual_work.csv")
    r9 = json.loads((R9_RAW/"decision.json").read_text(encoding="utf-8"))
    new = frame[frame.stage == STAGES[1]]
    joint = decision["absolute_success"][STAGES[1]]["joint_pass_count"]
    succeeded = decision["absolute_success"][STAGES[1]]["pass_check"]
    outcome = ("条件续训通过原绝对判据" if succeeded else "固定追加预算仍未通过原绝对判据")
    rows = []
    for stage in STAGES:
        part = frame[frame.stage == stage]
        row = {"阶段": stage}
        row.update({TITLES[k]: f"{part[k].mean():.8g} ± {part[k].std(ddof=0):.8g}" for k in THRESHOLDS})
        d = decision["absolute_success"][stage]
        row.update({"联合通过": f"{d['joint_pass_count']}/5", "判据": "PASS" if d["pass_check"] else "FAIL"})
        rows.append(row)
    summary = pd.DataFrame(rows)
    lines = ["# R10：固定载波的高斯外势耦合预算诊断", "",
        "实验ID：r10_wavepacket_budget_diagnosis。日期：2026-10-03（Asia/Shanghai）。全部结果为R9同一批五个训练祖先的条件续训，不是新独立重复。", "",
        "## 1. 结论", "",
        f"**{outcome}，{joint}/5种子联合通过。** R9第20000步的3/5和未建立完整动力学成功的判定保持不变；R10固定第26000步结果单独归档。没有筛种子、最佳检查点、预测归一化或新补救结构。", "",
        table(summary, list(summary.columns)), "",
        "均值±population SD（ddof=0），n=5配对训练祖先。最大范数漂移先取各seed全时间最大值，再统计五seed。五项均值及至少4/5联合规则同时要求。", "",
        "## 2. 为什么先做预算诊断", "",
        "R1–R5建立已知自由能支上的表示优势；R6/R7说明局部空通道切向修复并不保证长期耦合成功；R8成功方案A引入自由载波预训练及复包络。R9迁移高斯初谱后旋量与流误差明显低于总参数匹配C，但两个种子仅在范数项超标。R9旋量和流A/C均值比为0.03651626、0.04948497，不能用新增总容量单独解释优势。", "",
        "R9参考收敛与物理影响已独立验证；此处先固定物理、参考、表示及损失，增加耦合优化预算，避免同时改变机制和任务。", "",
        link("完整下一步计划", ROOT/"NEXT_STEP_PLAN_AFTER_R9.md")+" · "+link("R9报告", ROOT/"R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md")+" · "+link("R10预注册", DOCS/"preregistration.md"), "",
        "## 3. 条件续训协议与公平边界", "",
        "物理仍为m=5、V=0.2cos(x)、L=16π、T=5，R5固定33模态n=0…32、k=n/8、中心k0=2、σk=0.5归一化高斯初谱和等权±支。截断[-48,80]共129模态，扩展只零填充；q=1为n±8。R9参考及自由诊断完全沿用，无重新选择截断或势强。", "",
        "五个A的原20000步状态包含6000步载波预训练及14000步耦合训练。逐seed恢复Adam动量、全部RNG和时间采样迹，再追加6000步，固定总26000步（耦合20000）。载波权重逐位一致，时间导数与近似缺陷完整保留。", "",
        "结构和loss原样继承R8 A/R9 A：4×64 tanh、float64/complex128、Adam lr=10⁻³，每步32新时间×129模态；活动12932、冻结12802、总25734参数。只有完整未加权耦合旋量残差；精确频率与矩阵参考轨迹只进入独立后评价。", "",
        "R10耦合时间20000步与R9 B/C配对序列逐元素相同；其前14000步等于原A。但R10总预算26000步高于C的20000步，故不作同总预算A/C优势声明。本轮检验同一批祖先的预算敏感性，未隔离预训练的单独因果贡献。", "",
        "五物理核mask1/2/4/8/16；worker一线程，启动前互斥核验及进程内实际OS affinity一致。监控和报告工作使用另一核，仍共享内存、功耗与缓存资源。", "",
        "科学冻结manifest SHA-256："+sha256(RAW/"frozen_protocol_and_code.json")+"。只做两步维度、初值、梯度、NaN与落盘smoke及精确恢复核验，不按精度调参。", "",
        link("冻结清单", RAW/"frozen_protocol_and_code.json")+" · "+link("CPU调度", RAW/"runtime_plan.json")+" · "+link("理论核查", DOCS/"theory_check.md"), "",
        "## 4. 全部配对结果", ""]
    for stage in STAGES:
        part = frame[frame.stage == stage].copy()
        passed = decision["absolute_success"][stage]["joint_pass_seeds"]
        part["joint"] = ["PASS" if s in passed else "FAIL" for s in part.seed]
        lines += [f"### {stage}", "", table(part, ["seed"]+list(THRESHOLDS)+["joint"],
            ["Seed"]+list(TITLES.values())+["联合"]), ""]
    lines += ["差为R10−R9，误差负差或fidelity正差有利；两个阶段同一祖先配对，不作独立样本显著性声明。均值之比与配对比值均值分列。", "",
        table(changes, ["metric", "paired_difference_mean", "paired_difference_population_sd", "ratio_of_means", "mean_of_paired_ratios", "favorable_pair_count"],
            ["指标", "配对差均值", "配对差population SD", "均值比", "配对比值均值", "有利配对数"]), "",
        link("逐seed全部指标", RAW/"per_seed_metrics.csv")+" · "+link("逐seed配对差", RAW/"paired_changes.csv")+" · "+link("汇总", RAW/"summary_metrics.csv"), "",
        "## 5. 统一独立评价", "",
        "Nx=1024、Nt=201，周期端点不重复，Fourier积分不混叠；沿用R9 Parseval/2048点核验。旋量和复谱只消去一个全时空常数相位，raw值另存。参考占据≥10⁻⁶定义活跃mask，预测塌缩仍计入相干。", "",
        "密度、流、fidelity、总模态及±支占据、复系数、活跃相干、中心/宽度相对同一耦合参考的轨迹全部保存。新增支持仅为原初谱n<0或n>32，不把中心外原有宽谱误称新侧带。", "",
        table(new, ["seed", "density_rel_l2", "branch_population_rel_l2", "branch_complex_coefficient_rel_l2",
            "active_coherence_rel_l2", "plus_center_rmse", "minus_center_rmse", "plus_width_relative_l2", "minus_width_relative_l2"],
            ["Seed", "Density", "±支占据", "复支系数", "活跃相干", "+中心RMSE", "−中心RMSE", "+宽度L2", "−宽度L2"]), "",
        "## 6. 范数为何变化：恒等式与数值诊断", "",
        "$$\\dot N=2\\operatorname{Im}\\langle\\psi,R\\rangle,\\quad R=iF\\dot d+Gd-V\\psi,\\quad G=i\\dot F-H_0F.$$", "",
        "单位模载波使F†F=I，F†G为Hermitian，故载波缺陷的直接全局范数投影为零；实外势Hermitian耦合的全局投影也为零。于是范数直接变化来自包络切向2ReΣd†ḋ。这并不排除不准确载波改变包络优化难度的间接作用。", "",
        "保存4001点未归一化预测及完整复残差，用投影积分重建N(t)−N(0)，并保存4001/2001数值差；201点保存载波/包络/势分解。离散积分不是严格连续误差认证，也不认证无限谱。", "",
        table(new, ["seed", "fine_grid_max_norm_drift", "norm_balance_integration_max_abs_discrepancy",
            "norm_balance_4001_2001_integral_max_difference", "carrier_defect_direct_norm_projection_max_abs",
            "hermitian_coupling_global_norm_projection_max_abs", "envelope_vs_complete_norm_projection_max_difference"],
            ["Seed", "细网格max drift", "范数积分闭合差", "4001/2001积分差", "载波直接投影max", "势直接投影max", "包络与完整投影差"]), "",
        "载波unit-IVP RMS和相位斜率诊断与R9相同，因为载波张量未改变。最大载波RMS的R9 seed304却具有最小范数漂移，不能把相关量直接写成范数失败的唯一原因。", "",
        "R10保存145模态全物理残差（129截断内+两侧各8个外模态），外模态R_ext=−0.1边界源项。R9冻结评价器的正耦合源档案和独立负号补充原样保留；R10在其冻结前规定正确复符号。符号修订不改变R9原指标或优化。", "",
        table(new, ["seed", "residual_integral_fine", "full_physical_residual_integral",
            "exterior_residual_integral", "carrier_free_pde_residual_rms"],
            ["Seed", "截断内残差积分", "全物理残差积分", "外残差积分", "载波单位缺陷RMS"]), "",
        "## 7. 参数、工作量和真实成本", "",
        "每seed新增6000×32×129=24,768,000旋量残差点，复标量数49,536,000；累计132,096,000旋量点。R9继承107,328,000点。总优化步数不同于残差工作量，活动参数不同于总参数。", "",
        "下表耗时单位秒。累计训练耗时=原有两阶段+新增耦合，评价是固定最终检查点的完整评价；额外范数诊断forward 516129个旋量点、额外残差调用0，包含在本次评价耗时中。peak RSS为进程20ms采样及OS高水位方法，完整方法字符串保存在metrics。", "",
        "训练秒数沿用R9逐步计算计时口径，不含检查点I/O；总执行时间另由runtime_plan与执行事件UTC时间保存。不同并发任务的耗时不等同单一算法加速比。", "",
        table(costs, ["seed", "inherited_carrier_training_seconds", "inherited_coupled_training_seconds",
            "additional_training_seconds", "cumulative_training_seconds", "evaluation_seconds", "peak_rss_mib"],
            ["Seed", "原载波秒", "原耦合秒", "新增耦合秒", "累计训练秒", "新评价秒", "Peak RSS MiB"]), "",
        link("完整成本和残差工作量", RAW/"cost_and_residual_work.csv"), "",
        "## 8. 图表与source data", ""]
    for item in qa["figures"]:
        name = item["figure"]
        lines += [f"### {name}", "", item["claim"], "",
            f"![{name}](<{(FIGURES/(name+'.png')).as_posix()}>)", "",
            link("PNG", FIGURES/f"{name}.png")+" · "+link("SVG", FIGURES/f"{name}.svg")+" · "+link("PDF", FIGURES/f"{name}.pdf"),
            " · ".join(link("Source CSV", FIGURES/p) for p in item["source_csv"]), ""]
    lines += [link("FIGURE_QA", FIGURES/"FIGURE_QA.md"), "",
        "## 9. 产物与工程审计", "",
        "每run保存六件套及Adam/RNG恢复、事件、完整残差、载波诊断、范数闭合和分解。全部源R1–R9文件以SHA保护，R9报告和原失败记录不改写。", ""]
    for seed in SEEDS:
        directory = run_directory(seed)
        paths = ARTIFACTS+("recovery_state.pt", "execution_events.jsonl", "norm_balance.csv", "residual_decomposition.csv",
                          "residual_by_mode.csv", "residual_integral.csv", "carrier_diagnostics.csv")
        lines += [f"- Seed {seed}: "+" · ".join(link(name, directory/name) for name in paths)]
    lines += ["", link("Decision", RAW/"decision.json")+" · "+link("Audit", RAW/"audit.json")+" · "+
        link("历史保护", RAW/"protected_prior_results.json")+" · "+link("全产物SHA", RAW/"artifact_sha256.json")+" · "+
        link("最终pytest", RAW/"pytest_final_result.json")+" · "+link("链接检查", RAW/"link_check.json"), "",
        "## 10. 能说明什么，下一步是什么", "",
        ("固定追加6000步后达到绝对标准，支持同一批训练祖先的条件预算恢复。仍需新种子独立重复，不能据此宣布一般预算阈值或预训练单因素因果效果。"
         if succeeded else
         "固定追加6000步仍不足以达到绝对标准；逐seed结果保留改善和退化。不能因此宣称数学不可表示或载波缺陷是唯一原因，也不追加新的补救结构或预算搜索。"), "",
        "接下来应先用新配对种子检验冻结流程的重复性，再用预先固定预算和工作量的载波预训练消融区分预训练与容量/优化贡献。有独立重复证据后才做少量势强和时间跨度的稳健性检验。", "",
        "本轮只涉及已知有限Fourier高斯初谱、给定弱周期外势和同一批训练祖先；不证明未知连续谱发现、一般外势适用或通用Dirac求解器。", ""]
    text = "\n".join(lines)
    (ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md").write_text(text, encoding="utf-8")
    (DOCS/"report.md").write_text(text, encoding="utf-8")
    files = [ROOT/"src"/name for name in ("aggregate_r10_budget.py", "plot_r10_budget.py",
        "qa_r10_budget_figures.py", "build_r10_budget_report.py", "audit_r10_budget.py")]
    files += [ROOT/"tests/test_r10_reporting.py"]
    write_json(RAW/"reporting_code_sha256.json", dict(files={p.relative_to(ROOT).as_posix(): sha256(p) for p in files}))
    return ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md"


if __name__ == "__main__":
    print(build_report())
