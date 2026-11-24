from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS
from .periodic_potential_coupled_ltpb_decision import TARGET, CONTROL
from .train_periodic_potential_coupled_ltpb import ROOT, RAW, RUN_ARTIFACTS, write_json
from .plot_periodic_potential_coupled_ltpb import FIGURES, LABELS, sha

REPORT = ROOT/"PERIODIC_POTENTIAL_COUPLED_LTPB_REPORT.md"


def relative(path):
    return path.relative_to(ROOT).as_posix()


def fmt(value):
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    return f"{v:.6g}"


def table(frame, columns, labels=None):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "|" + "---|"*len(columns)]
    for _, row in frame.iterrows():
        values = [str(row[c]) if isinstance(row[c], str) else fmt(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report():
    runs = pd.read_csv(RAW/"runs.csv")
    decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    calibration = json.loads((RAW/"calibration/calibration.json").read_text(encoding="utf-8"))
    verification = json.loads((RAW/"reference_verification.json").read_text(encoding="utf-8"))
    selected = json.loads((RAW/"formal_selection.json").read_text(encoding="utf-8"))
    freeze = json.loads((RAW/"frozen_protocol_and_code.json").read_text(encoding="utf-8"))
    figures = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))["figures"]
    target = runs[runs.model == TARGET]
    classification = decision["classification"]
    absolute = decision["absolute_results"]
    mean = lambda metric: float(target[metric].mean())
    narratives = {
        "COMPARATIVE_ADVANTAGE": "现有 LTPB 在已校准弱周期势上达到绝对迁移标准，并在同基底 Complex 对照中获得预注册比较优势。",
        "TRANSFER_SUCCESS": "现有 LTPB 达到绝对迁移标准，但未同时达到预注册的同基底比较优势标准。",
        "REPRESENTATION_FAILURE": "LTPB 未达到绝对迁移标准，而相同能支基底的 Complex 达到标准，符合预注册的表示失效分类。",
        "INCONCLUSIVE": "两种共享谱模型均未达到绝对迁移标准。当前固定预算对照不足以把失败单独归因于 LTPB；仍须区分共同优化失败与表示限制。",
    }
    calibration_rows = []
    for row in calibration["candidates"]:
        calibration_rows.append({"T": row["t_max"], "V0": row["potential_strength"],
                                 "side": row["large_ladder_observables"]["max_total_occupation_outside_center"],
                                 "edge": row["large_ladder_observables"]["max_outer_layer_total_occupation"],
                                 "delta": row["truncation_comparison"]["padded_full_trajectory_spinor_relative_l2"],
                                 "pass": "PASS" if row["candidate_pass"] else "FAIL"})
    selected_reference = next(row for row in calibration["candidates"] if row["t_max"] == 5 and row["potential_strength"] == .2)
    tests = ROOT/"results/logs/periodic_potential_coupled_ltpb/final_pytest.xml"
    test_status = "最终全套测试记录待生成"
    if tests.exists():
        suite = ElementTree.parse(tests).getroot()
        total = sum(int(s.get("tests", 0)) for s in suite.iter("testsuite"))
        failed = sum(int(s.get("failures", 0))+int(s.get("errors", 0)) for s in suite.iter("testsuite"))
        test_status = f"{total} 项测试，失败/错误 {failed}；[JUnit XML]({relative(tests)})"
    mean_rows = []
    for model in PERIODIC_POTENTIAL_MODELS:
        data = runs[runs.model == model]
        row = {"model": LABELS[model]}
        for metric in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "density_rel_l2", "fidelity_mean", "modal_population_rel_l2", "max_norm_drift"):
            row[metric] = f"{data[metric].mean():.6g} ± {data[metric].std(ddof=0):.3g}"
        mean_rows.append(row)
    summaries = pd.DataFrame(mean_rows)
    display = runs.copy()
    display["model"] = display.model.map(LABELS)
    core = ["model", "seed", "spinor_rel_l2_gauge_aligned", "density_rel_l2", "current_rel_l2", "fidelity_mean", "fidelity_min"]
    dynamics = ["model", "seed", "modal_population_rel_l2", "branch_population_rel_l2", "max_norm_drift", "center_occupation_rel_l2", "lower_sideband_occupation_rel_l2", "upper_sideband_occupation_rel_l2"]
    aux = ["model", "seed", "active_coherence_rel_l2", "active_branch_phase_circular_rmse", "active_branch_phase_coverage", "dense_physical_pde_residual_rms", "final_loss_pde"]
    resources = ["model", "seed", "parameter_count", "training_seconds", "peak_rss_mb", "peak_rss_increment_mb", "final_spinor_rel_l2_gauge_aligned"]
    ratios = pd.DataFrame([{"metric": m, **values} for m, values in decision["mean_ratios_and_paired_uncertainty"].items()])
    paired = pd.DataFrame(decision["paired_seed_comparisons"])
    activation_rows = []
    for model in PERIODIC_POTENTIAL_MODELS:
        for seed in range(5):
            with np.load(RAW/"learned"/model/f"seed_{seed}/dense_grid.npz") as dense:
                outside = dense["mode_offsets"] != 0
                center = int(np.flatnonzero(~outside)[0])
                predicted = dense["modal_occupations_predicted"]
                exact = dense["modal_occupations_exact"]
                activation_rows.append({"model": model, "seed": seed,
                                        "max_outside_population_reference": float(np.max(np.sum(exact[:, outside], axis=1))),
                                        "max_outside_population_predicted": float(np.max(np.sum(predicted[:, outside], axis=1))),
                                        "final_center_population_reference": float(exact[-1, center]),
                                        "final_center_population_predicted": float(predicted[-1, center])})
    activation = pd.DataFrame(activation_rows)
    activation.to_csv(RAW/"activation_diagnostics.csv", index=False)
    selected_activation = activation[(activation.model == TARGET) & (activation.seed == 0)].iloc[0]
    calibration_text = table(pd.DataFrame(calibration_rows), ["T", "V0", "side", "edge", "delta", "pass"], ["T", "V₀", "最大侧带占据", "J6 最大边界占据", "J4/J6 完整旋量差", "原门槛"])
    lines = [
        "# 弱周期外势下的耦合谱 LTPB 迁移实验报告", "",
        f"自动分类：**{classification}**。{narratives[classification]}", "",
        f"LTPB 五 seed 平均 spinor error = {mean('spinor_rel_l2_gauge_aligned'):.6g}，current error = {mean('current_rel_l2'):.6g}，mean fidelity = {mean('fidelity_mean'):.6g}。"
        f"逐 seed 五项绝对条件同时通过 {absolute[TARGET]['joint_absolute_pass_count']}/5；对同基底 Complex 的 spinor/current 同时获胜 {decision['paired_joint_win_count']}/5。", "",
        "## 1. 问题、已有工作与本轮范围", "",
        "[已有自由波包报告](WAVEPACKET_SHARED_SPECTRAL_LTPB_REPORT.md)记录了无动量耦合时 Shared-Spectral-LTPB 的成功，"
        "[单动量试验](LEARNED_TWO_PHASE_PILOT_REPORT.md)与[机制消融](LTPB_MECHANISM_ABLATION_REPORT.md)支持能支对齐和独立载波的优化作用。"
        "本轮改变的是 PDE 的耦合结构：初态只占中心动量，外势动态产生原来零占据的侧带。它不是上一轮高斯初谱的直接复跑。", "",
        r"研究对象为 $i\partial_t\psi=[-i\sigma_x\partial_x+m\sigma_z+V_0\cos(qx)I_2]\psi$，"
        r"$x\in[-8\pi,8\pi)$，$m=5,k_0=2,q=1$。谱系数采用 $1/\sqrt L$ 的正交 Fourier 归一化。"
        r"初态中心为 $(u_++u_-)/\sqrt2$，其余模态严格零。", "",
        "## 2. 校准失败、授权修订与参考可靠性", "",
        "原要求的六个候选均无法同时满足耦合强度和 J4/J6 旋量收敛门槛。原始记录没有覆盖，完整结果如下：", "",
        calibration_text, "",
        "用户在正式神经网络训练前明确授权只将截断比较改为 J6/J8。"
        "[修订 01](PERIODIC_POTENTIAL_COUPLED_LTPB_AMENDMENT_01.md)固定 V₀=0.2、T=5、训练 J=6（13 模态），"
        "所有成功阈值、种子和训练预算保留。", "",
        f"修订后的完整复旋量收敛误差为 {selected['truncation_relative_l2']:.10g}（8001 时间点，J6 零填充到 J8）。"
        f"最大侧带占据 {selected_reference['large_ladder_observables']['max_total_occupation_outside_center']:.10g}；"
        f"J6 两端总占据最大值 {selected_reference['large_ladder_observables']['max_outer_layer_total_occupation']:.10g}。", "",
        f"参考 Hamiltonian 是 26×26 块三对角 Hermitian 矩阵。选中 J6 的 Hermiticity 最大误差 "
        f"{selected_reference['large_reference_checks']['hermiticity_max_abs']:.3g}，范数最大漂移 "
        f"{selected_reference['large_reference_checks']['norm_max_drift']:.3g}，时间逆转相对误差 "
        f"{selected_reference['large_reference_checks']['time_reversal_relative_l2']:.3g}；"
        f"全部候选与独立 scipy 矩阵指数的最大差为 {verification['max_expm_relative_l2']:.3g}。", "",
        "本实验训练和评价使用 J6 的有限阶梯；已保存 J6/J8 截断诊断。占据收敛和公共模态切片收敛均不冒充完整旋量收敛。", "",
        "[校准 JSON](results/raw/periodic_potential_coupled_ltpb/calibration/calibration.json) · "
        "[校准 CSV](results/raw/periodic_potential_coupled_ltpb/calibration/calibration_summary.csv) · "
        "[独立验证](results/raw/periodic_potential_coupled_ltpb/reference_verification.json) · "
        "[正式物理选择](results/raw/periodic_potential_coupled_ltpb/formal_selection.json)", "",
        "## 3. 模型、物理先验和公平性", "",
        "本报告中的简称与原始模型 ID 一一对应：", "",
        "| 报告/图表简称 | 原始模型 ID | 表示 |",
        "|---|---|---|",
        "| Physical Vanilla | `physical_vanilla` | 物理空间笛卡尔旋量 |",
        "| Spectral Complex / Complex | `coupled_shared_spectral_complex` | 自由能支基底中的共享复系数 |",
        "| Spectral LTPB / LTPB | `coupled_shared_spectral_ltpb` | 自由能支基底中的共享实振幅和独立相位 |", "",
        r"Physical-Vanilla 以 $\psi_0+\tau\Delta\psi$ 硬编码初值；两个共享谱网络都接收 "
        "$((k-k_0)/(Jq),t/T)$ 并输出四个实数。Complex 使用 "
        r"$c_\pm=c_{\pm,0}+\tau(r_\pm+iq_\pm)$；LTPB 使用 "
        r"$a_\pm=c_{\pm,0}+\tau r_\pm,S_\pm=\tau q_\pm$，"
        r"$\widetilde\psi=\sum_\pm a_\pm u_\pm e^{iS_\pm}$。实振幅允许有符号值。", "",
        "自由 $H(k)$ 的本征子空间是已知表示基底，既不是有外势完整 Hamiltonian 的本征态，也不取消动量耦合。"
        r"训练直接计算 $R_j=i\partial_t\widetilde\psi_j-H(k_j)\widetilde\psi_j"
        r"-(V_0/2)(\widetilde\psi_{j-1}+\widetilde\psi_{j+1})$。精确能量、本征演化相位、参考占据和传播解不进入 learned forward、loss、optimizer 或 checkpoint。", "",
        "三个模型均为 4×64 tanh、float64/complex128、Adam lr=1e-3、5000 steps、seeds 0–4，唯一损失为 PDE 强残差。"
        "谱模型每步 32 个新时间×13 模态=416 点，物理模型同批 32 时间×32 新位置=1024 点。"
        "初始化、时间和位置流分别为 500000+s、600000+s、700000+s；共享谱对的初始权重逐元素相同，时间流哈希逐 seed 相同。"
        "两共享谱模型各 12932 参数，物理基线 12996，差约 0.49%。CPU 每 run 单线程，固定 5 个并发进程，队列无重试。", "",
        r"谱损失为 $\operatorname{mean}_{t,s}\sum_j|R_j|^2/L$。Parseval 把保留 Fourier 投影的残差范数对应到物理空间；"
        "密集物理 PDE 诊断还保留边缘预测向阶梯外生成的两项残差，因此不把截断后的内部残差误认为完整物理残差。", "",
        "## 4. 评价定义与预注册判定", "",
        "Nx=256、Nt=201，物理 x 网格不重复周期端点。主旋量误差仅移除一项全时间/全空间共同的常数相位。"
        "对谱模型，该误差通过 Parseval 等于全保留模态误差；物理模型使用完整场误差，并另报保留阶梯投影误差和阶梯外占据。"
        "不逐模态或逐时间对齐。fidelity 每时刻按预测/参考范数归一化，另报范数漂移，避免 fidelity 掩盖振幅塌缩。", "",
        "仅在参考分支占据 ≥1e-6 时评价通道相干量；成对相干需双方都活跃。"
        "相位自身还需预测占据 ≥1e-12，报告覆盖率，所有 mask 和无效相位 NaN 均保存。"
        "相干量误差不排除预测塌缩为零的通道。", "",
        "绝对成功需要五 seed mean spinor≤0.10、current≤0.15、fidelity≥0.98、modal-population≤0.10、norm drift≤0.02，"
        "且至少 4/5 seeds 各自同时满足这五项。比较优势还需要 target 两项 mean error 均更小且至少 4/5 配对 seeds 两项同时胜出。"
        "Complex 成功而 LTPB 失败才进入 REPRESENTATION_FAILURE；二者共同失败进入 INCONCLUSIVE。", "",
        "[预注册](PERIODIC_POTENTIAL_COUPLED_LTPB_PREREGISTRATION.md) · "
        "[冻结哈希](results/raw/periodic_potential_coupled_ltpb/frozen_protocol_and_code.json) · "
        "[自动判定 JSON](results/raw/periodic_potential_coupled_ltpb/decision.json)", "",
        f"冻结时间（UTC）：`{freeze['frozen_utc']}`；冻结 manifest SHA-256：`{sha(RAW/'frozen_protocol_and_code.json')}`。"
        "科学代码/评价/判据在训练前冻结；输出用绘图和报告脚本另有源文件哈希，不改写原始指标。", "",
        "## 5. 全部 seeds 与统计结果", "",
        "下表为均值 ± population SD（ddof=0），无选择性剔除、重跑或最佳 seed 汇报。", "",
        table(summaries, list(summaries.columns), ["模型", "Spinor", "Current", "Density", "Fidelity", "Modal population", "Norm drift"]), "",
        "### 5.1 逐 seed 主指标", "", table(display, core), "",
        "### 5.2 逐 seed 耦合动力学与占据", "", table(display, dynamics), "",
        "### 5.3 相干、有效相位和残差", "", table(display, aux), "",
        "final_loss_pde 是第 5000 步最后一批随机配点在参数更新前的 PDE loss；dense_physical_pde_residual_rms 是最终参数在统一密集网格上的后评价。两者作用不同，成功判定不使用随机训练 loss。", "",
        "### 5.4 资源与末时刻旋量误差", "", table(display, resources), "",
        "RSS 为进程原生常驻内存峰值，增量以监测开始时为基准；训练时间不含后评价。"
        "时间/内存受固定并发调度和共享机器影响，不能作为跨硬件结论。", "",
        "[逐 run CSV](results/raw/periodic_potential_coupled_ltpb/runs.csv) · "
        "[完整均值/SD CSV](results/raw/periodic_potential_coupled_ltpb/summary.csv) · "
        "[配对差 CSV](results/raw/periodic_potential_coupled_ltpb/paired_comparisons.csv) · "
        "[比值与差值不确定性](results/raw/periodic_potential_coupled_ltpb/comparison_summary.csv)", "",
        "### 5.5 LTPB 对同基底 Complex 的实际优势", "",
        table(ratios, list(ratios.columns)), "", table(paired, list(paired.columns)), "",
        "差值为 LTPB−Complex，负值表示 LTPB 较小；比值不设置额外胜出倍数阈值。"
        "不确定性用五组配对差的 population SD 表示；五个 seeds 不足以支撑一般性显著性断言。", "",
        "## 6. 零初始振幅的预定诊断", "",
        f"固定 seed 0 的参考最大侧带总占据为 {selected_activation['max_outside_population_reference']:.8g}，"
        f"LTPB 仅为 {selected_activation['max_outside_population_predicted']:.8g}；"
        f"末时刻中心占据参考为 {selected_activation['final_center_population_reference']:.8g}，"
        f"LTPB 为 {selected_activation['final_center_population_predicted']:.8g}。"
        "因此其主要问题是新动量侧带被严重低估，接近保留中心模态的动力学，而非仅有一个不可观测全局相位偏差。", "",
        "[全部模型和 seeds 的侧带激活诊断](results/raw/periodic_potential_coupled_ltpb/activation_diagnostics.csv)。"
        f"Complex 五 seed 的最大范数漂移均值为 {runs[runs.model == CONTROL].max_norm_drift.mean():.8g}，说明它在该预算下存在严重振幅塌缩；"
        "这使其不能充当成功的表示能力控制组。", "",
        r"在初始为空的实基底通道，$c(t)=\tau r(t)e^{i\tau q(t)}$。对有限、光滑网络，"
        r"$\partial_t c(0)=r(0)/T$ 必为实数，且相位输出对这一初始虚部导数的参数敏感度严格为零。"
        r"但第一侧带的 PDE 要求 $\partial_t\widetilde\psi_{\pm1}(0)=-(iV_0/2)\widetilde\psi_0(0)$，"
        "在所选实能支基底下具有非零纯虚系数。因此当前硬相位门不能在 t=0 精确匹配该非零虚方向的一阶切线。", "",
        "这证明的是**本参数化在初始端点的 C1 切线限制**，不证明极坐标旋量普遍缺乏表达能力，"
        "也不证明有限时间 L2 必然无法逼近：网络可能通过很窄的初始时间过渡区改善积分误差。"
        "零振幅处物理相位无定义与固定 S(0)=0 后产生的端点切线限制相关，但不应混成同一统计结论。", "",
        table(target, ["seed", "initial_sideband_time_derivative_rel_l2", "initial_sideband_imaginary_tangent_parameter_sensitivity", "active_coherence_rel_l2", "active_branch_phase_coverage"]), "",
        "训练历史另存振幅/实部输出与相位/虚部输出的梯度范数；这些量只作机制诊断，没有加入损失。", "",
        "## 7. 图表证据", "",
        "所有统计图含五个 seeds；所有场图固定 seed 0。图表原始数据、SVG/PDF 与 PNG 同时提供。",
        "主误差图的范数漂移面板采用 symlog（线性阈值 10⁻³），避免将跨零的均值 ± SD 截成极小正数；"
        "负端仅为统计误差棒，不表示负漂移观测。其余五个面板为对数轴。精度–时间图中接近的 seed 点可能重合，完整数值见 source CSV。",
    ]
    for figure in figures:
        identifier = figure["id"]
        lines.extend(["", f"### {identifier}", "", f"![{identifier}]({relative(FIGURES/(identifier+'.png'))})", "",
                      f"[SVG]({relative(FIGURES/(identifier+'.svg'))}) · [PDF]({relative(FIGURES/(identifier+'.pdf'))}) · "+
                      " · ".join(f"[source {i+1}]({relative(FIGURES/source['file'])})" for i, source in enumerate(figure["source_data"])), "", figure["description"]])
    lines.extend(["", "[图证据契约](PERIODIC_POTENTIAL_COUPLED_LTPB_FIGURE_CONTRACT.md) · "
                  "[FIGURE_QA](results/figures/periodic_potential_coupled_ltpb/FIGURE_QA.md)", "",
                  "## 8. 结论边界与下一轮", "", narratives[classification], ""])
    if classification == "REPRESENTATION_FAILURE":
        next_text = "证据支持零初始振幅/固定初相位门造成的退化解释：对照使用相同基底和预算成功，而 LTPB 未达标，并且端点切线与梯度诊断一致。下一轮应先设计并单独预注册混合复数/极坐标表示或其他零通道启动表示，对同一个耦合问题复验后再扩大外势。"
    elif classification == "INCONCLUSIVE":
        next_text = "本轮不能把共同失败归因于 LTPB。零通道端点切线限制得到数学和自动微分证据，但它是否支配整体误差仍未得到成功 Complex 对照的确认。下一轮先检查固定损失尺度、长时优化与共享 k 网络的模式分辨难度；混合复数/极坐标表示可以作为另行预注册的研究支线，目前不宜扩大外势并声称已完成失效归因。"
    else:
        next_text = "该固定弱势实例支持现有参数化的迁移；端点切线限制仍存在，可能被积分误差和有限时间过渡掩盖。下一轮可先以独立预注册扩大势强或演化时间，继续保留同基底 Complex 对照、范数、边界和活跃相干诊断；无需仅凭零振幅这一局部事实就宣称必须改成混合表示。"
    lines.extend([next_text, "",
                  "表示能力、载波预条件和优化难度分别讨论：自由能支对齐能简化无势载波，但有势完整动力学仍耦合；"
                  "公共守恒量可能对相对相位错误不敏感，因此本报告以 spinor/current/coherence 和完整占据轨迹共同判断。"
                  "known k 和自由能支基底是物理先验，不是 learned 模型发现完整 Hamiltonian 本征态的证据。", "",
                  "本轮只涉及已知周期动量阶梯、单中心模态初态和一个经校准弱余弦势；不能外推至未知谱发现、一般外势、非周期波包或通用 Dirac 求解器。"
                  "没有训练任何补救模型，也没有基于正式结果改变学习率、宽度、步数、采样预算或成功阈值。", "",
                  "| 结论 | 直接证据 | 支持程度 |",
                  "|---|---|---|",
                  "| 本配置下未达到迁移成功 | 五种子主指标和冻结 decision.json | 直接支持 |",
                  "| LTPB 数值误差低于同基底 Complex | 五组配对 spinor/current 差值均为负 | 限定于本配置的直接支持，不等于成功迁移 |",
                  "| 零通道初始虚切线受限 | 硬初值公式与自动微分敏感度为零 | 参数化端点性质的直接支持 |",
                  "| 端点退化是整体失败的主因 | Complex 对照也失败，缺乏成功控制组 | 尚未确证 |", "",
                  "## 9. 实现验证、审计和全部原始数据", "",
                  f"测试：{test_status}。训练前已通过 134 项测试；参考验证含 Hermiticity、范数、可逆性、矩阵指数、Parseval 和截断对照。", "",
                  "[审计 JSON](results/raw/periodic_potential_coupled_ltpb/audit.json)核查 15 runs×6 artifacts、固定预算、参数量、随机流、单一全局相位、mask、指标复算、图源和链接。"
                  "[旧结果保护清单](results/raw/periodic_potential_coupled_ltpb/protection/preexisting_results_sha256.csv)覆盖 887 个既有文件。", "",
                  "实现入口：[物理与参考解](src/periodic_potential_physics.py)、[模型](src/periodic_potential_models.py)、"
                  "[训练](src/train_periodic_potential_coupled_ltpb.py)、[评价](src/periodic_potential_evaluate.py)、"
                  "[预注册判定](src/periodic_potential_coupled_ltpb_decision.py)、[正式运行队列](src/launch_periodic_potential_coupled_ltpb.py)。"
                  "正式队列拒绝重复派发，不能直接覆盖已完成结果。", "",
                  "在项目目录复核，不需要重新训练：", "",
                  "```powershell\npython -m pytest -q\npython -m src.audit_periodic_potential_coupled_ltpb\n```", "",
                  "| 模型 | seed | config | metrics | loss | time series | dense grid | checkpoint |", "|---|---:|---|---|---|---|---|---|"])
    for model in PERIODIC_POTENTIAL_MODELS:
        for seed in range(5):
            run = RAW/"learned"/model/f"seed_{seed}"
            lines.append(f"| {LABELS[model]} | {seed} | " + " | ".join(f"[{name}]({relative(run/name)})" for name in RUN_ARTIFACTS)+" |")
    lines.extend(["", "原始参考状态：", ""])
    for path in sorted((RAW/"calibration").glob("*.npz")):
        lines.append(f"- [{path.name}]({relative(path)})")
    lines.extend(["", "运行日志：[派发记录](results/logs/periodic_potential_coupled_ltpb/dispatch.json) · "
                  "[完成记录](results/logs/periodic_potential_coupled_ltpb/completion.json)。", "",
                  "本实验使用控制变量工作流将物理校准和 learned 判据分开冻结；科研作图工作流要求 source CSV、三个导出格式及逐图 PDF 渲染检查；"
                  "科研写作检查将实验结果、局部数学限制和未确证的机制解释分开。"
                  "[输出脚本哈希](results/raw/periodic_potential_coupled_ltpb/reporting_code_sha256.json)与训练前的科学代码冻结清单分开记录。", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    provenance_files = [Path(__file__), ROOT/"src/plot_periodic_potential_coupled_ltpb.py", ROOT/"src/qa_periodic_potential_coupled_ltpb_figures.py", ROOT/"src/audit_periodic_potential_coupled_ltpb.py"]
    write_json(RAW/"reporting_code_sha256.json", {"role": "output-only reporting code; scientific code was separately frozen before training",
                                                  "files": {relative(p): sha(p) for p in provenance_files}})
    return REPORT


if __name__ == "__main__":
    print(build_report())
