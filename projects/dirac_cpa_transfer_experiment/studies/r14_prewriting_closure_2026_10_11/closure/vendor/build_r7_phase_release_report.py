from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate_r7_phase_release import stage_runs, summary_table
from .freeze_r7_phase_release import ROOT, RAW, DOCS, sha256, verify_freeze, verify_history, write_json
from .r7_config import EXPERIMENT, MODELS, TRAINING_MODES
from .r7_decision import decide


FIGURES = ROOT/"results/figures"/EXPERIMENT
LOGS = ROOT/"results/logs"/EXPERIMENT
NAMES = dict(complex="Complex", original_ltpb="原 LTPB", phase_release_ltpb="相位释放 LTPB")
MODES = dict(ordinary="普通", causal="因果")


def link(path, label):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def number(value):
    if value is None or pd.isna(value): return "—"
    value = float(value)
    return f"{value:.5g}"


def table(headers, rows):
    lines = ["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]
    lines += ["| "+" | ".join(str(v) for v in row)+" |" for row in rows]
    return "\n".join(lines)


def summary_rows(runs, keys):
    rows = []
    for mode in TRAINING_MODES:
        for model in MODELS:
            part = runs[(runs.model == model)&(runs.training_mode == mode)]
            if part.empty: continue
            row = [NAMES[model], MODES[mode]]
            row += [number(part[key].mean())+" ± "+number(part[key].std(ddof=0)) for key in keys]
            rows.append(row)
    return rows


def run_rows(runs, keys):
    rows = []
    for _, entry in runs.sort_values(["training_mode", "model", "seed"]).iterrows():
        rows.append([NAMES[entry.model], MODES[entry.training_mode], int(entry.seed)]+[number(entry[key]) for key in keys])
    return rows


def tests_note():
    path = LOGS/"final_pytest.xml"
    if not path.exists(): path = LOGS/"pre_freeze_pytest.xml"
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    count = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failed = sum(int(s.attrib.get("failures", 0))+int(s.attrib.get("errors", 0)) for s in suites)
    return f"完整项目 pytest：{count} 项，失败/错误 {failed}；记录："+link(path, path.name)+"。"


def build_report():
    main, short = stage_runs("main"), stage_runs("short")
    if len(main) != 30 or len(short) != 9:
        raise ValueError("Report requires all 30 main and 9 short runs; partial results cannot support conclusions")
    decision = decide(main)
    reference = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    long = stage_runs("long")
    if decision["long_training_triggered_by_main"] and reference["long_reference_pass"] and len(long) != 15:
        raise ValueError("Triggered long experiment must finish before the final report")
    freeze = verify_freeze()
    history = verify_history()
    audit_path = RAW/"audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"status": "PENDING"}
    diagnostics = json.loads((RAW/"posthoc_failure_diagnostics.json").read_text(encoding="utf-8"))
    diagnostic_frame = pd.DataFrame(diagnostics["main_run_diagnostics"])
    surrogate = diagnostics["central_free_surrogate"]
    absolute = decision["absolute_results"]
    release_pass = [mode for mode in TRAINING_MODES if absolute[f"phase_release_ltpb/{mode}"]["pass"]]
    any_control = [f"{NAMES[m]}·{MODES[mode]}" for m in ("complex", "original_ltpb") for mode in TRAINING_MODES
                   if absolute[f"{m}/{mode}"]["pass"]]
    result_line = ("相位释放 LTPB 在"+"、".join(MODES[m] for m in release_pass)+"训练下通过完整动力学绝对判据。"
                   if release_pass else "相位释放 LTPB 在两种训练方式下均未通过完整动力学绝对判据。")
    sections = ["# R7：空通道相位释放与因果训练交叉实验报告", "",
                "实验：`coupled_ltpb_phase_release_r7`；执行日期：2026-10-01。", "",
                "## 1. 结论先读", "",
                "原模型的空通道一阶虚部切向限制已在数学和自动微分层面解除。"+result_line,
                f"自动分类：**{decision['classification']}**。它是本轮有限自由能支基底、弱外势耦合问题和固定训练预算的结果，不是通用 Dirac 求解器结论。",
                "修正版普通/因果五种子平均 spinor 分别为 "+number(main[(main.model == "phase_release_ltpb")&(main.training_mode == "ordinary")].spinor_rel_l2_gauge_aligned.mean())+
                "/"+number(main[(main.model == "phase_release_ltpb")&(main.training_mode == "causal")].spinor_rel_l2_gauge_aligned.mean())+
                "，current 为 "+number(main[(main.model == "phase_release_ltpb")&(main.training_mode == "ordinary")].current_rel_l2.mean())+
                "/"+number(main[(main.model == "phase_release_ltpb")&(main.training_mode == "causal")].current_rel_l2.mean())+"；完整统计见第7节。",
                "工程完整性与科学成功分开判定；全部 9 次短时和 30 次主实验均保留，主要结论固定采用 step 5000。",
                "",
                "## 2. 历史证据与本轮问题", "",
                "R1–R5 的自由问题显示：公共相位在单正能支载波预条件良好，混合能支时优势退化；共享谱 LTPB 在已知 Fourier 表示的自由波包上通过既定检验。它们不证明未知谱发现或外势耦合也能成功。",
                "R6 首次加入弱周期外势，两种共享谱表示都未达绝对成功；原 LTPB 五种子均值 spinor≈0.3667、current≈0.4561，正式为 INCONCLUSIVE。发现初始空通道虚部导数被硬相位门抑制，是局部结构证据，尚非全程失败的充分解释。",
                "本轮仅修改空初值通道的相位门，另做表示×训练的 3×2 消融。旧 R6 不重写、不覆盖、不拿旧随机采样结果替代本轮普通组。",
                "历史入口："+link(ROOT/"docs/comprehensive/report.md", "全课题报告")+"；"+
                link(ROOT/"docs/experiments/periodic_potential_coupled_ltpb/report.md", "R6 外势报告")+"；"+
                link(ROOT/"docs/experiments/wavepacket_shared_spectral_ltpb/report.md", "自由共享谱波包报告")+"。",
                "",
                "## 3. 方程、参考解与冻结", "",
                r"$$i\partial_t\psi=[-i\sigma_x\partial_x+m\sigma_z+V_0\cos(qx)I_2]\psi,$$",
                r"$$\psi=L^{-1/2}\sum_{j=-J}^{J}e^{i(k_0+jq)x}\widetilde\psi_j,\quad R_j=i\partial_t\widehat{\widetilde\psi}_j-H(k_j)\widehat{\widetilde\psi}_j-\frac{V_0}{2}(\widehat{\widetilde\psi}_{j-1}+\widehat{\widetilde\psi}_{j+1}),$$",
                r"$$H(k)=k\sigma_x+m\sigma_z,\qquad \widetilde\psi_0(0)=(u_+(k_0)+u_-(k_0))/\sqrt2;\quad\widetilde\psi_{j\ne0}(0)=0.$ $".replace(".$ $", "$$"),
                "主问题 L=16π，m=5，k₀=2，q=1，V₀=0.2，T=5，J=6（13 个模态，k∈[−4,8]）。端点外的系数在截断方程中为零，完整物理残差另计相邻截断外泄漏。",
                "采用块三对角 Hermitian Hamiltonian 的矩阵本征演化作参考，并独立与矩阵指数核对。沿用用户已批准的 J=6/8 收敛条件；不重新套用旧 J=4/6 校准，也未按 PINN 结果选物理参数。",
                table(["参考量", "T=5，J=6/8", "预验证 T=8，J=8/10"],
                      [[label, number(reference["main"][group][key]), number(reference["long_trials"][0][group][key])]
                       for label, group, key in [
                           ("最大中心外占据", "observables", "max_total_occupation_outside_center"),
                           ("最大最外层总占据", "observables", "max_outer_layer_total_occupation"),
                           ("零填充全复轨迹 relative L2", "comparison", "padded_full_trajectory_spinor_relative_l2"),
                           ("参考最大范数漂移", "checks", "norm_max_drift"),
                           ("时间可逆 relative L2", "checks", "time_reversal_relative_l2"),
                           ("Hermiticity max abs", "checks", "hermiticity_max_abs")]]),
                "两项参考各含 8001 时间点；T=8 按预设 J=8,10,12 顺序首个合格截断为 J=8。参考矩阵指数最大绝对差：T=5 "+number(reference["main"]["expm_max_absolute_error"])+"，T=8 "+number(reference["long_trials"][0]["expm_max_absolute_error"])+"。",
                "参考和全部训练后的评价代码与 learned forward/loss 隔离。允许使用自由本征方向和初值；不输入精确能量、时间载波、未来占据或参考轨迹。自由基底不是有外势的完整 Hamiltonian 本征态。",
                "预注册："+link(DOCS/"preregistration.md", "冻结协议")+"；"+link(DOCS/"figure_contract.md", "图表证据契约")+"；"+link(RAW/"reference_validation.json", "参考验证")+"。",
                "冻结清单 SHA-256：`"+freeze+"`。正式科学代码与依赖、测试、协议逐文件哈希保存在 "+link(RAW/"frozen_protocol_and_code.json", "冻结清单")+"。",
                "",
                "## 4. 表示与理论核查", "",
                r"令 $\tau=t/T_{gate}$。共享网络 $(k_{norm},\tau)\mapsto(r_+,r_-,b_+,b_-)$，通道旋量为 $c_+u_++c_-u_-$。三个模型为：", "",
                table(["表示", "每个分支的复系数", "支持判断"], [
                    ["Complex", r"$c=c_0+\tau(r+ib)$", "不做相位分离"],
                    ["原 LTPB", r"$c=(c_0+\tau r)e^{i\tau b}$", "所有通道相位在初时为零"],
                    ["相位释放 LTPB", r"已占据同原式；空初值 $c=\tau r e^{ib}$", "只依据精确初值是否为零；不动态切换"]]),
                "振幅为有符号实数；未增加混合复包络、幅值阈值正则化、精确载波或监督导数损失。全部通道初时非零的自由回归算例，在同权重下修正前后输出及时间导数逐元素一致。",
                r"对空通道，$c'_{old}(0)=r(0)/T_{gate}\in\mathbb R$，而 $c'_{release}(0)=r(0)e^{ib(0)}/T_{gate}$。PDE 要求 $\widetilde\psi'_{\pm1}(0)=-i(V_0/2)\widetilde\psi_0(0)$，投影到固定实基底后也是纯虚导数。",
                r"$$D_{(r,b)}(\Re c',\Im c')=\frac1{T_{gate}}\begin{pmatrix}\cos b&-r\sin b\\\sin b&r\cos b\end{pmatrix},\quad\det D=r/T_{gate}^2.$ $".replace(".$ $", "$$"),
                "修正版 r≠0 时输出坐标 Jacobian 秩 2；r=0 仍秩 1。参数自动微分证明原式初始虚部切向敏感度严格为零，修正版可非零；这不是共享网络在所有点条件良好、实际导数学准或长期动力学成功的证明。",
                r"令 $e=\widehat\psi-\psi$，常数 Hermitian 截断系统满足 Duhamel 估计：",
                r"$$\|e(t)\|_2\le\|e(0)\|_2+\int_0^t\|R(s)\|_2\,ds.$ $".replace(".$ $", "$$"),
                "估计仅用 raw 场和未加权的连续时间谱残差范数；不能把加权训练 loss 直接当成误差证书。本轮用 4001/2001 嵌套网格检验数值积分，仍不是严格连续认证。截断外泄漏另报。端点导数不匹配本身不能推出全时间 L2 误差正下界，因为可形成很窄的初始层。",
                "完整符号、单位、假设及推导："+link(DOCS/"theory_check.md", "理论核查")+"；自动验证："+link(RAW/"theory_validation.json", "局部与梯度检验")+"。",
                "",
                "## 5. 公平训练、评价与预注册判据", "",
                table(["项目", "固定设置"], [
                    ["网络/参数", "4×64 tanh，四个实输出，每种表示 12932 参数"],
                    ["精度/优化", "float64/complex128；Adam 1e−3；不早停，不调整超参"],
                    ["主预算", "3 表示×2 训练方式×5 seeds=30 runs；每个 5000 steps"],
                    ["样本", "[0,5] 八段，每段 4 个随机时间；32×13=416 完整耦合谱残差点/step"],
                    ["配对随机流", "初始化 seed=500000+s；独立时间 seed=600000+s；同 seed 六配置权重/样本序列一致"],
                    ["检查点", "固定 1000、2500、5000；仅 5000 判主结果；每 250 steps 可精确恢复状态"],
                    ["评价网格", "Nx=256,Nt=201；独立早期小正时间网格与 4001 点积分网格"],
                    ["执行/资源", "3 独立 CPU 核并发，每进程单线程，核心按 seed 轮换；记录实际训练 wall time 和 peak RSS"]]),
                "普通损失与 R6 连续目标相同，但采样改为分层，故普通组本轮全部重新训练。训练时间统计优化步骤，不含检查点评价/落盘；run_elapsed_seconds_this_session 另含这些成本。并发仍共享缓存/内存，时间只作本轮实测描述，不声称消除所有硬件偏差。",
                "因果工具来源：[Wang、Sankaran、Perdikaris，2022](https://arxiv.org/abs/2203.07404)，采用累计先前残差的指数权重和 stop-gradient；本轮固定分段、尺度、日程，无原论文权重触发早停，不声称逐项复现或本工具原创。",
                r"$$\ell_i=\operatorname{mean}_{t\in I_i,s}\sum_j |R_{j,s}|^2/L,\quad\ell_{scale}=m^2/(2L),\quad w_i=\exp[-\epsilon\sum_{h<i}\operatorname{stopgrad}(\ell_h/\ell_{scale})],\quad\mathcal L=\tfrac18\sum_iw_i\ell_i.$ $".replace(".$ $", "$$"),
                "w₁=1；ε=0.01、0.1、1、10、100 各固定 1000 steps。普通组 wᵢ=1，两组时间点完全相同。训练同时保存各段未加权残差及权重，不能只看加权 loss。",
                "旋量主指标最多消除一个全时空常数相位；同时保留 raw 误差。密度 ρ=ψ†ψ，概率流 j=ψ†σₓψ，模态/分支占据由总预测场投影。有效相位要求参考占据≥10⁻⁶且预测占据≥10⁻¹²，保存 mask/覆盖率；占据、相干误差不会因预测塌缩而剔除参考活跃通道。相干量用 cₐcᵦ*，天然不含整体相位。",
                "绝对成功：五种子均值 spinor≤0.10、current≤0.15、mean fidelity≥0.98、模态占据误差≤0.10、最大范数漂移≤0.02，且≥4/5 seeds 同时满足全部五项。相对优势必须先达到绝对成功，再要求两项核心均值都低于同训练方式对照、且≥4/5 配对双赢。所有 SD 为 population SD（ddof=0）；不做总体显著性声明。",
                "",
                "## 6. 短时诊断：9 次全部结果", "",
                "区间 [0,0.5]，seeds0–2，普通 PDE，1000 steps。时间输入和硬门控仍为 t/5；没有借缩短区间改变参数尺度。smoke 仅验证维度、梯度、NaN 和保存；短时结果未用于调参或筛选种子。",
                table(["模型", "训练", "seed", "spinor", "初始侧带导数误差", "虚部切向参数敏感度", "活跃相干误差"],
                      run_rows(short, ["spinor_rel_l2_gauge_aligned", "initial_sideband_time_derivative_rel_l2",
                                       "initial_sideband_imaginary_tangent_parameter_sensitivity", "active_coherence_rel_l2"])),
                table(["模型", "训练", "spinor 均值 ± SD", "初始导数误差均值 ± SD", "虚部切向敏感度均值 ± SD", "相干误差均值 ± SD"],
                      summary_rows(short, ["spinor_rel_l2_gauge_aligned", "initial_sideband_time_derivative_rel_l2",
                                           "initial_sideband_imaginary_tangent_parameter_sensitivity", "active_coherence_rel_l2"])),
                "敏感度>0 表示存在可优化方向，不等于已学出正确导数。短时结果保留修正版表现不好的种子：不能用个别改善宣称初始层问题已整体解决。零占据点不解释普通相位。",
                "",
                "## 7. 主实验：30 次完整结果与统计", "",
                "### 7.1 主要指标均值 ± population SD", "",
                table(["模型", "训练", "spinor aligned", "current", "fidelity mean", "模态占据", "范数最大漂移"],
                      summary_rows(main, ["spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean", "modal_population_rel_l2", "max_norm_drift"])),
                table(["配置", "绝对判据", "逐 seed 联合合格数"],
                      [[f"{NAMES[m]}·{MODES[mode]}", "PASS" if absolute[f"{m}/{mode}"]["pass"] else "FAIL",
                        f"{absolute[f'{m}/{mode}']['joint_absolute_pass_count']}/5"] for mode in TRAINING_MODES for m in MODELS]),
                "### 7.2 辅助精度与动力学", "",
                table(["模型", "训练", "raw spinor", "final aligned", "density", "minimum fidelity", "分支占据", "活跃相干"],
                      summary_rows(main, ["spinor_rel_l2_raw", "final_spinor_rel_l2_gauge_aligned", "density_rel_l2", "fidelity_min", "branch_population_rel_l2", "active_coherence_rel_l2"])),
                table(["模型", "训练", "中心占据误差", "下侧带误差", "上侧带误差", "最大中心外占据", "相位覆盖率", "有效相位 RMSE/rad"],
                      summary_rows(main, ["center_occupation_rel_l2", "lower_sideband_occupation_rel_l2", "upper_sideband_occupation_rel_l2",
                                         "max_outside_center_occupation", "active_branch_phase_coverage", "active_branch_phase_circular_rmse"])),
                "参考最大中心外占据为 "+number(reference["main"]["observables"]["max_total_occupation_outside_center"])+
                "。高相位覆盖率仅表示预测幅度超过极小定义阈值，不表示占据、相位或相干准确；侧带相对占据误差接近 1 表示几乎未重建，而非正常传播。",
                "### 7.3 所有种子的主要原始值", "",
                table(["模型", "训练", "seed", "spinor aligned", "current", "fidelity", "模态占据", "范数漂移"],
                      run_rows(main, ["spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean", "modal_population_rel_l2", "max_norm_drift"])),
                "### 7.4 所有种子的局部与残差诊断", "",
                table(["模型", "训练", "seed", "初始导数误差", "虚部切向敏感度", "中心外最大占据", "fine 未加权 PDE", "残差积分", "参数坐标最小奇异值"],
                      run_rows(main, ["initial_sideband_time_derivative_rel_l2", "initial_sideband_imaginary_tangent_parameter_sensitivity",
                                      "max_outside_center_occupation", "fine_unweighted_pde_loss", "residual_integral_fine", "initial_output_tangent_jacobian_min_singular_value"])),
                r"最后列取初始两侧带共四个输出坐标 Jacobian 的最小奇异值；非零但很小仍可能病态。参数敏感度具体为 $\|\nabla_\theta\sum_{j=\pm1,s}\Im\widehat{\widetilde\psi}'_{j,s}(0)\|_2$，不是整块参数Jacobian的最小奇异值或物理相位误差。完整全部指标见 "+link(RAW/"main_runs.csv", "逐运行 CSV")+"。",
                "",
                "## 8. 表示、因果训练与交互作用", "",
                "### 8.1 同训练方式的表示效应", "",
                table(["比较", "spinor 均值比", "current 均值比", "spinor 配对差 ± SD", "current 配对差 ± SD", "双赢 seeds", "优势判据"],
                      [[key, number(value["summary"]["spinor_rel_l2_gauge_aligned"]["ratio_of_means"]),
                        number(value["summary"]["current_rel_l2"]["ratio_of_means"]),
                        number(value["summary"]["spinor_rel_l2_gauge_aligned"]["paired_difference_mean"])+" ± "+number(value["summary"]["spinor_rel_l2_gauge_aligned"]["paired_difference_population_sd"]),
                        number(value["summary"]["current_rel_l2"]["paired_difference_mean"])+" ± "+number(value["summary"]["current_rel_l2"]["paired_difference_population_sd"]),
                        f"{value['paired_joint_win_count']}/5", "PASS" if value["comparative_advantage"] else "FAIL"]
                       for key, value in decision["comparisons"].items()]),
                "比值为修正版/对照的均值之比；差值为修正版−对照。相对低误差或 5/5 双赢但未达绝对成功，仍不能写成解决完整动力学。",
                "### 8.2 同表示的训练效应", "",
                table(["表示", "spinor 因果/普通", "current 因果/普通", "spinor 配对差 ± SD", "current 配对差 ± SD", "两项同时变好 seeds"],
                      [[NAMES[key], number(value["summary"]["spinor_rel_l2_gauge_aligned"]["ratio_of_means"]),
                        number(value["summary"]["current_rel_l2"]["ratio_of_means"]),
                        number(value["summary"]["spinor_rel_l2_gauge_aligned"]["paired_difference_mean"])+" ± "+number(value["summary"]["spinor_rel_l2_gauge_aligned"]["paired_difference_population_sd"]),
                        number(value["summary"]["current_rel_l2"]["paired_difference_mean"])+" ± "+number(value["summary"]["current_rel_l2"]["paired_difference_population_sd"]),
                        f"{value['paired_joint_win_count']}/5"] for key, value in decision["causal_minus_ordinary"].items()]),
                "训练效应为因果−普通；负值只表示本轮配对误差减少。因果训练是已有优化工具，不能将其收益全部归给表示修正。",
                "### 8.3 两因素交互", "",
                "交互=(修正版因果−原式因果)−(修正版普通−原式普通)。负值表示因果训练下相位修正更有利；这里只做五个配对种子的描述，不声称总体统计显著性。",
                table(["指标", "seed0", "seed1", "seed2", "seed3", "seed4", "均值 ± population SD"],
                      [[key]+[number(v) for v in val["per_seed"]]+[number(val["mean"])+" ± "+number(val["population_sd"])]
                       for key, val in decision["representation_training_interaction"].items()]),
                "全部逐种子差值/比值："+link(RAW/"paired_comparisons.csv", "配对表")+"；"+link(RAW/"paired_summary.csv", "配对均值与 SD")+"；"+link(RAW/"factor_interactions.csv", "交互原始值")+"。",
                "",
                "## 9. 检查点、未加权残差与成本", "",
                "1000、2500、5000 三点全部评价，没有按真解误差选最好的 checkpoint；中间结果仅用于解释优化轨迹。完整 90 行见 "+link(RAW/"checkpoint_metrics.csv", "固定检查点 CSV")+"。",
                table(["模型", "训练", "训练秒", "peak RSS/MB", "final 加权 loss", "fine 未加权 PDE", "外层泄漏 max norm"],
                      summary_rows(main, ["training_seconds", "peak_rss_mb", "final_weighted_objective", "fine_unweighted_pde_loss", "exterior_residual_leakage_max_l2_norm"])),
                "每主 run 恰为 2,080,000 个训练残差点、5000 个优化步骤；检查点和加密诊断的额外残差评价单独保存，不计入训练采样预算。不同表达与权重计算成本实测，步数相同不等于 wall time 相同。",
                "所有主 run 4001/2001 点积分相对差最大值为 "+number(main.residual_integral_quadrature_relative_difference.max())+
                "；raw 误差超过数值估计的最大量为 "+number(main.residual_estimate_max_numerical_violation.max())+
                "。积分界若约 O(1)，即使训练 loss 约 10⁻⁴，也无法证明要求的 0.10 旋量精度。",
                "CPU 核绑定、依赖版本、运行环境、权重/时间流哈希均随每个 config 保存。execution_events 中 start/resume/complete 原样保存；metrics 的 interruption_and_resume_events 字段计启动/恢复事件，不应把单次 start 解读为发生一次中断。",
                "### 9.1 失败解释诊断：未耦合中心模态吸引域", "",
                "此项在查看主结果后增加，仅解释失败，不更改冻结判据、不训练新模型、不参加 learned 排名。构造只含中心模态的精确自由演化（令外势作用被忽略），并与完整耦合参考和各预测场比较。此诊断可使用自由矩阵本征演化，因为它不进入任何 learned forward、loss、optimizer 或checkpoint。",
                r"该反事实场范数严格守恒，但一阶侧带残差为 $R_{\pm1}=-(V_0/2)\widetilde\psi_0$，于是未加权目标恰为 $V_0^2/(4L)$。",
                table(["未耦合中心诊断", "数值"], [[label, number(surrogate[key])] for label, key in
                      [("spinor aligned", "spinor_rel_l2_gauge_aligned"), ("current relative L2", "current_rel_l2"),
                       ("density relative L2", "density_rel_l2"), ("mean fidelity", "fidelity_mean"),
                       ("最大范数漂移", "max_norm_drift"), ("未加权残差 loss（解析）", "expected_unweighted_pde_loss"),
                       ("中心外占据", "outside_center_occupation")]]),
                table(["模型", "训练", "到未耦合中心场 raw 距离", "残差中一阶侧带占比", "相位坐标切向 min norm", "最后 I8 权重", "最后未加权/采样 loss"],
                      summary_rows(diagnostic_frame, ["predicted_raw_distance_to_uncoupled_center", "first_sideband_fraction_of_unweighted_residual",
                                                     "initial_phase_coordinate_tangent_norm_min", "last_eighth_bin_weight", "last_raw_training_loss"])),
                "若预测接近这一场、残差主要落在一阶侧带，支持‘学会中心载波但未学会耦合转移’的解释；小范数漂移和小归一化残差不是侧带成功证据。释放相位后的相位坐标切向仍可能因振幅很小而病态。此诊断不是唯一因果证明；Complex失败可能有不同的优化轨迹，不能用它证明全部失败都来自同一结构。",
                "全部逐run后验解释量："+link(RAW/"posthoc_failure_diagnostics.csv", "诊断 CSV")+"；"+link(RAW/"posthoc_failure_diagnostics.json", "诊断与边界 JSON")+"。",
                "",
                "## 10. 较长时间条件实验", ""]
    if len(long) == 15:
        long_decision = json.loads((RAW/"long_decision.json").read_text(encoding="utf-8"))
        sections += ["主实验触发了附加验证，按预注册选择 "+MODES[decision["long_training_mode_selected"]]+
                     "训练。T=8、V₀=0.2、J=8，从头训练三个模型×五种子、每个5000steps，输入及门控为 t/8。这不是训练区间外推。",
                     table(["模型", "训练", "spinor", "current", "fidelity", "模态占据", "范数漂移"],
                           summary_rows(long, ["spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean", "modal_population_rel_l2", "max_norm_drift"])),
                     table(["模型", "训练", "seed", "spinor", "current", "fidelity", "模态占据", "范数漂移"],
                           run_rows(long, ["spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean", "modal_population_rel_l2", "max_norm_drift"])),
                     "较长区间修正版绝对判据："+("PASS" if long_decision["release_absolute_success"] else "FAIL")+
                     "；完整数据："+link(RAW/"long_runs.csv", "长时间逐 run")+"、"+link(RAW/"long_summary.csv", "长时间汇总")+"、"+link(RAW/"long_decision.json", "长时间判定")+"。"]
    else:
        sections += ["T=8 参考预验证已通过，但主实验修正版没有达到任一训练方式的绝对成功条件，故依预注册**不启动**15次附加训练。不能写成‘T=8 learned 模型验证失败’或‘已通过’，也不以延长预算来寻找有利结果。"]
    sections += ["", "## 11. 图表证据", "",
                 "统计图包括全部五个配对种子（短时图为全部三个短时种子），误差棒为 population SD；动力学实例固定 seed0。每图均有 PNG、可编辑 SVG、PDF 和源 CSV。"]
    manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
    captions = dict(all_seed_primary_metrics="六配置主指标与绝对门槛：相对改善不等于成功。",
                    checkpoint_accuracy="全部种子的固定检查点轨迹：不择优选 checkpoint。",
                    initial_tangent_and_short_dynamics="局部敏感度、导数误差和早期侧带的区别；短时所有种子保留。",
                    main_initial_tangent="正式五种子的初始切向、所学导数、振幅/相位输出梯度及病态性。",
                    modal_occupation_heatmaps="中心外模态是否真的产生；共用占据尺度，固定 seed0。",
                    sideband_trajectories="中心和两侧带占据对照；零振幅相位不作普通物理解释。",
                    coherence_current="活跃相干误差与预测相位覆盖率，再以物理概率流独立验证。",
                    density_current_fields="完整空间密度和概率流的参考/预测/误差：单位置时间曲线不代表全场精度。",
                    residual_weights="因果权重、未加权段残差与 raw 误差积分界，避免小加权 loss 的误导。",
                    accuracy_cost="全部种子的精度、实测训练时间、RSS及损失轨迹。",
                    factor_interactions="表示×训练交互的五个配对原始值；不做显著性推断。",
                    long_time_validation="预注册条件触发的 T=8 从头训练结果。")
    for n, fig in enumerate(manifest["figures"], 1):
        name = fig["figure_id"]
        sections += ["", f"图{n}："+captions.get(name, fig["caption"]),
                     f"![图{n}](<{(FIGURES/(name+'.png')).as_posix()}>)", "",
                     link(FIGURES/(name+".svg"), "SVG")+" · "+link(FIGURES/(name+".pdf"), "PDF")+" · "+
                     "；".join(link(ROOT/src["path"], "源数据 "+Path(src["path"]).stem) for src in fig["source_data"])]
    sections += ["", "图表 QA："+link(FIGURES/"FIGURE_QA.md", "导出/源数据检查")+"；"+link(FIGURES/"visual_review.md", "目视审查")+"。",
                 "", "## 12. 六个问题与结论边界", "",
                 "1. **原硬相位缺陷是否解除？** 是，限定在初始空通道的一阶虚部切向：允许非零，r≠0 输出坐标秩2；零幅退化、共享参数条件数及高阶动力学并未因此消失。",
                 "2. **是否恢复侧带和相干动力学？** "+result_line+" 必须同时看侧带占据、相干、旋量和流；总范数良好或相位覆盖率高不能替代这些证据。",
                 "3. **表示、因果及交互贡献？** 上述配对表分别给出同训练方式表示差、同表示训练差和差中之差；没有把因果工具收益算作相位方法原创或普遍收益。",
                 "4. **有绝对成功对照吗？** "+("有："+"、".join(any_control)+"。" if any_control else "Complex 和原 LTPB 均未达到本轮绝对判据。")+
                 " 修正版相对优势必须先有自身绝对成功，再看同训练方式均值和≥4/5双赢；对应结果见8.1，不用任意倍数或个别种子替代判据。",
                 "5. **较长时间与当前主张？** "+("已按预设条件实施，完整结果见第10节；通过仅限该T=8耦合谱问题。" if len(long) else "未触发 learned T=8 训练；当前只有参考预验证，不能提出长时间求解成功主张。"),
                 "6. **下一轮方向？** "+("在保持当前消融证据的基础上，可另行预注册少量稀疏观测约束，并保留无数据对照；不能把本轮纯PDE成功外推为反演成功。" if release_pass else
                     "优先完善耦合动力学的表示/优化与残差尺度诊断，再独立预注册；暂不直接进入稀疏数据约束，否则会混淆纯PDE学习不足与数据补偿效果。"),
                 "",
                 "本轮没有训练额外补救结构。若共享 Complex 与修正版共同失败，支持固定预算下共同优化/表示条件性尚未解决，而非单凭零初值就证明 LTPB 表达能力不可能。若修正版失败但同基底 Complex 成功，才构成更强的表示退化对照。",
                 "严格区别：**表达能力**是存在参数表示目标；**载波预条件**是将快速相位显式分离；**优化难度**是固定随机初值和预算能否找到目标；**守恒盲点**是小范数漂移不识别侧带/相干错误；**物理先验**是已知有限 Fourier 梯级及自由能支方向，不是已知完整外势动力学。",
                 "五种子不是总体显著性证明。仅检验已知初始中心模态、单弱周期势、固定m/L/V₀、有限Hermitian截断；未检验未知谱、通用外势、真实数据、反演、KG或束缚态。不声称首次；最小相位释放的原创性需另行针对性文献核查。",
                 "",
                 "## 13. 实现、测试、审计与完整产物", "",
                 "理论核查与控制变量技能促使本轮分别检验局部切向/全局动力学、统一配对与停止规则；科学作图与PDF检查技能落实源表、全种子、固定seed0、可编辑导出和实际渲染目视QA；学术写作技能按研究问题—证据—边界闭环组织本中文报告。技能未改变用户阈值或训练预算。",
                 tests_note(),
                 "新增测试涵盖：严格硬初值、类型/参数量/配对初始化、原/修正版初始导数、坐标Jacobian秩、参数敏感度、全已占据自由回归、耦合残差、stopgrad/ε日程、参考隔离、独立采样流、恢复Adam及全部RNG、判定完整性。",
                 "实际正式优化步骤 159000（9000短时+150000主），另有工程smoke120步；全部39正式run均一次启动完成，没有中断、挑选性重启或失败seed额外预算。冻结后仅修复后处理的UTF-8读取/布尔JSON序列化和图表可读性，详见 "+link(LOGS/"postprocessing_notes.md", "后处理修复记录")+"。",
                 f"历史哈希：{history['checked_files']} 个既有文件核对，改动/丢失 {len(history['changed_or_missing'])}。本目录无Git仓库；没有创建提交、回滚或覆盖旧结果。保护清单："+link(RAW/"protected_history.json", "历史保护清单")+"。",
                 "自动工程审计："+str(audit["status"])+"，记录 "+link(audit_path, "audit.json")+"。审计包括独立重算指标/mask、Fourier重建、硬初值、数值残差估计、配对权重和时间流、预算/检查点、冻结哈希、历史完整性及报告链接；工程PASS不意味着科学PASS。",
                 "入口："+link(ROOT/"src/run_r7_phase_release.py", "独立运行")+"；"+link(ROOT/"src/aggregate_r7_phase_release.py", "聚合")+"；"+link(ROOT/"src/plot_r7_phase_release.py", "作图")+"；"+link(ROOT/"src/audit_r7_phase_release.py", "审计")+"；"+link(ROOT/"src/build_r7_phase_release_report.py", "报告生成")+"。",
                 "统计和判定："+link(RAW/"short_runs.csv", "短时逐run")+" · "+link(RAW/"short_summary.csv", "短时汇总")+" · "+link(RAW/"main_runs.csv", "主逐run")+" · "+link(RAW/"main_summary.csv", "主汇总")+" · "+link(RAW/"decision.json", "自动科学判定")+" · "+link(RAW/"reporting_code_sha256.json", "报告/审计代码哈希")+"。",
                 "日志/实际命令："+link(LOGS, "日志目录")+"；每run独立 execution_events.jsonl 保留启动/恢复/完成记录，recovery_state.pt 另含优化器、采样及全RNG供精确恢复。",
                 "",
                 "### 13.1 每个运行的原始证据链接", "",
                 "下表六件主产物逐run齐备。每run另含 early_time_series.csv、residual_integral.csv、residual_by_segment_mode.csv；主/长run另含固定检查点目录。", ""]
    rows = []
    for stage in ("short", "main", "long"):
        for path in sorted((RAW/stage).glob("*/*/seed_*/metrics.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            directory = path.parent
            row = [f"{stage}/{run['model']}/{run['training_mode']}/s{run['seed']}"]
            row += [link(directory/name, label) for name, label in [("config.json", "config"), ("metrics.json", "metrics"),
                        ("loss_history.csv", "loss"), ("time_series.csv", "series"), ("dense_grid.npz", "dense"), ("model_state.pt", "model")]]
            row += [link(directory/"checkpoints", "固定检查点") if stage != "short" else "—"]
            rows.append(row)
    sections += [table(["运行", "配置", "指标", "损失", "时间序列", "密集场", "权重", "检查点"], rows), "",
                 "### 13.2 可复现入口", "",
                 "在项目根目录执行独立入口的 prepare、smoke、freeze、short、main、聚合、long、聚合；long 依据冻结判定自动触发/跳过。重复调用已完整的run仅验证并返回，不覆盖；未完成run从精确恢复状态继续，不挑选性重启。实际命令已经逐子进程落盘，源版本以冻结清单为准。最后完整pytest、作图、报告、审计并复核报告；报告/图表格式代码与训练冻结代码分开记录哈希。", ""]
    reporting_paths = [ROOT/"src"/name for name in ("aggregate_r7_phase_release.py", "plot_r7_phase_release.py",
                       "audit_r7_phase_release.py", "build_r7_phase_release_report.py", "r7_report_qa.py", "r7_failure_diagnostics.py")]
    reporting_paths += sorted((ROOT/"tests").glob("test_r7_reporting*.py"))
    write_json(RAW/"reporting_code_sha256.json", {"scope": "post-training aggregation/figure/audit/report; frozen science and thresholds unchanged",
               "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in reporting_paths}})
    content = "\n\n".join(part.strip() for part in sections if part.strip())+"\n"
    for target in (ROOT/"R7_PHASE_RELEASE_EXPERIMENT_REPORT.md", DOCS/"report.md"):
        target.write_text(content, encoding="utf-8")
    print(f"Built complete R7 report: {len(main)} main, {len(short)} short, {len(long)} long; {decision['classification']}", flush=True)


if __name__ == "__main__":
    build_report()
