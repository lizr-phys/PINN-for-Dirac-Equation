from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wavepacket_config import MODEL_LABELS, WAVEPACKET_MODELS, WAVEPACKET_SEEDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw" / "wavepacket_shared_spectral_ltpb"
FIGURES = RESULTS / "figures" / "wavepacket_shared_spectral_ltpb"
REPORT = ROOT / "WAVEPACKET_SHARED_SPECTRAL_LTPB_REPORT.md"


def _fmt(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "N/A"
    absolute = abs(value)
    if absolute != 0.0 and (absolute < 1.0e-4 or absolute >= 1.0e4):
        return f"{value:.4e}"
    return f"{value:.{digits}f}"


def _mean_sd(summary: pd.DataFrame, model: str, metric: str) -> str:
    row = summary.set_index("model").loc[model]
    return (
        f"{_fmt(float(row[f'{metric}_mean']))} ± "
        f"{_fmt(float(row[f'{metric}_std']))}"
    )


def _raw_link_table() -> list[str]:
    lines = [
        "### 每个 learned run 的六件套",
        "",
        "| Model | Seed | Raw artifacts |",
        "|---|---:|---|",
    ]
    for model in WAVEPACKET_MODELS:
        for seed in WAVEPACKET_SEEDS:
            base = f"results/raw/wavepacket_shared_spectral_ltpb/learned/{model}/seed_{seed}"
            links = " · ".join(
                f"[{name}]({base}/{name})"
                for name in (
                    "config.json",
                    "metrics.json",
                    "loss_history.csv",
                    "time_series.csv",
                    "dense_grid.npz",
                    "model_state.pt",
                )
            )
            lines.append(f"| {MODEL_LABELS[model]} | {seed} | {links} |")
    return lines


def build_report() -> Path:
    runs = pd.read_csv(RESULTS / "wavepacket_shared_spectral_ltpb_runs.csv")
    summary = pd.read_csv(RESULTS / "wavepacket_shared_spectral_ltpb_summary.csv")
    comparisons = pd.read_csv(
        RESULTS / "wavepacket_shared_spectral_ltpb_comparisons.csv"
    )
    decision = json.loads(
        (RESULTS / "wavepacket_shared_spectral_ltpb_decision.json").read_text(
            encoding="utf-8"
        )
    )
    target = runs[runs.model == "shared_spectral_ltpb"].sort_values("seed")
    decision_word = "PASS" if decision["overall_pass"] else "FAIL"
    interpolation_word = (
        "PASS"
        if decision["heldout_midpoint_phase_slope"]["pass"]
        else "FAIL"
    )

    lines: list[str] = [
        "# 自由 Dirac 连续高斯波包的共享谱 LTPB 实验报告",
        "",
        "> 本报告对应 5 个 learned models × 5 个预注册 seeds。Spectral oracle 直接使用精确色散，只作参考上限，不进入 learned 排名。",
        "",
        "## 1. 已有证据与本次唯一问题",
        "",
        "此前实验从公共单相位在正负能支混合态中的退化出发，证明单动量 LTPB 可以在不写入精确时间频率时自行学习两支相反载频；双离散波数 Multi-LTPB 又在四个预先列出的 `(k,±)` 通道上通过了预注册判据。然而，那些网络仍显式枚举有限通道，参数化并未回答一个网络能否在波数方向共享规律。",
        "",
        "本实验只回答：一个共享 `(k,t)` 网络能否用自由 Dirac 谱残差统一表示 33 个高斯占据波数，并在未参与训练的相邻中点上保持正确相位斜率，从而恢复连续波包的场、密度、流和正负支分裂。",
        "",
        "## 2. 物理问题、谱规范与 Parseval",
        "",
        "在 `x∈[-8π,8π), L=16π, t∈[0,5], m=5` 上求解自由一维 Dirac 方程。训练波数为 `k_n=n/8, n=0,...,32`，初始高斯中心 `k0=2`、宽度 `σk=0.5`，正负支等权。采用正交 Fourier 规范",
        "",
        r"$$\psi(x,t)=L^{-1/2}\sum_n e^{ik_nx}\widetilde\psi_n(t),\qquad c_{n,\pm}(0)=g_n/\sqrt2,$$",
        "",
        "因此总范数为 1、每支占据为 1/2。谱训练直接计算非对角强残差",
        "",
        r"$$R_n=i\partial_t\widetilde\psi_n-(k_n\sigma_x+m\sigma_z)\widetilde\psi_n,$$",
        "",
        "而不是含精确 `E(k)` 的支方程。Parseval 给出 `∫|R(x)|²dx=Σ_n|R_n|²`；实现中谱 loss 的缩放与物理空间平均残差完全一致。该等价限于 33 模态 Fourier 子空间；物理基线可能生成的带外残差不被伪装为带内谱残差。",
        "",
        "## 3. 方法、控制模型与精确 oracle",
        "",
        "Shared-Spectral-LTPB 使用一个共享 `4×64 tanh` MLP 接收归一化 `(k,t/T)` 并输出 `(r+,r-,q+,q-)`：",
        "",
        r"$$a_\pm(k,t)=g(k)/\sqrt2+(t/T)r_\pm(k,t),\qquad S_\pm(k,t)=(t/T)q_\pm(k,t),$$",
        "",
        r"$$\widehat{\widetilde\psi}=a_+u_+(k)e^{iS_+}+a_-u_-(k)e^{iS_-}.$$",
        "`u±(k)` 由已知符号 `H(k)` 的连续实规范构造。learned forward、loss、输入、optimizer、config 与 checkpoint 均没有精确能量或解析时间载波；精确 `E(k)` 首次出现在 reference/evaluator 中。",
        "",
        "| Model | 输入/参数化 | 参数量 |",
        "|---|---|---:|",
    ]
    for model in WAVEPACKET_MODELS:
        row = summary.set_index("model").loc[model]
        description = {
            "physical_vanilla": "物理空间笛卡尔旋量修正",
            "physical_sp_cpa": "物理空间公共单相位",
            "physical_component_polar": "物理旋量分量双极坐标相位",
            "shared_spectral_complex": "共享能支基底、普通复分支系数",
            "shared_spectral_ltpb": "共享能支基底、逐支实振幅与相位",
        }[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {description} | "
            f"{int(round(float(row['parameter_count_mean']))):,} |"
        )
    lines.extend(
        [
            "",
            "两个共享谱模型具有完全相同的 MLP、参数量、初始权重和逐步时间样本。17、33、65 个有效模态的构造测试均保持同一可训练参数量；这只证明参数量不随模态数线性增长，不证明三种分辨率都能训练成功。Spectral oracle 直接写入精确 `E(k)`，参数为 0，明确排除于排名。",
            "",
            "## 4. 预注册与公平性",
            "",
            "[预注册](WAVEPACKET_SHARED_SPECTRAL_LTPB_PREREGISTRATION.md) 在任何正式结果前固定了物理参数、五种模型、seeds `0–4`、5000 steps、Adam `1e-3`、float64/complex128、网络规模、测试网格和全部阈值。[代码冻结清单](results/wavepacket_shared_spectral_ltpb_preregistration_sha256.json) 记录正式代码 SHA-256；[旧结果保护清单](results/wavepacket_shared_spectral_ltpb_protected_baseline.json) 保护 670 个已有文件。",
            "",
            "每一步对所有模型共享 32 个时间样本；谱模型在全部 33 个 `k` 上得到 1056 个点，物理模型将同一时间批与 32 个空间点组合得到 1024 点。初始化、时间采样和物理空间采样采用相互独立的显式随机流。所有 learned runs 只使用 PDE loss，没有标签、能量、相位或守恒监督，且没有选择性重跑。",
            "",
            "## 5. 逐 seed 正式结果",
            "",
            "| Model | Seed | Spinor | Density | Current | Fidelity | Train phase | Held-out phase | PDE RMS | Train s | Peak ΔRSS MB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runs.sort_values(["model", "seed"], kind="stable").itertuples():
        lines.append(
            f"| {MODEL_LABELS[str(row.model)]} | {int(row.seed)} | "
            f"{_fmt(float(row.spinor_rel_l2_gauge_aligned),4)} | "
            f"{_fmt(float(row.density_rel_l2),4)} | "
            f"{_fmt(float(row.current_rel_l2),4)} | "
            f"{_fmt(float(row.fidelity_mean),6)} | "
            f"{_fmt(float(row.training_phase_slope_weighted_relative_error),4)} | "
            f"{_fmt(float(row.heldout_phase_slope_weighted_relative_error),4)} | "
            f"{_fmt(float(row.dense_physical_pde_residual_rms),4)} | "
            f"{_fmt(float(row.training_seconds),1)} | "
            f"{_fmt(float(row.peak_rss_increment_mb),1)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 五 seed 均值与 population SD",
            "",
            "| Model | Spinor | Density | Current | Fidelity | Coefficient | PDE RMS | Train s | Peak ΔRSS MB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in WAVEPACKET_MODELS:
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{_mean_sd(summary,model,'spinor_rel_l2_gauge_aligned')} | "
            f"{_mean_sd(summary,model,'density_rel_l2')} | "
            f"{_mean_sd(summary,model,'current_rel_l2')} | "
            f"{_mean_sd(summary,model,'fidelity_mean')} | "
            f"{_mean_sd(summary,model,'spectral_coefficient_rel_l2_gauge_aligned')} | "
            f"{_mean_sd(summary,model,'dense_physical_pde_residual_rms')} | "
            f"{_mean_sd(summary,model,'training_seconds')} | "
            f"{_mean_sd(summary,model,'peak_rss_increment_mb')} |"
        )
    lines.extend(
        [
            "",
            "![主要误差比较](results/figures/wavepacket_shared_spectral_ltpb/primary_error_comparison.png)",
            "",
            "![逐 seed 结果](results/figures/wavepacket_shared_spectral_ltpb/per_seed_results.png)",
            "",
            "## 7. 训练波数与 held-out 中点相位",
            "",
            "成功判据中的斜率来自预测总场的 Fourier–能支投影，只移除一个全时空常数规范相位；不使用网络内部 `S±` 代替物理相位。中点没有进入训练，且没有重新归一化高斯初值。",
            "",
            "| Seed | Training weighted slope error | Training min valid | Midpoint weighted slope error | Midpoint min valid | Midpoint PDE RMS |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target.itertuples():
        lines.append(
            f"| {int(row.seed)} | "
            f"{_fmt(float(row.training_phase_slope_weighted_relative_error))} | "
            f"{_fmt(float(row.training_phase_min_valid_fraction),3)} | "
            f"{_fmt(float(row.heldout_phase_slope_weighted_relative_error))} | "
            f"{_fmt(float(row.heldout_phase_min_valid_fraction),3)} | "
            f"{_fmt(float(row.heldout_spectral_pde_residual_rms))} |"
        )
    lines.extend(
        [
            "",
            f"预注册中点判据：**{interpolation_word}**。完整的 33×2 训练斜率和 32×2 中点斜率、目标、误差、有效比例与曲率均保存在每个 target run 的 `dense_grid.npz` 及图表 source data 中。",
            "",
            "![训练波数与中点相位斜率](results/figures/wavepacket_shared_spectral_ltpb/phase_slope_train_midpoint.png)",
            "",
            "## 8. 波包密度、概率流与分支动力学",
            "",
            "![波包密度时空图](results/figures/wavepacket_shared_spectral_ltpb/wavepacket_density_spacetime.png)",
            "",
            "![概率流时空图](results/figures/wavepacket_shared_spectral_ltpb/wavepacket_current_spacetime.png)",
            "",
            "| Seed | + center RMSE | − center RMSE | + velocity rel | − velocity rel | + width rel L2 | − width rel L2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in target.itertuples():
        lines.append(
            f"| {int(row.seed)} | {_fmt(float(row.branch_plus_center_rmse))} | "
            f"{_fmt(float(row.branch_minus_center_rmse))} | "
            f"{_fmt(float(row.branch_plus_velocity_relative_error))} | "
            f"{_fmt(float(row.branch_minus_velocity_relative_error))} | "
            f"{_fmt(float(row.branch_plus_width_rel_l2))} | "
            f"{_fmt(float(row.branch_minus_width_rel_l2))} |"
        )
    branch_pass = decision["wavepacket_branch_dynamics"]["pass"]
    lines.extend(
        [
            "",
            f"自动分支动力学判据：**{'PASS' if branch_pass else 'FAIL'}**。中心和宽度均由各支 Fourier–能支投影重建后的密度圆周矩得到，不由总密度目测判断。",
            "",
            "![正负支中心与宽度](results/figures/wavepacket_shared_spectral_ltpb/branch_center_width.png)",
            "",
            "![谱系数误差](results/figures/wavepacket_shared_spectral_ltpb/spectral_coefficient_errors.png)",
            "",
            "## 9. 预注册自动判定",
            "",
            f"总判定：**{decision_word}**。机器可读详情见 [decision JSON](results/wavepacket_shared_spectral_ltpb_decision.json)。",
            "",
            "| 条件 | 实际值 | 阈值 | 结果 |",
            "|---|---:|---:|---|",
        ]
    )
    abs_rules = decision["absolute_thresholds"]
    for label, key in (
        ("Mean spinor", "spinor_mean_le_0p05"),
        ("Mean current", "current_mean_le_0p10"),
        ("Mean fidelity", "fidelity_mean_ge_0p99"),
    ):
        rule = abs_rules[key]
        lines.append(
            f"| {label} | {_fmt(float(rule['value']))} | "
            f"`{rule['operator']} {rule['threshold']}` | "
            f"{'PASS' if rule['pass'] else 'FAIL'} |"
        )
    for metric in ("spinor", "current"):
        rule = decision["mean_baseline_advantage"][metric]
        lines.append(
            f"| {metric} vs best learned baseline ({rule['best_baseline']}) | "
            f"{_fmt(float(rule['improvement_factor']),2)}× | `>=10×` | "
            f"{'PASS' if rule['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            f"| Paired-seed joint advantage | {decision['paired_joint_pass_count']}/5 | `>=4/5` | {'PASS' if decision['paired_seed_rule_pass'] else 'FAIL'} |",
            f"| Training weighted phase slope | {_fmt(float(decision['training_phase_slope']['five_seed_mean_weighted_relative_error']))} | `<0.01` | {'PASS' if decision['training_phase_slope']['pass'] else 'FAIL'} |",
            f"| Held-out weighted phase slope | {_fmt(float(decision['heldout_midpoint_phase_slope']['five_seed_mean_weighted_relative_error']))} | `<0.02` | {'PASS' if decision['heldout_midpoint_phase_slope']['pass'] else 'FAIL'} |",
            f"| Branch center/width/splitting joint rule | — | frozen compound rule | {'PASS' if branch_pass else 'FAIL'} |",
            "",
            "## 10. 优化、时间与内存",
            "",
            "训练时间仅覆盖 5000 个 Adam steps；peak RSS 使用训练进程内统一轮询口径，报告训练期间绝对峰值及相对训练起点增量。参数量不随共享谱模态数增长并不意味着计算量或内存不增长：每步谱配置点数仍与模态数成正比。",
            "",
            "![训练曲线](results/figures/wavepacket_shared_spectral_ltpb/training_curves.png)",
            "",
            "![精度、时间与内存](results/figures/wavepacket_shared_spectral_ltpb/accuracy_time_memory.png)",
            "",
            "## 11. 结论解释与失败归因",
            "",
        ]
    )
    if decision["overall_pass"]:
        lines.append(
            "全部预注册条件同时通过，因此结果支持：在这个已知自由 Dirac Fourier 表示和 `[0,4]` 波数区间内，共享能支相位网络不仅拟合了 33 个训练模态，也在相邻中点上插值了正确的物理相位斜率，并恢复了正负支分裂传播。"
        )
    elif decision["heldout_midpoint_phase_slope"]["pass"]:
        lines.append(
            "总判定失败，但中点相位插值本身通过。报告据各子判据区分总场优化、概率流、波包宽度或基线优势，而不通过后验调参隐藏失败。"
        )
    else:
        lines.append(
            "中点判据失败，因此无论训练模态或总场误差多低，本实验最多只能称为“33 模态共享拟合”，不能声称连续谱插值成功。"
        )
    lines.extend(
        [
            "",
            "机制上必须区分：Shared-Spectral-LTPB 的表示集合、显式逐支相位带来的载波预条件、5000-step Adam 的优化结果、低占据模态可辨识性，以及已知 `H(k)` 与 `u±(k)` 带来的物理先验。已知符号本身已经包含产生色散的物理规律；本实验检验网络能否从离散残差约束统一插值该规律，不是从无物理先验的数据中发现色散。",
            "",
            "## 12. 本实验不能推出的结论",
            "",
            "1. 中点是 `H(k)` 参数化 ODE 的区间内插值探针，不是同一 `L=16π` 周期盒允许的 Fourier 模态。",
            "2. 初始 Fourier 支撑、质量和自由符号均已知；实验没有发现未知谱，也没有检验区间外波数。",
            "3. 自由问题中不同 `k` 完全解耦；外势会造成动量耦合，本轮没有加入也不能代表这种情况。",
            "4. 参数量与模态数无关不等于运行成本无关，更不等于无穷连续谱已被解决。",
            "5. 本结果不是通用 Dirac 求解器、传统谱方法的替代性证明、文献新颖性结论或“世界首次”声明。",
            "",
            "## 13. 数据、图表与审计索引",
            "",
            "- [逐 run CSV](results/wavepacket_shared_spectral_ltpb_runs.csv)、[五 seed 汇总](results/wavepacket_shared_spectral_ltpb_summary.csv)、[baseline 比较](results/wavepacket_shared_spectral_ltpb_comparisons.csv)",
            "- [自动判定](results/wavepacket_shared_spectral_ltpb_decision.json)、[最终审计](results/wavepacket_shared_spectral_ltpb_audit.json)",
            "- [预注册代码散列](results/wavepacket_shared_spectral_ltpb_preregistration_sha256.json)、[旧结果保护清单](results/wavepacket_shared_spectral_ltpb_protected_baseline.json)",
            "- [Figure manifest](results/figures/wavepacket_shared_spectral_ltpb/figure_manifest.json)、[FIGURE_QA](results/figures/wavepacket_shared_spectral_ltpb/FIGURE_QA.md)",
            "- [Oracle config](results/raw/wavepacket_shared_spectral_ltpb/oracle/config.json)、[metrics](results/raw/wavepacket_shared_spectral_ltpb/oracle/metrics.json)、[time series](results/raw/wavepacket_shared_spectral_ltpb/oracle/time_series.csv)、[dense grid](results/raw/wavepacket_shared_spectral_ltpb/oracle/dense_grid.npz)",
            "",
            *_raw_link_table(),
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return REPORT


def main() -> None:
    print(build_report())


if __name__ == "__main__":
    main()
