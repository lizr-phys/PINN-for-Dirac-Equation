from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORT_PATH = PROJECT_ROOT / "EXPERIMENT_REPORT.md"

LABELS = {
    "vanilla": "Vanilla",
    "sp_cpa": "SP-CPA",
    "sp_cpa_conservative": "SP-CPA-Cons.",
    "two_branch": "Two-Branch oracle",
}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def f(value: float, digits: int = 4) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e3:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def mean_sd(values: pd.Series) -> str:
    array = values.to_numpy(float)
    if len(array) == 1:
        return f(array[0]) + " (n=1)"
    return f"{f(np.mean(array))} +/- {f(np.std(array, ddof=0))}"


def inject(text: str, name: str, content: str) -> str:
    block = f"<!-- {name} START -->\n{content.strip()}\n<!-- {name} END -->"
    pattern = re.compile(
        rf"<!-- {re.escape(name)} START -->.*?<!-- {re.escape(name)} END -->",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(block, text)
    marker = f"<!-- {name} -->"
    if marker not in text:
        raise RuntimeError(f"Missing report marker: {marker}")
    return text.replace(marker, block)


def per_seed_section(main: pd.DataFrame) -> str:
    ordered = main.sort_values(["case", "model", "seed"])
    accuracy_headers = [
        "Case",
        "Model",
        "Seed",
        "spinor L2",
        "final L2",
        "rho L2",
        "j L2",
        "F mean/min/final",
        "phase MAE/RMSE (rad)",
        "branch coeff. L2",
        "branch occ. RMSE",
    ]
    accuracy_rows: list[list[str]] = []
    conservation_headers = [
        "Case",
        "Model",
        "Seed",
        "C normalized RMS",
        "max norm drift",
        "max abs energy drift",
        "final PDE",
        "final struct",
        "final total",
        "params",
        "wall time (s)",
    ]
    conservation_rows: list[list[str]] = []
    for _, row in ordered.iterrows():
        prefix = [str(row.case), LABELS[row.model], str(int(row.seed))]
        accuracy_rows.append(
            prefix
            + [
                f(row.spinor_rel_l2),
                f(row.final_spinor_rel_l2),
                f(row.rho_rel_l2),
                f(row.current_rel_l2),
                "/".join(
                    (f(row.fidelity_mean, 6), f(row.fidelity_min, 6), f(row.fidelity_final, 6))
                ),
                f"{f(row.relative_phase_mae_rad)}/{f(row.relative_phase_rmse_rad)}",
                f(row.branch_coefficient_rel_l2),
                f(row.branch_occupation_rmse),
            ]
        )
        conservation_rows.append(
            prefix
            + [
                f(row.continuity_normalized_rms),
                f(row.max_norm_drift),
                f(row.max_abs_energy_drift),
                f(row.final_loss_pde),
                f(row.final_loss_structure),
                f(row.final_loss_total),
                str(int(row.parameter_count)),
                f(row.training_seconds, 1),
            ]
        )
    return (
        "以下第一张表给出所有要求的波函数/物理量/分支指标；第二张表给出守恒、最终 loss、参数量和逐 run 墙钟时间。"
        "完整复数分支系数时间序列在每个 run 的 `time_series.csv`，全部未汇总字段在 `metrics.json`。墙钟时间受并行进程竞争影响，不作为吞吐量公平基准。\n\n"
        + md_table(accuracy_headers, accuracy_rows)
        + "\n\n"
        + md_table(conservation_headers, conservation_rows)
    )


def aggregate_section(main: pd.DataFrame) -> str:
    metrics_a = [
        ("spinor_rel_l2", "spinor L2"),
        ("rho_rel_l2", "rho L2"),
        ("current_rel_l2", "j L2"),
        ("fidelity_mean", "F mean"),
        ("relative_phase_rmse_rad", "phase RMSE"),
        ("branch_coefficient_rel_l2", "branch coeff. L2"),
    ]
    metrics_b = [
        ("continuity_normalized_rms", "C norm. RMS"),
        ("max_norm_drift", "max norm drift"),
        ("max_abs_energy_drift", "max abs E drift"),
        ("final_loss_pde", "final PDE"),
        ("final_loss_structure", "final struct"),
    ]
    rows_a: list[list[str]] = []
    rows_b: list[list[str]] = []
    for (case, model), group in main.groupby(["case", "model"], sort=True):
        prefix = [case, LABELS[model], str(len(group))]
        rows_a.append(prefix + [mean_sd(group[column]) for column, _ in metrics_a])
        rows_b.append(prefix + [mean_sd(group[column]) for column, _ in metrics_b])
    note = (
        "误差写作 `mean +/- population SD`。前三个 learned models 均为 n=3；Two-Branch 为预注册的单个 oracle seed，"
        "不把 n=1 伪装成不确定性估计。"
    )
    return (
        note
        + "\n\n"
        + md_table(
            ["Case", "Model", "n"] + [label for _, label in metrics_a], rows_a
        )
        + "\n\n"
        + md_table(
            ["Case", "Model", "n"] + [label for _, label in metrics_b], rows_b
        )
    )


def sensitivity_section(summary: pd.DataFrame) -> str:
    data = summary[summary.experiment.eq("sensitivity")].sort_values("lambda_struct")
    headers = [
        "lambda",
        "spinor L2",
        "rho L2",
        "j L2",
        "F mean",
        "phase RMSE",
        "C norm. RMS",
        "max norm drift",
        "max abs E drift",
        "final PDE/struct",
    ]
    rows = []
    for _, row in data.iterrows():
        rows.append(
            [
                f(row.lambda_struct, 1),
                f(row.spinor_rel_l2),
                f(row.rho_rel_l2),
                f(row.current_rel_l2),
                f(row.fidelity_mean, 6),
                f(row.relative_phase_rmse_rad),
                f(row.continuity_normalized_rms),
                f(row.max_norm_drift),
                f(row.max_abs_energy_drift),
                f"{f(row.final_loss_pde)}/{f(row.final_loss_structure)}",
            ]
        )
    return (
        "固定 seed 0；`lambda=1` 是完全相同配置的 main run 引用，不重复训练。\n\n"
        + md_table(headers, rows)
    )


def gradient_section() -> str:
    data = pd.read_csv(RESULTS_ROOT / "gradient_conflict_summary.csv")
    rows = [
        [
            row.case,
            str(int(row.diagnostic_points)),
            f(row.gradient_cosine_mean),
            f(row.gradient_conflict_fraction, 4),
        ]
        for _, row in data.iterrows()
    ]
    return md_table(
        ["Case", "diagnostic points", "mean cosine", "fraction cosine < 0"], rows
    )


def interpretation_section(main: pd.DataFrame, summary: pd.DataFrame) -> str:
    means = main.groupby(["case", "model"]).mean(numeric_only=True)
    p_v = means.loc[("P", "vanilla")]
    p_s = means.loc[("P", "sp_cpa")]
    p_c = means.loc[("P", "sp_cpa_conservative")]
    m_v = means.loc[("M", "vanilla")]
    m_s = means.loc[("M", "sp_cpa")]
    m_c = means.loc[("M", "sp_cpa_conservative")]
    m_t = means.loc[("M", "two_branch")]
    gain_p = (p_v.spinor_rel_l2 - p_s.spinor_rel_l2) / p_v.spinor_rel_l2
    gain_m = (m_v.spinor_rel_l2 - m_s.spinor_rel_l2) / m_v.spinor_rel_l2
    sens10 = summary[
        summary.experiment.eq("sensitivity") & np.isclose(summary.lambda_struct, 10.0)
    ].iloc[0]
    sp0 = main[
        main.case.eq("M") & main.model.eq("sp_cpa") & main.seed.eq(0)
    ].iloc[0]
    return f"""1. **Case P：SP-CPA 明显优于 Vanilla。** 三 seed 平均旋量误差由 `{f(p_v.spinor_rel_l2)}` 降至 `{f(p_s.spinor_rel_l2)}`，相对下降 `{100*gain_p:.1f}%`；平均 fidelity 从 `{f(p_v.fidelity_mean)}` 升至 `{f(p_s.fidelity_mean, 6)}`。SP-CPA-Conservative 仍保持高精度（spinor `{f(p_c.spinor_rel_l2)}`），并进一步降低 rho、j、连续性、范数和能量误差，但 spinor 略差于纯 SP-CPA。

2. **公共单相位优势在 Case M 大幅下降。** Vanilla 与 SP-CPA 的平均旋量误差分别为 `{f(m_v.spinor_rel_l2)}` 和 `{f(m_s.spinor_rel_l2)}`，相对收益只有 `{100*gain_m:.1f}%`，远低于 Case P 的 `{100*gain_p:.1f}%`；SP-CPA 的流误差仍为 `{f(m_s.current_rel_l2)}`、相对相位 RMSE `{f(m_s.relative_phase_rmse_rad)} rad`。它能表示目标解，但没有把两个相反时间载频同时变慢。

3. **主实验 lambda=1 只部分改善守恒。** 相对 SP-CPA，SP-CPA-Conservative 的 Case M 平均连续性 RMS 从 `{f(m_s.continuity_normalized_rms)}` 降至 `{f(m_c.continuity_normalized_rms)}`，最大范数漂移从 `{f(m_s.max_norm_drift)}` 降至 `{f(m_c.max_norm_drift)}`；但最大绝对能量漂移由 `{f(m_s.max_abs_energy_drift)}` 恶化到 `{f(m_c.max_abs_energy_drift)}`。因此不能声称主 lambda=1 让三项守恒量全部变好。

4. **守恒改善没有同步恢复相位、概率流或 fidelity。** lambda=1 的平均 spinor/j 误差反而为 `{f(m_c.spinor_rel_l2)}/{f(m_c.current_rel_l2)}`，平均 fidelity `{f(m_c.fidelity_mean)}`，相对相位 RMSE `{f(m_c.relative_phase_rmse_rad)} rad`，都差于纯 SP-CPA。更强的 lambda=10 把 seed-0 的连续性/范数/能量指标从 `{f(sp0.continuity_normalized_rms)}/{f(sp0.max_norm_drift)}/{f(sp0.max_abs_energy_drift)}` 改善到 `{f(sens10.continuity_normalized_rms)}/{f(sens10.max_norm_drift)}/{f(sens10.max_abs_energy_drift)}`，但 spinor/j 变为 `{f(sens10.spinor_rel_l2)}/{f(sens10.current_rel_l2)}`、fidelity 仅 `{f(sens10.fidelity_mean, 6)}`。这与解析 wrong-phase 反例共同支持“守恒约束对分支相对相位不敏感”。

5. **Two-Branch 恢复混合支精度。** Case M oracle 的 spinor/j 误差为 `{f(m_t.spinor_rel_l2)}/{f(m_t.current_rel_l2)}`、fidelity `{f(m_t.fidelity_mean, 6)}`、相对相位 RMSE `{f(m_t.relative_phase_rmse_rad)} rad`；spinor 误差约比 SP-CPA 小 `{m_s.spinor_rel_l2/m_t.spinor_rel_l2:.0f}` 倍。这把主要困难定位到载波/分支预条件，而不是解析解、AD residual 或测试指标实现。

6. **共同失败模式必须单独说明。** Vanilla 在 P/M 两个 case 都出现远离 t=0 的振幅衰减，最大范数漂移约 `{f(p_v.max_norm_drift)}/{f(m_v.max_norm_drift)}`；Case M 的总能量本来为零，所以振幅坍缩仍可给出很小能量漂移。低 PDE loss 或小 Case-M 能量漂移本身都不是正确解的证据。"""


def hypothesis_section(main: pd.DataFrame) -> str:
    means = main.groupby(["case", "model"]).mean(numeric_only=True)
    p_v = means.loc[("P", "vanilla"), "spinor_rel_l2"]
    p_s = means.loc[("P", "sp_cpa"), "spinor_rel_l2"]
    m_v = means.loc[("M", "vanilla"), "spinor_rel_l2"]
    m_s = means.loc[("M", "sp_cpa"), "spinor_rel_l2"]
    gain_p = (p_v - p_s) / p_v
    gain_m = (m_v - m_s) / m_v
    return f"""**结论：在“固定预算下的载波解调/优化优势”意义上，核心假设得到支持；在“公共单相位加复振幅无法表示混合态”的字面意义上，假设不成立。**

- 支持证据：SP-CPA 相对 Vanilla 的旋量误差收益从 Case P 的 `{100*gain_p:.1f}%` 降到 Case M 的 `{100*gain_m:.1f}%`；Two-Branch 在 Case M 恢复约两个数量级的 spinor/current 精度；Case M 的 PDE/结构梯度平均余弦为 `-0.529`，冲突比例 `88.9%`。
- 守恒结论需限定：主 lambda=1 改善连续性与范数但恶化能量；敏感性 lambda=10 才同时显著改善三类守恒指标，而 spinor、相对相位、概率流和 fidelity 并未改善。这比“守恒损失总能变得很小”更细致，也更符合实际结果。
- 机制边界：`A` 是任意复旋量，所以 `A exp(iS)` 仍具有完整表示能力；观察到的是有限网络、归一化时间输入和 5000-step Adam 下的优化/归纳偏置退化，不是数学不可能。"""


def next_steps_section() -> str:
    return """值得继续研究正负能量分支感知或特征通道化的 Dirac-PINN，但下一步应把“oracle 载波”逐渐降为可学习结构，而不是直接宣称 Two-Branch 已是通用方法：

1. 用可学习的正/负能投影通道、局部色散或双时间 Fourier feature 代替硬编码的 `E,u_+,u_-`，并与参数量匹配的双通道 baseline 比较。
2. 先控制本实验暴露的长时域振幅坍缩：比较 causal/时间推进采样、分段时间窗、归一化硬约束和不含相位分解的 Fourier-time PINN；否则一般 PINN 失败会混入载波结论。
3. 对 lambda 敏感性和 Two-Branch 扩展到至少三个 seed，并报告 Pareto 前沿，而不是寻找单个“最好 lambda”。
4. 从单 k 平面波扩展到含多个 k 的波包、外势和散射；此时需要检验局部正负能分支定义是否稳定，以及通道耦合是否破坏优势。
5. 显式处理 SP-CPA 的规范冗余，可加入 gauge fixing 或幅度慢变正则，再验证“公共相位”是否真的承担载波解调，而不是由复振幅吸收全部相位。"""


def main() -> None:
    summary = pd.read_csv(RESULTS_ROOT / "summary.csv")
    main_runs = summary[summary.experiment.eq("main")].copy()
    if len(main_runs) != 20:
        raise RuntimeError(f"Expected 20 main runs, found {len(main_runs)}")
    text = REPORT_PATH.read_text(encoding="utf-8")
    text = inject(text, "PER_SEED_RESULTS", per_seed_section(main_runs))
    text = inject(text, "AGGREGATE_RESULTS", aggregate_section(main_runs))
    text = inject(text, "SENSITIVITY_RESULTS", sensitivity_section(summary))
    text = inject(text, "GRADIENT_RESULTS", gradient_section())
    text = inject(text, "INTERPRETATION", interpretation_section(main_runs, summary))
    text = inject(text, "HYPOTHESIS_DECISION", hypothesis_section(main_runs))
    text = inject(text, "NEXT_STEPS", next_steps_section())
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
