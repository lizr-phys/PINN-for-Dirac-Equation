from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .multimode_config import MODE_KEYS, MODEL_LABELS, MULTIMODE_MODELS, MULTIMODE_SEEDS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "MULTIMODE_LTPB_EXPERIMENT_REPORT.md"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "invalid"
    if number == 0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _mean_sd(summary: pd.DataFrame, model: str, metric: str) -> str:
    row = summary.set_index("model").loc[model]
    return f"{_fmt(row[f'{metric}_mean'])} ± {_fmt(row[f'{metric}_std'])}"


def _raw_links(model: str, seed: int) -> str:
    base = f"results/raw/multimode_ltpb/learned/{model}/seed_{seed}"
    names = ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")
    return " · ".join(f"[{name}]({base}/{name})" for name in names)


def build_report(results_root: Path = RESULTS, report_path: Path = REPORT) -> Path:
    runs = pd.read_csv(results_root / "multimode_ltpb_runs.csv")
    summary = pd.read_csv(results_root / "multimode_ltpb_summary.csv")
    comparisons = pd.read_csv(results_root / "multimode_ltpb_comparisons.csv")
    decision = json.loads((results_root / "multimode_ltpb_decision.json").read_text(encoding="utf-8"))
    audit_path = results_root / "multimode_ltpb_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None

    lines: list[str] = [
        "# 双波数正负能支混合 Dirac 方程的 Multi-LTPB 实验报告",
        "",
        "> 本报告严格对应正式目录中的 5 个 learned models × 5 个预注册 seeds。Four-carrier oracle 使用精确载频，只是诊断上限，不进入 learned 排名。",
        "",
        "## 1. 已有工作与本次问题",
        "",
        "此前单动量实验表明：公共单相位 SP-CPA 在纯正能支上有效，但在等权正负能支混合中载波预条件优势退化；总连续性、总范数和总能量不能充分识别相对相位、电流与旋量误差。LTPB 通过能支对齐的两个独立相位恢复了单动量混合态。机制消融进一步提示，本征能支基底和每支独立相位的一一对应都很重要，但单模态结果本身不能说明多个空间载频能否同时被处理。",
        "",
        "本实验因此询问：在初值已知包含 `k=2` 与 `k=5` 两个离散 Fourier 模态时，Multi-LTPB 能否只从自由 Dirac PDE 强残差中学习四个时间载频，并同时恢复旋量和概率流？",
        "",
        "## 2. 物理问题与精确参考",
        "",
        "在 `L=2π, m=5, T=5` 上求解",
        "",
        r"$$i\partial_t\psi=(-i\sigma_x\partial_x+m\sigma_z)\psi.$$",
        "",
        "四通道顺序固定为 `(k=2,+), (k=2,-), (k=5,+), (k=5,-)`，精确参考为",
        "",
        r"$$\psi(x,t)=\frac{1}{2\sqrt L}\sum_{l=1}^2\left[u_+(k_l)e^{ik_lx-iE_lt}+u_-(k_l)e^{ik_lx+iE_lt}\right],\quad E_l=\sqrt{k_l^2+m^2}.$$",
        "",
        "四个空间模态在每个固定时刻的空间 Hilbert 内积下正交；这里没有假定四个非共振时间载波在 `[0,5]` 上正交。归一化空间模态的精确投影系数模长为 `1/2`、占据为 `1/4`；模型内部使用未除以 `sqrt(L)` 的基底，因此内部初始系数为 `1/(2sqrt(L))`。",
        "",
        "## 3. 模型与 oracle 的区别",
        "",
        "Multi-LTPB 使用",
        "",
        r"$$\widehat\psi=\sum_l e^{ik_lx}\left[a_{l,+}u_+(k_l)e^{iS_{l,+}}+a_{l,-}u_-(k_l)e^{iS_{l,-}}\right].$$",
        "",
        "其中 `a=a0+τr`, `S=τq`, `τ=t/T`，所以初值严格满足。`k_l` 与由已知符号 `H_k` 计算的 `u_±(k_l)` 来自初值结构，可以进入 learned forward；精确 `E_l`、目标斜率和解析时间载波只在训练后评价中出现。MultiBranchComplex 使用相同四通道基底但学习普通复系数；ComponentPolar 不使用能支基底，只在原始两旋量分量上分相位；SP-CPA 只有一个公共相位；Vanilla 使用笛卡尔复旋量修正。",
        "",
        "Four-carrier oracle 直接写入 `exp(∓iE_lt)`，因此只回答“精确载波预条件的上限是什么”，不能作为 learned method 的证据。",
        "",
        "| 模型 | hidden width | 实输出 | 参数量 | 已知能支基底 | 显式独立相位 |",
        "|---|---:|---:|---:|---|---|",
    ]
    parameter_map = runs.groupby("model").parameter_count.first().to_dict()
    rows_model = [
        ("vanilla", 64, 4, "否", "否"),
        ("sp_cpa", 64, 5, "否", "公共单相位"),
        ("component_polar", 64, 4, "否", "两旋量分量"),
        ("multi_branch_complex", 63, 8, "是", "否"),
        ("multi_ltpb", 63, 8, "是", "四通道"),
    ]
    for model, width, outputs, basis, phase in rows_model:
        lines.append(f"| {MODEL_LABELS[model]} | {width} | {outputs} | {int(parameter_map[model]):,} | {basis} | {phase} |")

    lines.extend(
        [
            "",
            "## 4. 预注册、公平性与评价规范",
            "",
            "[预注册](MULTIMODE_LTPB_PREREGISTRATION.md) 在正式结果产生前冻结了模型、五个 seeds、5000 steps、1024 个每步重采样配点、Adam `1e-3`、`float64/complex128`、`4×64` 或参数匹配的 `4×63` 网络、128×201 测试网格和自动判据。[预注册与代码 SHA-256](results/multimode_ltpb_preregistration_sha256.json) 提供代码冻结证据；[旧结果保护清单](results/multimode_ltpb_protected_baseline.json) 用于确认本实验没有覆盖既有结果。",
            "",
            "同 seed 的五个模型使用相同的显式初始化 seed 协议与完全相同的配点序列；初始化 generator 与配点 generator 分离。全部 learned runs 只优化 Dirac PDE strong residual，没有解析标签、精确能量、相位监督或守恒辅助损失。",
            "",
            "相位评价先从总场做 Fourier-能支投影。对旋量和四个投影系数仅移除一个由全时空 overlap 决定的常数整体相位；禁止逐时刻或逐模态对齐，因此不会把共同错误频率从主 spinor 指标或相位斜率中消掉。密度、电流、fidelity、占据和相对相位本身规范不变。",
            "",
            "## 5. 逐 seed 正式结果",
            "",
            "| Model | Seed | Spinor rel L2 | Current rel L2 | Fidelity | Density rel L2 | Modal coeff rel L2 | Final PDE loss | Train s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runs.sort_values(["model", "seed"]).itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.model]} | {int(row.seed)} | {_fmt(row.spinor_rel_l2_gauge_aligned)} | {_fmt(row.current_rel_l2)} | {_fmt(row.fidelity_mean, 6)} | {_fmt(row.density_rel_l2)} | {_fmt(row.mode_coefficient_rel_l2_gauge_aligned)} | {_fmt(row.final_loss_pde)} | {_fmt(row.training_seconds, 2)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 五 seed 均值与 population SD",
            "",
            "| Model | Spinor rel L2 | Current rel L2 | Fidelity | Density rel L2 | Modal coeff rel L2 | PDE residual RMS | Train s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MULTIMODE_MODELS:
        lines.append(
            f"| {MODEL_LABELS[model]} | {_mean_sd(summary, model, 'spinor_rel_l2_gauge_aligned')} | {_mean_sd(summary, model, 'current_rel_l2')} | {_mean_sd(summary, model, 'fidelity_mean')} | {_mean_sd(summary, model, 'density_rel_l2')} | {_mean_sd(summary, model, 'mode_coefficient_rel_l2_gauge_aligned')} | {_mean_sd(summary, model, 'dense_pde_residual_rms')} | {_mean_sd(summary, model, 'training_seconds')} |"
        )

    lines.extend(
        [
            "",
            "![主要误差比较](results/figures/multimode_ltpb/primary_error_comparison.png)",
            "",
            "![逐 seed 结果](results/figures/multimode_ltpb/per_seed_results.png)",
            "",
            "## 7. 四个 learned phases 与模态诊断",
            "",
            "成功判据使用总场投影系数的 phase slope，而不是网络内部 `S_{l,s}(x,t)`。后者在允许空间依赖时存在通道重参数化冗余，只作为振幅漂移和相位曲率的辅助诊断。",
            "",
            "| Seed | Channel | Learned slope | Target | Relative error | R² | Valid fraction |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    ltpb = runs[runs.model == "multi_ltpb"].sort_values("seed")
    for row in ltpb.itertuples():
        for mode in MODE_KEYS:
            lines.append(
                f"| {int(row.seed)} | `{mode}` | {_fmt(getattr(row, f'{mode}_phase_slope'), 6)} | {_fmt(getattr(row, f'{mode}_phase_slope_target'), 6)} | {_fmt(getattr(row, f'{mode}_phase_slope_relative_error'))} | {_fmt(getattr(row, f'{mode}_phase_slope_r2'))} | {_fmt(getattr(row, f'{mode}_phase_valid_fraction'), 3)} |"
            )
    lines.extend(["", "| Channel | Mean absolute relative slope error | Pre-registered `<1%` |", "|---|---:|---|"])
    for item in decision["phase_slope_channels"]:
        lines.append(
            f"| `{item['mode']}` | {_fmt(item['mean_absolute_relative_error'])} | {'PASS' if item['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "![四个投影相位](results/figures/multimode_ltpb/learned_four_phases.png)",
            "",
            "![模态系数与占据误差](results/figures/multimode_ltpb/modal_coefficient_errors.png)",
            "",
            "## 8. 密度、概率流与相对相位",
            "",
            "`x=0` 的密度和概率流曲线直接检验多个空间/时间载频干涉；只恢复总范数或密度不能替代 current、spinor 和支间/波数间相对相位证据。每个 run 的 `time_series.csv` 保存两个支内与两个跨波数相对相位曲线及 circular errors。",
            "",
            "![x=0 密度与概率流](results/figures/multimode_ltpb/representative_density_current.png)",
            "",
            "## 9. 预注册自动判定",
            "",
            f"总判定：**{'PASS' if decision['overall_pass'] else 'FAIL'}**。完整机器可读证据见 [decision JSON](results/multimode_ltpb_decision.json)。",
            "",
            "| 条件 | 实际值 | 阈值 | 结果 |",
            "|---|---:|---:|---|",
        ]
    )
    absolute_labels = {
        "spinor_mean_le_0p05": "Multi-LTPB mean spinor",
        "current_mean_le_0p10": "Multi-LTPB mean current",
        "fidelity_mean_ge_0p99": "Multi-LTPB mean fidelity",
    }
    for key, item in decision["absolute_thresholds"].items():
        lines.append(f"| {absolute_labels[key]} | {_fmt(item['value'], 6)} | `{item['operator']} {item['threshold']}` | {'PASS' if item['pass'] else 'FAIL'} |")
    for metric in ("spinor", "current"):
        item = decision["mean_baseline_advantage"][metric]
        lines.append(f"| {metric} vs best baseline ({MODEL_LABELS[item['best_baseline']]}) | {_fmt(item['improvement_factor'], 3)}× | `>= 10×` | {'PASS' if item['pass'] else 'FAIL'} |")
    lines.append(f"| Paired-seed joint 10× advantage | {decision['paired_joint_pass_count']}/5 | `>=4/5` | {'PASS' if decision['paired_seed_rule_pass'] else 'FAIL'} |")
    lines.append(f"| Phase-slope channels below 1% | {decision['phase_slope_pass_count']}/4 | `>=3/4` | {'PASS' if decision['phase_slope_rule_pass'] else 'FAIL'} |")

    target_summary = summary.set_index("model").loc["multi_ltpb"]
    lines.extend(["", "## 10. 机制解释与失败归因", ""])
    if decision["overall_pass"]:
        lines.append(
            "本实验通过了预注册的联合标准，因此在这个受控自由、双离散动量问题中，证据支持：把独立可学习相位与已知 `(k_l,±)` 能支通道一一对齐，能把单动量的载波预条件机制推广到四载频情形。这个结论同时依赖已知 Fourier 波数和 Dirac 能支基底；它不是“网络从零发现了谱分解”。"
        )
    else:
        failed = [key for key, item in decision["absolute_thresholds"].items() if not item["pass"]]
        if not decision["mean_baseline_advantage"]["spinor"]["pass"]:
            failed.append("spinor tenfold baseline advantage")
        if not decision["mean_baseline_advantage"]["current"]["pass"]:
            failed.append("current tenfold baseline advantage")
        if not decision["paired_seed_rule_pass"]:
            failed.append("four-of-five paired-seed rule")
        if not decision["phase_slope_rule_pass"]:
            failed.append("three-of-four phase-slope rule")
        lines.append(
            "预注册联合标准未全部通过，失败项为：" + ", ".join(failed) + "。因此不能声称双动量推广已被本实验确证。"
        )
    pde_mean = float(target_summary["final_loss_pde_mean"])
    slope_valid = all(min(item["per_seed_valid_fractions"]) >= 0.9 for item in decision["phase_slope_channels"])
    lines.extend(
        [
            "",
            f"- **表达能力：** 单元测试通过手工设置四个 `q=Tω` 且 `r=0` 证明 Multi-LTPB 包含精确解；故数值失败不能直接归因为表示集合不包含目标，但这不等于有限网络容易找到该解。",
            f"- **优化收敛：** Multi-LTPB final collocation PDE loss 的五 seed mean 为 `{_fmt(pde_mean)}`；结合 dense PDE residual、训练曲线和 seed 波动判断长期 Adam 优化是否仍是限制。",
            f"- **模态可辨识性：** 四个精确初始投影均非零；评价使用物理投影而非内部相位。五 seed 的 90% 有效比例门槛{'全部满足' if slope_valid else '未全部满足'}。内部 `x` 依赖仍可能造成重参数化冗余。",
            "- **参数预算：** 所有 learned models 的参数量差不超过约 1.56%，因此本实验没有以额外容量换取 Multi-LTPB 优势；但这不能证明 4×63 对双动量问题已充分。",
            "- **物理先验：** Multi-LTPB 与 MultiBranchComplex 都获得已知 `k_l` 和 `u_±(k_l)`；若前者更好，差异才可归于显式能支-相位配对与相位载波预条件，而不是仅仅知道能支基底。",
            "",
            "![训练曲线](results/figures/multimode_ltpb/training_curves.png)",
            "",
            "![精度-时间比较](results/figures/multimode_ltpb/accuracy_time_tradeoff.png)",
            "",
            "## 11. 本实验不能推出的结论",
            "",
            "1. 这里的两个整数波数来自已知初值并被显式写入模型；没有学习未知谱，也没有处理连续 Fourier 波包。",
            "2. 自由 Hamiltonian 的每个 `k` 通道彼此解耦；一般外势会在动量间耦合，使固定通道与独立相位不再闭合。",
            "3. 单个质量、两个波数、一个时间区间和 Adam 预算不能证明更长时间、更多模态或其他参数下的稳健性。",
            "4. 即使 PASS，也只能说明已知离散双动量上的机制推广，不是通用 Dirac 求解器，更不是新颖性或世界首次声明。",
            "",
            "## 12. 下一步：连续波包与光滑外势",
            "",
            "扩展到多 `k` 波包需要解决通道数量随谱分辨率增长、连续谱相位的正则化、相邻 `k` 的共享表示、低占据模态相位不可辨识，以及内部包络导致的跨通道重参数化。加入光滑外势后还需处理动量耦合、局域正负能投影的非唯一性、避免把目标解谱信息泄漏进基底、过零振幅处的相位奇点，并重新设计能够区分局域能支交换的概率流与相对相位诊断。",
            "",
            "## 13. 数据、图表与审计索引",
            "",
            "- [逐 run CSV](results/multimode_ltpb_runs.csv)、[五 seed 汇总](results/multimode_ltpb_summary.csv)、[baseline 比较](results/multimode_ltpb_comparisons.csv)",
            "- [自动判定](results/multimode_ltpb_decision.json)、[结果审计](results/multimode_ltpb_audit.json)",
            "- [Figure manifest](results/figures/multimode_ltpb/figure_manifest.json)、[FIGURE_QA](results/figures/multimode_ltpb/FIGURE_QA.md)",
            "- [Oracle config](results/raw/multimode_ltpb/oracle/config.json)、[oracle metrics](results/raw/multimode_ltpb/oracle/metrics.json)、[oracle time series](results/raw/multimode_ltpb/oracle/time_series.csv)、[oracle dense grid](results/raw/multimode_ltpb/oracle/dense_grid.npz)",
            "",
            "### 每个 learned run 的六件套",
            "",
            "| Model | Seed | Raw artifacts |",
            "|---|---:|---|",
        ]
    )
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            lines.append(f"| {MODEL_LABELS[model]} | {seed} | {_raw_links(model, seed)} |")
    if audit is not None:
        lines.extend(["", f"报告构建时可见的审计状态：**{audit.get('status', 'UNKNOWN')}**。最终报告写入后会再次运行审计与链接检查。"])
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Multi-LTPB experiment report")
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(build_report(args.results_root, args.report))


if __name__ == "__main__":
    main()
