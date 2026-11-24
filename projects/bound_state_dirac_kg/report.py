"""Generate the Chinese report strictly from saved numerical results."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .physics import ROOT,config


def table(headers,rows):
    esc=lambda value:str(value).replace("|",r"\|")
    return "| " + " | ".join(map(esc,headers)) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| "+" | ".join(map(esc,row))+" |" for row in rows)


def main():
    R=ROOT/"results"; cfg=config()
    df=pd.read_csv(R/"formal_summary.csv")
    if len(df)!=6: raise RuntimeError("Report requires all six formal runs.")
    ref=json.loads((R/"reference"/"summary.json").read_text())
    conv=pd.read_csv(R/"reference"/"convergence.csv")
    lines=["# 一维 Dirac / Klein–Gordon 定态束缚态：基础验证报告\n",
        "计算日期：2026-09-26。全部数据来自本目录实际运行；每个模型三个随机种子，无失败种子删选。\n",
        "## 1. 模型和已确认结果\n",
        "采用 ℏ=c=1，m=1，V(x)=−0.5 exp(−x²/8)，仅含静态时间分量矢量耦合，Lorentz 标量势 S=0。"
        "传统参考方法先确认束缚态，再冻结参数。每个方程只训练最低正能、第一分量无节点的束缚态。\n",
        table(["模型","参考能量","PINN 三种子均值 ± 样本标准差","波函数相对 L²（均值 ± 标准差）"],
              [[k.upper(),f'{ref[k]["energy"]:.12f}',f'{g.energy.mean():.12f} ± {g.energy.std():.2e}',
                f'{g.wavefunction_rel_l2.mean():.3e} ± {g.wavefunction_rel_l2.std():.2e}'] for k,g in df.groupby("kind")]),
        f'\n两参考能量相差 **{ref["dirac"]["energy"]-ref["kg"]["energy"]:.9f}**，不能视为同谱。'
        "这是一项基础数值复现与验证，不构成原创物理结果，也不证明 PINN 优于传统方法。\n",
        "## 2. 物理约定与辨态\n",
        "定态分离 Ψ=e^(−iEt)ψ、Φ=e^(−iEt)φ；Dirac 取 ψ=(u,iv)ᵀ，则\n"
        "\n\\[u'=-(E-V+m)v,\\qquad v'=(E-V-m)u.\\]\n"
        "\nKG 为 \\(\\phi''+[(E-V)^2-m^2]\\phi=0\\)。若采用标量势，质量应替换为 m+S；不能将其当作上述 V。"
        "V→0 时 κ=√(m²−E²)，指数衰减束缚态位于 −m<E<m，本轮只找 0<E<m。右端边界为 "
        "v=κu/(E+m) 或 φ′=−κφ。完整符号推导、Dirac 消元后的 V′ 项与 KG 正交关系见 [THEORY.md](THEORY.md)。\n",
        "Dirac 使用 ∫(u²+v²)dx=1。KG 使用 ∫φ²dx=1 固定幅度，电荷约定为 ρ=2(E−V)|φ|²；"
        f'参考总电荷 **Q={ref["kg"]["charge"]:.10f}>0**。所有 PINN 目标态亦为正电荷，|φ|² 不被解释为一般概率密度。\n',
        "## 3. 参考解可信度\n",
        "主参考为 L=28 上的自适应 BVP 配点（tol=2×10⁻¹²），再用 DOP853 反向射击及 "
        "N=513 的 Fourier 谱方法独立核验。KG 谱计算使用关于 E 的二次铅笔伴随线性化。\n",
        table(["模型","射击 |ΔE|","Fourier |ΔE|","射击波形 L² 差","Fourier 波形 L² 差","BVP 连续残差 L²"],
              [[k.upper(),f'{ref[k]["shooting_energy_difference"]:.2e}',f'{ref[k]["fourier_energy_difference"]:.2e}',
                f'{ref[k]["shooting_l2"]:.2e}',f'{ref[k]["fourier_l2"]:.2e}',f'{ref[k]["bvp_residual_l2"]:.2e}'] for k in ["dirac","kg"]]),
        "\nL=12,16,20,24,28；BVP 初始网格 81/161/321 点、容差 10⁻⁷/10⁻⁹/2×10⁻¹²；"
        "Fourier 在 L=20 上用 N=33,65,129,257,513 加密。最终 BVP 自适应实际节点数分别为 "
        f'{ref["dirac"]["adaptive_nodes"]}（Dirac）、{ref["kg"]["adaptive_nodes"]}（KG）。\n',
        table(["模型","BVP E(L=20)−E(L=28) 绝对值","Fourier 波形差 L=20","Fourier 波形差 L=24","Fourier 波形差 L=28"],
              [[k.upper(),f'{conv.query("kind == @k and method == \'bvp\' and L == 20 and N == 321").abs_delta.iloc[0]:.2e}']+
               [f'{conv.query("kind == @k and method == \'fourier\' and L == @L and N == 257").wavefunction_l2.iloc[0]:.2e}' for L in [20,24,28]]
               for k in ["dirac","kg"]]),
        "\n能量一致性到约 10⁻¹⁴–10⁻¹⁵、波形一致性到约 10⁻¹²；这只是经验一致性，不是严格误差上界。"
        "保守采用参考能量不确定量级 2×10⁻¹²、公共评价区间 [−20,20] 上参考波形误差量级 10⁻⁹ 进行本轮误差分离，均明显小于六次 PINN 误差。\n",
        "中心差分的导数符号 sin(kh)/h 在零动量和网格极限处有两个零点，计算中确实出现成对候选能级；"
        "其高频分量及连续残差暴露了离散污染。正式采用奇数网格的非局域 Fourier 导数，并用 BVP/射击核验，"
        "没有把中心差分能隙内候选值当作物理态。近 E=1 的高激发候选点仍受区间效应影响，本报告不将其计为已验证成果。"
        "完整网格和倍增原始记录见 [参考数据](results/reference/convergence.csv) 与 [倍增诊断](results/reference/naive_dirac_doubling.csv)。\n",
        "## 4. PINN 与三种子验证\n",
        "MLP 为三层 32 宽 tanh、float64；能量作为 sigmoid 参数训练。Dirac 参数数 2243，KG 2210（均含能量参数）。"
        "偶/奇宇称、第一分量无节点和指数包络为硬结构；归一化与有限端点 Robin 条件为软损失。"
        "384 点残差求积、256 点归一化求积，Adam 2000 次后 L-BFGS 最多 1200 次。\n",
        "**先验披露：** 参考谱用于选定 [0.5,0.8] 的宽能区与目标宇称；训练没有参考波函数监督、参考能量回归项或能量最小化项。"
        "固定预算结束后才评估参考误差。初始能量覆盖约 0.588–0.716；种子 0/1/2 代表不同参数初始化，配点固定。"
        "损失公式、初始化、L-BFGS 数值尺度调整及首轮数据说明见 [METHODS.md](METHODS.md)。\n",
        table(["模型/种子","E_PINN","|ΔE|","相对能量误差","波函数相对 L²","耗时/s"],
              [[f'{r.kind}/{r.seed}',f'{r.energy:.12f}',f'{r.energy_abs_error:.3e}',f'{r.energy_rel_error:.3e}',
                f'{r.wavefunction_rel_l2:.3e}',f'{r.seconds:.1f}'] for r in df.itertuples()]),
        "\n只对齐一个整体符号，Dirac 两分量共用；不对预测结果事后重归一化。下表在独立 8001 点测试网格上计算；"
        "R_L² 是残差向量的积分范数，R_RMS=R_L²/√40。\n",
        table(["模型/种子","|N−1|","R_L²","R_RMS","max|R|","端点振幅","Q"],
              [[f'{r.kind}/{r.seed}',f'{r.normalization_error:.2e}',f'{r.residual_l2:.2e}',f'{r.residual_rms:.2e}',
                f'{r.residual_max:.2e}',f'{r.boundary_amplitude:.2e}',f'{r.charge:.9f}'] for r in df.itertuples()]),
        "\n六次第一分量节点数均为 0，宇称误差约机器精度。Dirac 的下分量相对误差较大：\n",
        table(["Dirac 种子","u 相对 L²","v 相对 L²"],
              [[r.seed,f'{r.component_0_rel_l2:.3e}',f'{r.component_1_rel_l2:.3e}'] for r in df.query("kind == 'dirac'").itertuples()]),
        "\n两个定态的连续性方程残差在选定实表示下代数上为零；这一点由参数化保证，不能当作独立精度证据。"
        "上表非零的微分方程残差才是求解误差诊断。训练时间为单线程 CPU 的实际墙钟，正式运行有两个进程并发；"
        "不能据此作严格硬件性能比较。传统单次高精度 BVP 仅需约 "
        f'{conv.query("method == \'bvp\' and L == 28 and N == 321").seconds.mean():.2f} 秒；PINN 的 Dirac/KG 平均耗时分别为 '
        f'{df.query("kind == \'dirac\'").seconds.mean():.1f}/{df.query("kind == \'kg\'").seconds.mean():.1f} 秒，'
        "本例传统方法明显更适合作为高精度求解器。\n",
        "## 5. 衰减与敏感性\n",
        table(["模型","κ_ref","参考 |y(20)|","参考域外归一化积分","PINN 最大端点/峰值","PINN 端点对数斜率范围"],
              [[k.upper(),f'{ref[k]["kappa"]:.9f}',f'{ref[k]["boundary_amplitude_at_20"]:.3e}',
                f'{ref[k]["tail_norm_estimate_beyond_20"]:.3e}',f'{g.boundary_to_peak.max():.3e}',
                f'{g.measured_log_slope_at_boundary.min():.6f}–{g.measured_log_slope_at_boundary.max():.6f}'] for k,g in df.groupby("kind")]),
        "\nRobin 条件是软约束，因此端点斜率仍有小偏差；不能因端点振幅小便声称尾形完全准确。"
        "域外归一化积分 ∫|y|²dx 按自由尾估计为约 10⁻¹³；其平方根（域外 L² 范数）为约 5×10⁻⁷。"
        "表中波函数误差在各自有限计算区间上评价，没有混入域外截断误差。\n"]
    gridrows=[]
    for k in ["dirac","kg"]:
        grids=[pd.read_csv(R/"formal"/k/f"seed_{s}"/"test_grid_convergence.csv") for s in [0,1,2]]
        max_l2=max(abs(g.wavefunction_rel_l2.iloc[-1]-g.wavefunction_rel_l2.iloc[-2]) for g in grids)
        max_r=max(abs(g.residual_l2.iloc[-1]-g.residual_l2.iloc[-2]) for g in grids)
        gridrows.append([k.upper(),f'{max_l2:.3e}',f'{max_r:.3e}'])
    lines += [table(["模型","测试网格 4001→8001：最大 L² 指标变化","最大残差范数变化"],gridrows),
              "\n另在种子 0 固定预算下分别将残差/归一化求积加密为 768/512 点、或将 L 扩展至 24：\n"]
    sensrows=[]
    for tag,label in [("formal","基准"),("grid_refined","配点加密"),("domain_extended","L=24")]:
        path=R/f"{tag}_summary.csv"
        if not path.exists(): raise RuntimeError(f"Sensitivity run incomplete: {path}")
        for r in pd.read_csv(path).query("seed == 0").itertuples():
            sensrows.append([r.kind,label,f'{r.energy:.12f}',f'{r.energy_abs_error:.2e}',f'{r.wavefunction_rel_l2:.2e}',f'{r.residual_l2:.2e}'])
    lines += [table(["模型","设置","E","|ΔE|","相对 L²","R_L²"],sensrows),
        "\n这些是单种子敏感性检查，不能当作训练误差随网格单调收敛的证明。PINN 优化路径会随配点和域长变化，"
        "即便积分已收敛，最终训练误差也可能波动；所有敏感性运行均保留，未替换正式三种子结果。\n",
        "## 6. 图表\n",
        "PINN 图包含全部三个正式种子；最后一组图展示独立参考验证。CSV/NPZ 来源和图注记录在 [图表清单](results/figures/manifest.json)，另附可编辑 SVG 与 PDF。\n"]
    for stem,title in [("01_potential_energy","势阱与目标能级"),("02_dirac_components","Dirac 逐分量与点误差"),
                       ("03_kg_wavefunction_charge","KG 波函数、电荷及点误差"),("04_residuals_tails","独立网格残差与完整尾部"),
                       ("05_training","训练曲线与事后能量误差"),("06_reference_convergence","参考收敛与倍增反例")]:
        lines.append(f'\n![{title}](results/figures/{stem}.png)\n')
    lines += ["\n## 7. 可靠范围与仍需改进\n",
        "已经可靠：给定浅高斯时间分量矢量势下，两模型最低正能束缚态存在；其能量不同，波形平滑、具有正确宇称和指数尾；"
        "三种子 PINN 的能量、归一化与波形误差均已定量核验，参考误差明显较小。\n",
        "仍需改进：Dirac 下分量和尾部 Robin 关系的精度、训练后期停滞、PINN 对优化设置的敏感性。"
        "没有验证更深势阱、超临界现象、负能态或完整激发态谱；没有完成效率优势或通用性论证。"
        "KG 激发态应使用含 (Eₙ+Eₘ−2V) 权重的关系，不能直接加普通 L² 正交惩罚。\n",
        "## 8. 可复现交付与来源\n",
        "代码、配置、测试、原始检查点/曲线/密集预测及参考多项式均位于本目录。运行方式见 [README.md](README.md)，"
        "完整逐种子指标见 [formal_summary.csv](results/formal_summary.csv)，审计见 [audit.json](results/audit.json)。"
        "已有时间演化代码与结果未由本任务改写，也未作为本征态训练标签。\n",
        "物理/方法来源：Hasegawa 的一维标量与矢量 Dirac 势阱研究 "
        "[DOI:10.1016/j.physe.2014.01.011](https://doi.org/10.1016/j.physe.2014.01.011)；"
        "Aquino Curi 等的 KG 耦合与电荷讨论 [DOI:10.1140/epjp/i2019-12571-8](https://doi.org/10.1140/epjp/i2019-12571-8)；"
        "Tisseur–Meerbergen 二次本征值综述 [DOI:10.1137/S0036144500381988](https://doi.org/10.1137/S0036144500381988)；"
        "Beenakker 等倍增综述 [DOI:10.1002/andp.202300081](https://doi.org/10.1002/andp.202300081)；"
        "Raissi 等 PINN 方法 [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)。"
        "出处、核实链接及适用边界见 [REFERENCES.md](REFERENCES.md)。\n"]
    (ROOT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("Wrote REPORT.md from completed results.")


if __name__=="__main__":main()
