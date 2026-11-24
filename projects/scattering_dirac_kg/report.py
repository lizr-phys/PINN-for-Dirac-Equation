"""Generate the Chinese report from actual raw metrics, including failures."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .physics import ROOT, KINDS, ENERGIES, read_json


def sci(value):
    return "—" if value is None or pd.isna(value) else f"{float(value):.2e}"


def mean_sd(values):
    values = np.asarray(values, float)
    return f"{values.mean():.2e} ± {values.std(ddof=1):.1e}"


def build_report():
    fixed = pd.read_csv(ROOT/"results"/"formal_v1_summary.csv")
    baseline = pd.read_csv(ROOT/"results"/"mlp_baseline_summary.csv")
    conditional = pd.read_csv(ROOT/"results"/"conditional_v1_summary.csv")
    costs = pd.read_csv(ROOT/"results"/"scan_costs.csv")
    sensitivity = pd.read_csv(ROOT/"results"/"sensitivity_summary.csv")
    gate = read_json(ROOT/"results"/"fixed_validation_gate.json")
    hp = read_json(ROOT/"results"/"reference"/"high_precision"/"verification.json")
    wave_ref = read_json(ROOT/"results"/"reference"/"verification.json")
    separation = pd.read_csv(ROOT/"results"/"reference_separation.csv")
    extrema = read_json(ROOT/"results"/"reference"/"transmission_extrema.json")
    heldout = conditional[~conditional.training_energy]
    figure_manifest = read_json(ROOT/"figures"/"manifest.json")
    lines = ["# 一维 Dirac / Klein–Gordon 定态散射 PINN 实验报告", "",
        "计算日期：2026-10-04。全部数字由实际保存的训练和评价数据生成，失败与开发试验不删除。",
        "", "## 已确认的结果", "",
        f"主随机特征物理网络（PIELM 类）完成两模型×四指定能量×三种子，共 {len(fixed)} 次正式运行，{int(fixed.passed.sum())} 次通过联合验收。最大波函数相对 L² 误差 **{fixed.wave_relative_l2.max():.3e}**，最大 R/T 绝对误差 **{max(fixed.R_absolute_error.max(), fixed.T_absolute_error.max()):.3e}**。常规全权重 MLP 完成同样的 {len(baseline)} 次运行，当前预算下通过 **{int(baseline.passed.sum())}/{len(baseline)}**；失败结果完整保留。",
        "",
        f"条件网络在 41 个能量节点上仅由方程训练；101 点扫描中，三个种子共有 {len(heldout)} 个未训练能量评价样本，通过 {int(heldout.passed.sum())}/{len(heldout)}，最大波形 L² 误差 {heldout.wave_relative_l2.max():.3e}，最大 R/T 绝对误差 {max(heldout.R_absolute_error.max(),heldout.T_absolute_error.max()):.3e}。这只验证同一固定势下能量插值，不能视为对新势的泛化。",
        "",
        "Dirac 与 KG 的散射结果存在可分辨差异；不预设两者相同。基础一维散射求解和随机特征输出层训练都有已有方法基础，本项目定位为可复现验证与方法比较，不宣称原创物理成果，也不预设 PINN 比传统方法更好。",
        "", "## 物理与相位约定", "",
        "ℏ=c=1、m=1、S=0、V(x)=−0.5 exp(−x²/8)，时间分量矢量耦合。E>m 是指定入射能量，网络没有可训练能量。k=√(E²−1)，左入射系数固定为 1+0i，全局参考场使用 exp(ikx)，右侧无入射。Dirac 的单位通量自由旋量为 C_D(1,±k/(E+1))ᵀ，C_D=√((E+1)/(2k))；KG 自由幅度 C_K=1/√(2k)。",
        "",
        "Dirac 流 j=ψ†σ_xψ，KG 流 j=2 Im(φ*φ')、密度 ρ=2(E−V)|φ|²。相同自由介质给出 R=|r|²、T=|t|²。所有波形保持固定入射相位，不作事后符号/相位拟合或幅度归一化。散射态不作全空间 L² 归一化，不使用束缚态的实分量或偶奇宇称。",
        "",
        "复数 Dirac 重建 ψ₂=−iψ₁'/(E−V+1)，消元方程 ψ₁''+[V'/(E−V+1)]ψ₁'+[(E−V)²−1]ψ₁=0；它比 KG 多导数项。原两条复一阶残差在独立网格评价，重建恒等满足的那一条仅作为符号/实现检查。完整渐近式、Robin 条件、自由旋量及可解析方阱公式见 [理论核对](docs/theory.md)。",
        "", "## 参考解可信度", "",
        f"DOP853 反向 Jost 和独立四实分量 BVP 的波形、幅度、相位与流交叉核验；无势检验 r=0、t=1，方势阱对照解析复振幅。扩域、容差和测试网格检验中，可靠细设置最大波形差 {wave_ref['maximum_wave_difference']:.3e}。",
        "",
        f"第一次固定验证的波形通过，但部分复振幅比双精度参考差只大数倍，未通过严格的 100 倍误差分离门。保留原失败门与指标后，增加 60/70 位 Taylor 求解器；阶数、步长与 L=20→24 最大振幅差 **{hp['maximum_amplitude_difference']:.3e}**。固定能量振幅评价采用该参考，波形仍由验证过的 Jost 提供。complex128 存储的振幅参考分辨率保守记为 1e−15，最终 {gate['accepted']}/{gate['attempted']} 固定运行通过参考分离。",
        "",
        "| 模型 | E | 波形最小误差/参考差 | r 最小误差/参考差 | t 最小误差/参考差 |", "|---|---|---|---|---|"]
    for _, s in separation.iterrows():
        lines.append(f"| {s.kind.upper()} | {s.energy:g} | {s.wave_relative_l2_minimum_separation:.1e} | {s.r_complex_error_minimum_separation:.1e} | {s.t_complex_error_minimum_separation:.1e} |")
    analytic = pd.read_csv(ROOT/"results"/"reference"/"analytic_tests.csv")
    lines += ["", f"无势最大 |r|={analytic.free_r_error.max():.2e}、|t−1|={analytic.free_t_error.max():.2e}；方阱最大复振幅差={max(analytic.square_r_error.max(),analytic.square_t_error.max()):.2e}。", "",
        "以上是数值收敛证据，不是严格误差上界。加密评价网格用于检查误差积分与残差是否稳定，不把同一求解器同配置的零差当成独立准确性证明。见 [参考原始数据](results/reference)、[收敛 CSV](results/reference/convergence.csv)、[任意精度记录](results/reference/high_precision)。",
        "", "## 固定能量结果：所有三种子", "",
        "主网络为已知入射场加复网络修正、硬通道边界、固定随机 Gaussian RBF 隐藏层；只训练复输出层。N=32，132 个实可训练参数、64 个冻结隐藏参数，总计 196。方程残差 SVD 最小二乘，无参考标签或流损失。种子反映隐藏特征位置/尺度；与深 MLP 的 Xavier 初始化分布不同。详见 [方法与先验](docs/methods.md)。",
        "",
        "下表误差为均值±样本标准差（n=3），近零 R 使用绝对误差。参考值来自高精度复振幅，所有波函数按固定入射相位评价。",
        "",
        "| 模型 | E | 参考 R | 参考 T | ΔR 绝对误差 | ΔT 绝对误差 | 波形相对 L² | 独立残差 RMS |", "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for energy in ENERGIES:
            s = fixed[(fixed.kind==kind)&(fixed.energy==energy)]
            lines.append(f"| {kind.upper()} | {energy:g} | {s.reference_R.iloc[0]:.10g} | {s.reference_T.iloc[0]:.10g} | {mean_sd(s.R_absolute_error)} | {mean_sd(s.T_absolute_error)} | {mean_sd(s.wave_relative_l2)} | {mean_sd(s.residual_rms)} |")
    lines += ["", "| 模型 | E | r 复振幅误差 | t 复振幅误差 | r 相位误差/rad | t 相位误差/rad | 最大通量和偏差 | 最大边界误差 | 训练秒均值±SD |", "|---|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for energy in ENERGIES:
            s = fixed[(fixed.kind==kind)&(fixed.energy==energy)]
            rp = "不解释（小幅度）" if s.r_phase_error.isna().all() else mean_sd(s.r_phase_error.dropna())
            lines.append(f"| {kind.upper()} | {energy:g} | {mean_sd(s.r_complex_error)} | {mean_sd(s.t_complex_error)} | {rp} | {mean_sd(s.t_phase_error)} | {s.unity_error.max():.2e} | {s.boundary_error.max():.2e} | {mean_sd(s.training_seconds)} |")
    lines += ["", f"主组独立网格最大局部流跨度 {fixed.current_span.max():.3e}；Dirac/ KG 分量误差、反射/透射复数、训练时间、参数量、通道误差及全部种子见 [逐次 CSV](results/formal_v1_summary.csv)。边界硬约束和重建恒等满足的残差达到机器精度是结构性质，不作为求解精度的唯一证明。",
        "", "## 失败及优化局限", "",
        "| 模型 | E | 成功数/三种子 | 波形 L² 均值±SD | 残差 RMS 均值±SD | 边界误差最大 | 训练秒均值±SD |", "|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        for energy in ENERGIES:
            s = baseline[(baseline.kind==kind)&(baseline.energy==energy)]
            lines.append(f"| {kind.upper()} MLP | {energy:g} | {int(s.passed.sum())}/3 | {mean_sd(s.wave_relative_l2)} | {mean_sd(s.residual_rms)} | {s.boundary_error.max():.2e} | {mean_sd(s.training_seconds)} |")
    lines += ["",
        f"常规 MLP 的全部权重使用 Adam 500→L-BFGS 250、软边界权重 10，直接表达复总场。Dirac 可训练参数 {int(baseline[baseline.kind=='dirac'].total_parameters.iloc[0])}、KG {int(baseline[baseline.kind=='kg'].total_parameters.iloc[0])}，无冻结隐藏参数。当前预算下的残差和小边界误差仍不足以确定正确散射波形、R/T 及相位，近阈值条件性与振荡表示均可能影响收敛。这里只能确认当前配置的优化/精度失败；没有 NTK 或 Hessian 因果分析，不声称识别了全部失败机制，更不声称所有深 PINN 都失败。",
        "",
        "开发低容量（8 个 RBF 单元）在 E=2 的 Dirac 试验具有精确入射/出射边界，却出现严重透射与波形错误，直接说明边界正确不是充分精度证明。低阶条件原型在靠近阈值的未训练能量也未通过流/波形联合标准。全部开发模型、残差、训练曲线、错误和失败门在 `results/pilot_*`、`results/conditional_pilot` 和 `results/archived_reference_resolution_gate`。首次自动微分实现的复/实矩阵 dtype 问题已由独立 AD 测试发现并修复；修复前源码归档。",
        "",
        "早期条件原型在正式固定门完成前曾作开发试验，这是开发阶段的顺序偏离，单列保留，不混入正式统计。正式条件组只在固定门通过后执行。MLP 训练期间发生过评价代码短暂编辑和文件行尾转换；已恢复实际进程导入的原始源码，并记录元数据哈希校正，模型和历史不改动。",
        "", "## 条件网络与未训练能量", "",
        "空间随机特征相同，但每个复输出系数由 k(E) 的 20 阶 Chebyshev 特征统一表达；2772 个实可训练参数、64 个冻结参数。41 个 k 的 Lobatto 训练节点、每能量 256 个空间配点，联合物理最小二乘。推理输入 (x,E)，不再解方程、逐能量拟合或更新权重。三个种子全部评价，扫描步长 ΔE=0.0099。",
        "",
        "| 模型 | 未训练能量数/种子 | 通过/全部样本 | 波形 L² 均值±SD | 最大波形 L² | 最大 R 绝对误差 | 最大 T 绝对误差 | 最大复 t 误差 |", "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        s = heldout[heldout.kind==kind]
        lines.append(f"| {kind.upper()} | {s.energy.nunique()} | {int(s.passed.sum())}/{len(s)} | {mean_sd(s.wave_relative_l2)} | {s.wave_relative_l2.max():.2e} | {s.R_absolute_error.max():.2e} | {s.T_absolute_error.max():.2e} | {s.t_complex_error.max():.2e} |")
    lines += ["", "此处跨能量 SD 不是纯种子方差。每能量各三种子的均值、样本 SD 和范围另在 aggregate CSV，表中跨能量结果只描述误差分布。训练能量和未训练能量逐行标明；小 |r| 相位不解释。",
        "", "## 扫描成本：实际整条运行", "",
        "| 模型 | 方法 | 扫描能量 | 训练/积分秒 | 全部推理秒 | 总秒 | 最大波形 L² | 最大 T 绝对误差 |", "|---|---|---|---|---|---|---|---|"]
    for _, row in costs.iterrows():
        lines.append(f"| {row.kind.upper()} | {row.method} | {int(row.scan_energies)} | {row.training_seconds:.3f} | {row.inference_seconds:.3f} | {row.total_seconds:.3f} | {sci(row.maximum_wave_relative_l2)} | {sci(row.maximum_T_error)} |")
    lines += ["", "逐能量重训列为每模型种子 0 的 101 次独立实际训练；条件网络列为三个种子分别进行一次训练加 101 能量推理后的成本均值。传统列同时给出高精度 Jost 和实际 rtol=1e−8 的扫描求解及同网格波形采样，较松容差解的真实误差也列出。各方法精度不同；不以推理时间掩盖训练，也不以高容差传统参考的成本代替最快传统算法。",
        "",
        "较松容差 Jost 的实际扫描约 6.3–6.5 秒，最大波形误差约 2.6e−8；本轮逐能量 RBF 约 11 秒，条件网络约 19–20 秒，最大波形误差均更大。在此固定势、101 点网格与实际背景负载下，传统方法给出更小的最大波形误差和更低实测总成本。条件网络没有在本轮扫描规模下摊薄联合训练成本；更大查询量的盈亏点需要另测，不能直接外推。",
        "",
        "训练时间含配点/残差矩阵构造及求解，不含导入、保存、开发和参考评价；传统时间含积分与场采样，不含导入/保存。机器上已有 R12 五个训练 worker 持续运行，部分开发参考与 MLP 也并发；表中为背景负载下的实测墙钟时间，不能称为隔离资源的严格公平速度排名。需要在空闲机器重复效率验证。详细成本和口径见 [扫描成本](results/scan_costs.csv)、[范围说明](results/cost_scope.json)。当前结果没有证据证明 PINN 全面优于传统方法。",
        "", "## 透射极值与解释范围", "",
        "先用每模型 1001 个参考能量搜索局部 R 最小值，再连续局部优化并以 L=20、更小步长和更紧容差核验。以下为实能轴上确认的透射极大值，不等同于已找到复能共振极点。",
        "", "| 模型 | 透射极大 E | R | T | 独立设置复 t 差 |", "|---|---|---|---|---|"]
    for e in extrema:
        lines.append(f"| {e['kind'].upper()} | {e['energy']:.10f} | {e['R']:.3e} | {e['T']:.12f} | {e['crosscheck_t_difference']:.2e} |")
    if not extrema:
        lines.append("| — | 该扫描未发现可确认的内部局部峰 | — | — | — |")
    lines += ["", "没有解析延拓、S 矩阵复能极点、寿命或束缚态关联计算；不得据这些峰宣称束缚态能量与透射峰一一对应。精细极值搜索成本独立于 101 点扫描基准。",
        "", "## 区间与配点敏感性", "",
        "两模型、E=1.02/2、种子 0 分别加密空间配点至 768 或扩大训练半区间至 20；保留全部结果。不是多种子敏感性验证，扩大区间后固定特征数的覆盖密度变化也会影响误差。",
        "", "| 设置 | 模型 | E | 波形 L² | 残差 RMS | T 绝对误差 | 成功 |", "|---|---|---|---|---|---|---|"]
    for _, s in sensitivity.iterrows():
        lines.append(f"| {s.group} | {s.kind.upper()} | {s.energy:g} | {s.wave_relative_l2:.2e} | {s.residual_rms:.2e} | {s.T_absolute_error:.2e} | {bool(s.passed)} |")
    lines += ["", "## 图表与来源", ""]
    for f in figure_manifest:
        name = f["name"]
        lines += [f"![{f['caption']}](figures/{name}.png)", "", f"{f['caption']} 原始数据："+"、".join(f"`{s}`" for s in f["sources"])+"。", ""]
    lines += ["## 复现、证据和待改进问题", "",
        "代码、配置、测试、每次模型与配点、奇异值/条件数、训练曲线、4001/1601 点复波形、参考和所有残差均在本目录。NPZ 使用 NumPy 读取，无 pickle；MLP `model.pt` 为 state_dict，使用 weights_only=True 加载。",
        "",
        "```powershell", "Set-Location 'E:\\PINN for Quantum Physics\\projects'", "python -m pytest scattering_dirac_kg/tests -q", "python -m scattering_dirac_kg.run --stage all", "```", "",
        "相同配置与源码的完成训练会复用；源代码或训练配置改变时拒绝静默覆盖，使用新运行组。参考与扫描阶段复用已验证原始输出。全新重训和源码版本复制流程见 README；不能仅改配置却沿用旧参考或门控。",
        "",
        "待改进：为全权重 MLP 开展预算与表示消融、近阈值的自适应能量分辨率、随机特征条件数和压缩、同精度传统方法成本比较、更广势函数与更大区间覆盖。当前没有三维、强场、超临界、反粒子通道或复能共振验证。",
        "",
        "原始相关论文、具体支持范围和 DOI 见 [文献核实](docs/references.md)，原始 Crossref 响应及规范化说明已保存。现有四项目的 15,202 个基线文件中 15,150 个字节一致；52 个变化均来自开始前已运行的 R12 训练输出，零文件丢失。束缚态代码和数据全部字节一致；本任务没有改写旧项目。完整差异及进程证据见 [保留审计说明](results/preservation_assessment.json)，不能声称整个活跃旧目录字节冻结。新交付完整性见 [审计](results/audit.json) 和 SHA256 清单；最终测试日志另存 [tests.txt](results/tests.txt)。"]
    (ROOT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    build_report()
