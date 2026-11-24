"""All figures use actual saved outputs; no synthetic or selected seed curves."""
from __future__ import annotations
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .physics import ROOT, KINDS, ENERGIES, energy_name, save_json, read_json

BLUE, ORANGE, GRAY = "#276D9B", "#BB6B36", "#515861"
COLORS = dict(dirac=BLUE, kg=ORANGE)
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
    "font.size":7, "axes.titlesize":8, "axes.labelsize":7, "legend.fontsize":6,
    "svg.fonttype":"none", "pdf.fonttype":42, "axes.spines.top":False,
    "axes.spines.right":False, "axes.linewidth":.7, "lines.linewidth":1.,
    "legend.frameon":False, "savefig.facecolor":"white"})


def panel_labels(axes):
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.text(-.1, 1.04, string.ascii_lowercase[i], transform=ax.transAxes, fontweight="bold", fontsize=8)


def segmented_phase(amplitudes, floor=1e-5):
    values = np.asarray(amplitudes)
    phase = np.full(len(values), np.nan)
    valid = abs(values)>=floor
    breaks = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for left, right in zip(breaks[::2], breaks[1::2]):
        phase[left:right] = np.unwrap(np.angle(values[left:right]))
    return phase


def make_figures():
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    manifest = []

    def save(fig, name, caption, sources):
        fig.tight_layout(pad=1.3)
        paths = []
        for suffix in ("png", "svg", "pdf"):
            path = out/f"{name}.{suffix}"
            fig.savefig(path, dpi=220 if suffix=="png" else 600, bbox_inches="tight")
            paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        plt.close(fig)
        manifest.append(dict(name=name, files=paths, caption=caption, sources=sources,
             statistics="All seeds; mean/sample SD where shown; no significance tests.",
             backend="Python/matplotlib", image_integrity="No data editing; logarithmic display uses a 1e-16 floor only."))

    ref = pd.read_csv(ROOT/"results"/"reference"/"scan.csv")
    cond = pd.read_csv(ROOT/"results"/"conditional_v1_summary.csv")
    fixed = pd.read_csv(ROOT/"results"/"formal_v1_summary.csv")
    baseline = pd.read_csv(ROOT/"results"/"mlp_baseline_summary.csv")
    train_nodes = np.array(read_json(ROOT/"results"/"conditional_validation.json")["training_energy_nodes"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for row, kind in enumerate(KINDS):
        r = ref[ref.kind==kind]
        for j, metric in enumerate(("R", "T")):
            ax = axes[row, j]
            ax.plot(r.energy, r[metric], color=GRAY, label="Jost reference")
            for seed in (0, 1, 2):
                c = cond[(cond.kind==kind)&(cond.seed==seed)].sort_values("energy")
                ax.plot(c.energy, c[metric], color=COLORS[kind], alpha=.65, linestyle=("--",":","-.")[seed], label=f"conditional seed {seed}")
            ax.plot(train_nodes, np.full_like(train_nodes, .035), "|", transform=ax.get_xaxis_transform(), color="black", markersize=3)
            ax.set(xlabel="Incident energy E / m", ylabel=metric, title=f"{kind.upper()} ({'reflection' if j==0 else 'transmission'})")
            if j==0:
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-13)
            else:
                ax.set_ylim(.82, 1.005)
            if row==0 and j==0:
                ax.legend(loc="lower left")
    panel_labels(axes)
    save(fig, "01_scattering_scan", "两模型参考与三个条件网络的 R(E)、T(E)；底部刻线为训练能量，扫描点多数为未训练能量。", ["results/reference/scan.csv", "results/conditional_v1_summary.csv"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    for row, kind in enumerate(KINDS):
        r = ref[ref.kind==kind].sort_values("energy")
        for col, amplitude in enumerate(("t", "r")):
            values = r[amplitude+"_real"].to_numpy()+1j*r[amplitude+"_imag"].to_numpy()
            axes[row, col].plot(r.energy, segmented_phase(values, 1e-5 if amplitude=="r" else 0.), color=GRAY, label="Reference")
            for seed in (0,1,2):
                c = cond[(cond.kind==kind)&(cond.seed==seed)].sort_values("energy")
                values = c[amplitude+"_real"].to_numpy()+1j*c[amplitude+"_imag"].to_numpy()
                axes[row, col].plot(c.energy, segmented_phase(values, 1e-5 if amplitude=="r" else 0.), linestyle=("--", ":", "-.")[seed], alpha=.7, color=COLORS[kind])
            axes[row,col].set(xlabel="E / m", ylabel=f"arg({amplitude}) / rad", title=kind.upper())
    axes[0,0].legend()
    panel_labels(axes)
    save(fig, "02_scattering_phases", "固定入射相位后的透射与反射相位；各连续有效段展开，|r|<10⁻⁵ 的反射相位屏蔽，段间不解释相位跳跃。", ["results/reference/scan.csv", "results/conditional_v1_summary.csv"])

    for kind in KINDS:
        components = 2 if kind=="dirac" else 1
        fig, axes = plt.subplots(4, 2*components, figsize=(7.2, 7.2 if kind=="dirac" else 6.4), squeeze=False, sharex=True)
        for row, energy in enumerate(ENERGIES):
            arrays = [np.load(ROOT/"results"/"formal_v1"/kind/energy_name(energy)/f"seed_{seed}"/"evaluation.npz") for seed in (0,1,2)]
            x = arrays[0]["x"]
            for component in range(components):
                for part in (0,1):
                    ax = axes[row, 2*component+part]
                    operation = np.real if part==0 else np.imag
                    ax.plot(x, operation(arrays[0]["reference"][:, component]), color=GRAY, label="Reference")
                    for seed, d in enumerate(arrays):
                        ax.plot(x, operation(d["state"][:, component]), color=COLORS[kind], alpha=.55, linestyle=("--",":","-.")[seed], label=f"Seed {seed}")
                    symbol = f"psi{component+1}" if kind=="dirac" else "phi"
                    ax.set_title(f"E={energy:g}; {'Re' if part==0 else 'Im'}({symbol})")
                    if row==3:
                        ax.set_xlabel("x m")
                    if part==0:
                        ax.set_ylabel("Unit-flux field")
        axes[0,0].legend(ncol=2)
        panel_labels(axes)
        save(fig, f"03_{kind}_complex_components", "四个指定能量的复波函数逐分量，三种子全部显示；重合曲线表示误差低于此尺度的可见度，误差另图显示。", [f"results/formal_v1/{kind}/E_*/seed_*/evaluation.npz"])

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6))
    for row, kind in enumerate(KINDS):
        for col, energy in enumerate(ENERGIES):
            ax = axes[row,col]
            for seed in (0,1,2):
                d = np.load(ROOT/"results"/"formal_v1"/kind/energy_name(energy)/f"seed_{seed}"/"evaluation.npz")
                error = abs(d["error"][:,0]) if kind=="kg" else np.linalg.norm(d["error"], axis=1)
                ax.semilogy(d["x"], np.maximum(error,1e-16), color=COLORS[kind], linestyle=("-","--",":")[seed], alpha=.7)
            ax.set(xlabel="x m", ylabel="|Field error|", title=f"{kind.upper()} E={energy:g}")
    panel_labels(axes)
    save(fig, "04_pointwise_errors", "固定入射相位后的逐点波形误差，Dirac 为复旋量模、KG 为复 φ 的模；全部三种子，不进行事后缩放。", ["results/formal_v1/*/E_*/seed_*/evaluation.npz"])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    metrics = ["wave_relative_l2", "residual_rms", "T_absolute_error"]
    labels = ["Relative field L2", "Independent residual RMS", "Absolute T error"]
    for col, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[col]
        for kind_index, kind in enumerate(KINDS):
            for family_index, (data, family) in enumerate(((fixed,"RBF"),(baseline,"MLP"))):
                selected = data[data.kind==kind]
                values = selected[metric].to_numpy()
                xx = kind_index*2+family_index
                ax.scatter(xx+np.linspace(-.15,.15,len(values)), np.maximum(values,1e-16), s=12,
                           facecolor=COLORS[kind] if family=="RBF" else "none", edgecolor=COLORS[kind], linewidth=.65)
                ax.plot([xx-.2,xx+.2], [values.mean()]*2, color="black", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xticks(range(4), ["D RBF", "D MLP", "K RBF", "K MLP"], rotation=25)
        ax.set_ylabel(label)
    panel_labels(axes)
    save(fig, "05_seed_error_distribution", "全部固定能量与三种子的误差分布；每组 n=12（四能量×三种子），短黑线为算术均值；失败 MLP 不剔除。组间不是等参数单因素对照。", ["results/formal_v1_summary.csv", "results/mlp_baseline_summary.csv"])

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.7))
    for row, kind in enumerate(KINDS):
        for col, energy in enumerate(ENERGIES):
            ax = axes[row,col]
            for seed in (0,1,2):
                d = np.load(ROOT/"results"/"formal_v1"/kind/energy_name(energy)/f"seed_{seed}"/"evaluation.npz")
                ax.plot(d["x"], d["current"], color=COLORS[kind], alpha=.7, linestyle=("-","--",":")[seed])
            refT = fixed[(fixed.kind==kind)&(fixed.energy==energy)].reference_T.iloc[0]
            ax.axhline(refT, color=GRAY, linewidth=.7, linestyle="--")
            ax.ticklabel_format(axis="y", useOffset=True, style="sci", scilimits=(-3,3))
            ax.set(xlabel="x m", ylabel="Local j(x)", title=f"{kind.upper()} E={energy:g}")
    panel_labels(axes)
    save(fig, "06_local_current", "单位入射通量下 j(x) 和参考透射流；坐标科学计数偏移使微小流变化可见。守恒为辅助证据，训练没有流损失。", ["results/formal_v1/*/E_*/seed_*/evaluation.npz"])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for i, metric in enumerate(("wave_relative_l2", "T_absolute_error", "unity_error")):
        for kind in KINDS:
            selected = cond[(cond.kind==kind)&(~cond.training_energy)].sort_values("energy")
            for seed in (0,1,2):
                s = selected[selected.seed==seed]
                axes[i].semilogy(s.energy, np.maximum(s[metric],1e-16), color=COLORS[kind], alpha=.6, linestyle=("-","--",":")[seed])
        axes[i].axhline(5e-4, color=GRAY, linestyle="--", linewidth=.7)
        axes[i].set(xlabel="Held-out E / m", ylabel=metric.replace("_"," "))
    panel_labels(axes)
    save(fig, "07_conditional_heldout_errors", "未训练能量上三种子条件网络的波形、T 绝对误差与流守恒偏差；水平线为统一验收阈值。", ["results/conditional_v1_summary.csv"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    for row, kind in enumerate(KINDS):
        for energy in ENERGIES:
            for seed in (0,1,2):
                directory = ROOT/"results"/"formal_v1"/kind/energy_name(energy)/f"seed_{seed}"
                h = pd.read_csv(directory/"history.csv")
                axes[row,0].semilogy(h.step, h.physics_loss, color=COLORS[kind], alpha=.45)
                directory = ROOT/"results"/"mlp_baseline"/kind/energy_name(energy)/f"seed_{seed}"
                h = pd.read_csv(directory/"history.csv")
                axes[row,1].semilogy(h.step, h.loss, color=COLORS[kind], alpha=.4)
        axes[row,0].set(title=f"{kind.upper()} RBF direct output-layer solve", xlabel="Solve/refinement stage", ylabel="Physics objective")
        axes[row,1].set(title=f"{kind.upper()} total-field MLP", xlabel="Objective evaluations", ylabel="Physics + soft BC objective")
    panel_labels(axes)
    save(fig, "08_training_curves", "全部固定能量/种子的训练记录。RBF 为直接解及细化阶段，MLP 为 Adam/L-BFGS 目标调用；损失尺度不同，不比较曲线绝对值。", ["results/formal_v1/*/E_*/seed_*/history.csv", "results/mlp_baseline/*/E_*/seed_*/history.csv"])

    benchmark = pd.read_csv(ROOT/"results"/"scan_costs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, kind in zip(axes, KINDS):
        s = benchmark[benchmark.kind==kind]
        ax.bar(range(len(s)), s.total_seconds, color=[GRAY, GRAY, COLORS[kind], COLORS[kind]], alpha=.85)
        ax.set_xticks(range(len(s)), s.method, rotation=20, ha="right")
        ax.set(title=f"{kind.upper()} 101-energy scan", ylabel="Measured total solve + inference / s")
    panel_labels(axes)
    save(fig, "09_scan_total_cost", "同一扫描的传统积分、逐能量重训、条件网络训练加全部推理成本；条件网络为三个种子成本的均值。机器已有 R12 后台训练，时间为实际负载下的墙钟值；准确度不同，不是隔离资源的速度排名。", ["results/scan_costs.csv"])

    convergence = pd.read_csv(ROOT/"results"/"reference"/"convergence.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for kind in KINDS:
        d = convergence[(convergence.kind==kind)&(convergence.grid_points==4001)]
        lengths = [10,12,16,20]
        errors = [d[d.variant==f"domain_{L}"].wave_l2_difference.max() for L in lengths]
        axes[0].semilogy(lengths, np.maximum(errors,1e-16), "o-", color=COLORS[kind], label=kind.upper(), markersize=3)
        tolerances = [1e-8,1e-10,3e-13]
        errors = [d[d.variant==f"tolerance_{tol:g}"].wave_l2_difference.max() for tol in tolerances]
        axes[1].loglog(tolerances, np.maximum(errors,1e-16), "o-", color=COLORS[kind], markersize=3)
        grids = [1001,2001,4001]
        errors = [convergence[(convergence.kind==kind)&(convergence.grid_points==n)&(convergence.variant=="bvp")].wave_l2_difference.max() for n in grids]
        axes[2].semilogy(grids, np.maximum(errors,1e-16), "o-", color=COLORS[kind], markersize=3)
    axes[0].set(xlabel="Reference half domain L", ylabel="Max field L2 difference")
    axes[1].set(xlabel="Jost relative tolerance", ylabel="Max field L2 difference")
    axes[2].set(xlabel="Independent test points", ylabel="Max BVP/Jost L2 difference")
    axes[0].legend()
    panel_labels(axes)
    save(fig, "10_reference_convergence", "参考区间、容差和测试网格加密证据；值为四个固定能量的最大差异。零差仅为同算法同设置的恒等复算，不证明准确性。", ["results/reference/convergence.csv"])
    save_json(out/"manifest.json", manifest)


if __name__ == "__main__":
    make_figures()
