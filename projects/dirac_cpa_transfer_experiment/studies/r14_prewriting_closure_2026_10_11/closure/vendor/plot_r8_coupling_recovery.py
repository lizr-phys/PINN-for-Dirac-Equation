from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .r8_config import ROOT, RAW, FIGURES, CANDIDATES, CONTROLS, THRESHOLDS
from .r8_decision import read_stage
from .freeze_r7_phase_release import write_json, sha256

LABELS = {"A_carrier_complex": "A: carrier + complex", "B_free_correction": "B: free + correction",
          "C_release_windows": "C: release + windows", "D_carrier_windows": "D: carrier + windows",
          "complex": "Complex", "phase_release": "Phase release",
          "A_identity_carrier": "A: carrier removed", "D_identity_carrier": "D: carrier removed",
          "B_static_free": "B: static free part", "window_complex": "Window Complex"}
SHORT = {"A_carrier_complex": "A", "B_free_correction": "B", "C_release_windows": "C", "D_carrier_windows": "D",
         "complex": "Complex", "phase_release": "Release", "A_identity_carrier": "A−p",
         "D_identity_carrier": "D−p", "B_static_free": "B−free", "window_complex": "Complex\nwindows"}
COLORS = {"A_carrier_complex": "#326E97", "B_free_correction": "#8C76A8", "C_release_windows": "#378677",
          "D_carrier_windows": "#C56965", "complex": "#787878", "phase_release": "#B58146",
          "A_identity_carrier": "#888888", "D_identity_carrier": "#888888", "B_static_free": "#888888", "window_complex": "#555555"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42})


def finish(fig, name, sources, caption, scope):
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES/"source_data").mkdir(exist_ok=True)
    source_paths = []
    for suffix, table in sources.items():
        p = FIGURES/"source_data"/(name+"_"+suffix+".csv")
        table.to_csv(p, index=False)
        source_paths.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p), rows=len(table)))
    for index, axis in enumerate(a for a in fig.axes if a.axison and a.get_label() != "<colorbar>"):
        axis.text(-.08, 1.025, chr(97+index), transform=axis.transAxes, weight="bold", fontsize=10, va="bottom")
    fig.canvas.draw()
    exports = []
    for ext in ("png", "svg", "pdf"):
        p = FIGURES/(name+"."+ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        exports.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)))
    plt.close(fig)
    return dict(figure_id=name, caption=caption, seed_scope=scope, source_data=source_paths,
                exports=exports, archetype="quantitative grid or field plate + quantitative evidence",
                statistics="all registered seeds; mean +/- population SD, ddof=0; no significance inference")


def primary(frame, names, name, scope):
    fields = [("spinor_rel_l2_gauge_aligned", "Spinor relative L2"), ("current_rel_l2", "Current relative L2"),
              ("fidelity_mean", "Mean fidelity"), ("modal_population_rel_l2", "Modal population relative L2"),
              ("max_norm_drift", "Maximum norm drift"), ("sideband_complex_coefficient_rel_l2", "First-sideband complex coefficient L2")]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.5), layout="constrained")
    for ax, (key, title) in zip(axes.flat, fields):
        extents = []
        for i, model in enumerate(names):
            group = frame[frame.model == model].sort_values("seed")
            v = group[key].to_numpy()
            extents.extend([*v, v.mean()-v.std(ddof=0), v.mean()+v.std(ddof=0)])
            ax.scatter(i+np.linspace(-.15, .15, len(v)), v, c=COLORS[model], s=21, zorder=3)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=0), fmt="_", c="#222222", capsize=3, zorder=4)
        if key in THRESHOLDS:
            ax.axhline(THRESHOLDS[key], ls=":", c="#B35555", lw=1)
        if key == "fidelity_mean":
            ax.set_ylim(0, 1.035)
        else:
            if min(extents) > 0:
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=1e-4)
                ax.set_ylim(bottom=min(0, min(extents))-.03*max(extents))
        ax.set(title=title, xticks=range(len(names)), xticklabels=[SHORT[n] for n in names])
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(scope+" | all seeds; black mean ± population SD; dotted absolute thresholds", fontsize=10)
    return finish(fig, name, {"runs": frame}, "Joint physical accuracy, not loss-only or best-seed evidence.", scope)


def dense_for(model, stage="validation", seed=100, case="main", balanced=False):
    name = model+("_balanced" if balanced else "")
    p = RAW/stage/case/name/f"seed_{seed}"/"dense_grid.npz"
    with np.load(p) as data:
        return {key: data[key].copy() for key in data.files}


def modal_dynamics(selected, validation):
    dense = {n: dense_for(n) for n in selected}
    first = dense[selected[0]]
    nt, nk = first["modal_occupations_exact"].shape
    vmax = max(1., max(float(d["modal_occupations_predicted"].max()) for d in dense.values()))
    error_max = max(float(abs(d["modal_occupations_predicted"]-d["modal_occupations_exact"]).max()) for d in dense.values())
    fig = plt.figure(figsize=(10.5, 3.1+2.25*len(selected)), layout="constrained")
    grid = fig.add_gridspec(len(selected)+1, 3)
    sources, im = [], None
    for row, n in enumerate(["reference"]+selected):
        d = first if n == "reference" else dense[n]
        pop = d["modal_occupations_exact"] if n == "reference" else d["modal_occupations_predicted"]
        for col, (values, title) in enumerate(((pop, "Reference" if n=="reference" else LABELS[n]),
            (np.zeros_like(pop) if n=="reference" else abs(pop-d["modal_occupations_exact"]), "Absolute population error"))):
            ax = fig.add_subplot(grid[row, col])
            im = ax.pcolormesh(d["t"], d["mode_offsets"], values.T, shading="nearest",
                rasterized=True, antialiased=False,
                cmap="viridis" if col==0 else "magma", vmin=0, vmax=vmax if col==0 else max(error_max, 1e-12))
            ax.set(xlabel="t", ylabel="Mode offset j", title=title)
            fig.colorbar(im, ax=ax, label="Population" if col==0 else "Absolute error")
        ax = fig.add_subplot(grid[row, 2])
        for offset, style in ((0, "-"), (-1, "--"), (1, ":")):
            j = np.flatnonzero(d["mode_offsets"] == offset)[0]
            ax.plot(d["t"], d["modal_occupations_exact"][:, j], c="black", ls=style, lw=1.15, alpha=.6)
            if n != "reference":
                ax.plot(d["t"], pop[:, j], c=COLORS[n], ls=style, lw=1.25)
        ax.set(xlabel="t", ylabel="Population", title="Center / first sidebands", ylim=(-.02, vmax*1.03))
        if row==0:
            ax.legend(handles=[Line2D([], [], c="black", ls=style, label=f"j={off:+d}") for off, style in ((0,"-"),(-1,"--"),(1,":"))], fontsize=7)
        for i, time in enumerate(d["t"]):
            for j, offset in enumerate(d["mode_offsets"]):
                sources.append(dict(model=n, seed=100, t=time, mode_offset=int(offset), reference=d["modal_occupations_exact"][i,j], predicted=pop[i,j], absolute_error=abs(pop[i,j]-d["modal_occupations_exact"][i,j])))
    return finish(fig, "03_modal_dynamics", {"modal_grid": pd.DataFrame(sources)},
        "Fixed seed 100 tests actual sideband generation and center depletion on common population scales.", "independent validation, fixed seed 100")


def branch_coherence(selected, validation):
    dense = {n: dense_for(n) for n in selected}
    first = dense[selected[0]]
    fig, axes = plt.subplots(3, 3, figsize=(11.1, 8.5), layout="constrained")
    rows = []
    for col, offset in enumerate((0, -1, 1)):
        j = int(np.flatnonzero(first["mode_offsets"]==offset)[0])
        for b, style in ((0,"-"), (1,"--")):
            axes[0,col].plot(first["t"], first["branch_occupations_exact"][:, j,b], color="black", ls=style, lw=1, alpha=.55)
            for n, d in dense.items():
                axes[0,col].plot(d["t"], d["branch_occupations_predicted"][:,j,b], color=COLORS[n], ls=style, lw=1.15)
                cp, ce = d["branch_coefficients_predicted_gauge_aligned"][:,j,b], d["branch_coefficients_exact"][:,j,b]
                valid = d["branch_phase_evaluation_mask"][:,j,b]
                for i, time in enumerate(d["t"]):
                    rows.append(dict(model=n, seed=100, mode_offset=offset, branch=b, t=time,
                        reference_real=ce[i].real, reference_imag=ce[i].imag,
                        predicted_real=cp[i].real, predicted_imag=cp[i].imag,
                        reference_occupation=abs(ce[i])**2, predicted_occupation=abs(cp[i])**2,
                        reference_phase_active=bool(d["reference_active_branch_mask"][i,j,b]), phase_valid=bool(valid[i]),
                        phase_error_rad=d["branch_phase_error_rad_masked_nan"][i,j,b]))
        axes[0,col].set(title=f"Free-basis branch populations, j={offset:+d}", xlabel="t", ylabel="Branch population")
        ce = first["branch_coefficients_exact"][:,j,0]
        axes[1,col].plot(first["t"], ce.real, color="black", lw=1.05, label="reference Re")
        axes[1,col].plot(first["t"], ce.imag, color="black", ls="--", lw=1.05, label="reference Im")
        for n,d in dense.items():
            cp = d["branch_coefficients_predicted_gauge_aligned"][:,j,0]
            axes[1,col].plot(d["t"], cp.real, color=COLORS[n], lw=1.05)
            axes[1,col].plot(d["t"], cp.imag, color=COLORS[n], ls="--", lw=1.05)
        axes[1,col].set(title=f"Complex + coefficient, j={offset:+d}", xlabel="t", ylabel="Re / Im coefficient")
        center = int(np.flatnonzero(first["mode_offsets"]==0)[0])
        flat_center, flat_j = 2*center, 2*j+1
        reference = first["channel_coherence_exact"][:, flat_center, flat_j]
        mask = first["reference_active_pair_mask"][:, flat_center, flat_j]
        axes[2,col].plot(first["t"], np.where(mask, reference.real, np.nan), color="black", lw=1.1)
        axes[2,col].plot(first["t"], np.where(mask, reference.imag, np.nan), color="black", ls="--", lw=1.1)
        for n,d in dense.items():
            predicted = d["channel_coherence_predicted"][:,flat_center,flat_j]
            axes[2,col].plot(d["t"], np.where(mask, predicted.real, np.nan), color=COLORS[n], lw=1.05)
            axes[2,col].plot(d["t"], np.where(mask, predicted.imag, np.nan), color=COLORS[n], ls="--", lw=1.05)
            for i,time in enumerate(d["t"]):
                rows.append(dict(model=n, seed=100, mode_offset=offset, branch="coherence", t=time,
                    reference_real=reference[i].real, reference_imag=reference[i].imag,
                    predicted_real=predicted[i].real, predicted_imag=predicted[i].imag,
                    reference_phase_active=bool(mask[i]), phase_valid=bool(mask[i])))
        axes[2,col].set(title="Center + / selected − coherence", xlabel="t", ylabel="Re / Im coherence")
    handles = [Line2D([], [], c="black", label="reference"), *[Line2D([], [], c=COLORS[n], label=LABELS[n]) for n in selected],
               Line2D([], [], c="black", ls="--", label="Im / negative branch")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    return finish(fig, "04_branch_coherence", {"complex_and_mask":pd.DataFrame(rows), "all_seed_metrics":validation},
        "Complex sidebands and gauge-invariant coherences; reference-inactive phase samples are not scored.", "fixed seed 100; all five seeds in source metrics")


def fields(selected):
    dense = {n:dense_for(n) for n in selected}
    first = dense[selected[0]]
    fig, axes = plt.subplots(1+2*len(selected), 2, figsize=(10.5, 2.0*(1+2*len(selected))+1), layout="constrained")
    sources = []
    for col, key in enumerate(("rho", "current")):
        vmax = max(float(abs(d[key+"_predicted"]).max()) for d in dense.values())
        vmax = max(vmax, float(abs(first[key+"_exact"]).max()))
        error_vmax = max(float(abs(d[key+"_predicted"]-d[key+"_exact"]).max()) for d in dense.values())
        panels = [("reference", first[key+"_exact"], False)]
        for n,d in dense.items():
            panels += [(n,d[key+"_predicted"], False),(n,d[key+"_predicted"]-d[key+"_exact"], True)]
        for row,(n, values, is_error) in enumerate(panels):
            ax=axes[row,col]
            image=ax.pcolormesh(first["x"], first["t"], abs(values) if is_error else values, shading="nearest",
                rasterized=True, antialiased=False,
                cmap="magma" if is_error else ("viridis" if key=="rho" else "RdBu_r"),
                vmin=0 if is_error or key=="rho" else -vmax,
                vmax=max(error_vmax,1e-12) if is_error else vmax)
            ax.set(title=("Reference" if n=="reference" else LABELS[n]+(" | absolute error" if is_error else ""))+" : "+("density" if key=="rho" else "current"), xlabel="x", ylabel="t")
            fig.colorbar(image, ax=ax, label="Absolute error" if is_error else key)
            for i,time in enumerate(first["t"]):
                for j,x in enumerate(first["x"]):
                    sources.append(dict(observable=key, panel_model=n, absolute_error_panel=is_error,
                                        seed=100, t=time, x=x, value=float(abs(values[i,j]) if is_error else values[i,j])))
    return finish(fig, "05_density_current", {"physical_grid":pd.DataFrame(sources)},
        "Reference, predictions and absolute field errors, fixed seed 100 with shared scales.", "fixed seed 100")


def carrier_gradient(validation):
    selected = [n for n in validation.model.unique() if (RAW/"validation/main"/n/"seed_100/carrier_diagnostics.csv").exists()]
    fig, axes=plt.subplots(2,2,figsize=(10.5,6.5),layout="constrained")
    sources, gradrows, residualrows = [], [], []
    for n in validation.model.unique():
        part=validation[validation.model==n]
        if n in selected:
            for _,run in part.iterrows():
                table=pd.read_csv(RAW/run.run_path/"carrier_diagnostics.csv")
                table["model"],table["seed"]=n,int(run.seed)
                sources.append(table)
            d=dense_for(n)
            slopes=d["learned_carrier_phase_slope"].mean(0)
            target=d["carrier_target_phase_slope_evaluation_only"]
            for b,style in ((0,"-"),(1,"--")):
                axes[0,0].plot(d["wave_numbers"],slopes[:,b],c=COLORS[n],ls=style,lw=1.3,label=SHORT[n]+(" +" if b==0 else " −"))
                axes[0,0].plot(d["wave_numbers"],target[:,b],c="black",ls=":",lw=.9,
                    label="Reference ±E (evaluation only)" if b==0 and n==selected[0] else None)
            vals=part.carrier_phase_slope_mean_relative_error.to_numpy()
            x=selected.index(n)
            axes[0,1].scatter(x+np.linspace(-.1,.1,len(vals)),vals,c=COLORS[n],s=19)
        for _,run in part.iterrows():
            h=pd.read_csv(RAW/run.run_path/"loss_history.csv")
            h=h[h.phase=="coupled"].copy()
            h["model"],h["seed"]=n,int(run.seed)
            gradrows.append(h[["model","seed","step","window","output_r_gradient_norm","output_b_gradient_norm"]])
            axes[1,0].plot(h.step,h.output_b_gradient_norm,c=COLORS[n],lw=.6,alpha=.25)
            residual=pd.read_csv(RAW/run.run_path/"residual_by_segment_mode.csv")
            residual["model"],residual["seed"]=n,int(run.seed)
            residualrows.append(residual)
            by_mode=residual.groupby("mode_offset").unweighted_component_mean_power.mean()
            axes[1,1].plot(by_mode.index,by_mode.values,c=COLORS[n],alpha=.3,lw=.9)
    axes[0,0].set(title="Learned free-carrier slopes (seed 100)",xlabel="k",ylabel="Phase slope / time")
    axes[0,0].legend(fontsize=7,ncol=2)
    axes[0,1].set(title="Free-carrier slope error, all five seeds",xticks=range(len(selected)),xticklabels=[SHORT[n] for n in selected],ylabel="Mean relative error")
    if not selected:
        axes[0,0].text(.5,.5,"No carrier model selected",ha="center",transform=axes[0,0].transAxes)
        axes[0,1].text(.5,.5,"Not applicable",ha="center",transform=axes[0,1].transAxes)
    axes[0,1].set_ylim(bottom=0)
    axes[0,1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    for ax in (axes[1,0],axes[1,1]):
        ax.set_yscale("symlog",linthresh=1e-8)
    axes[1,0].set(title="Coupled phase / imaginary-output gradient",xlabel="Total optimization step",ylabel="Gradient norm")
    axes[1,1].set(title="Unweighted mode residuals, all seeds",xlabel="Mode offset j",ylabel="Component-mean power / L")
    fig.legend(handles=[Line2D([],[],c=COLORS[n],label=LABELS[n]) for n in validation.model.unique()],loc="outside lower center",ncol=4,fontsize=8)
    data={"all_seed_metrics":validation,"gradients":pd.concat(gradrows),"mode_residuals":pd.concat(residualrows)}
    if sources:
        data["carrier_slopes"]=pd.concat(sources)
    return finish(fig,"06_carrier_gradient",data,"Frozen learned carriers retain time derivatives; carrier error is evaluated independently and included in full residuals.","all five validation seeds; slope curves fixed seed100")


def residual_bounds(validation):
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout="constrained")
    rows=[]
    for n in validation.model.unique():
        for _,run in validation[validation.model==n].iterrows():
            r=pd.read_csv(RAW/run.run_path/"residual_integral.csv")
            r["model"],r["seed"]=n,int(run.seed)
            rows.append(r)
            axes[0].plot(r.t,r.raw_spinor_error_norm,c=COLORS[n],lw=.8,alpha=.25)
            axes[1].plot(r.t,r.error_bound,c=COLORS[n],lw=.8,alpha=.25)
            axes[2].scatter(run.residual_integral_fine,run.raw_final_spinor_error_norm,c=COLORS[n],s=22)
    axes[0].set(title="Raw spinor error norm",xlabel="t",ylabel="L2 norm")
    axes[1].set(title="Unweighted residual integral estimate",xlabel="t",ylabel="Initial error + integral")
    limit=max(axes[2].get_xlim()[1],axes[2].get_ylim()[1])
    axes[2].plot([0,limit],[0,limit],c="black",ls=":",lw=1)
    axes[2].set(title="Final error vs numerical estimate",xlabel="Residual integral estimate",ylabel="Raw final error")
    fig.legend(handles=[Line2D([],[],c=COLORS[n],label=LABELS[n]) for n in validation.model.unique()],loc="outside lower center",ncol=4,fontsize=8)
    return finish(fig,"07_residual_bound",{"integral":pd.concat(rows),"all_seed_metrics":validation},"Hermitian-system raw error estimate uses unweighted residuals, not a continuous certificate or weighted training loss.","all five validation seeds")


def learning_cost(validation):
    fig,axes=plt.subplots(2,3,figsize=(11.1,6.7),layout="constrained")
    rows=[]
    context=pd.read_csv(RAW/"cost_runtime_context.csv")
    validation=validation.merge(context[["stage","case","model","seed","balanced","cohort_shared_cpu_core"]],
        on=["stage","case","model","seed","balanced"],validate="one_to_one")
    names=list(validation.model.unique())
    for n in names:
        group=validation[validation.model==n]
        for _,run in group.iterrows():
            h=pd.read_csv(RAW/run.run_path/"loss_history.csv")
            h["model"],h["seed"]=n,int(run.seed)
            rows.append(h)
            coupling=h[h.phase=="coupled"]
            free=h[h.phase=="carrier"]
            axes[0,0].plot(coupling.step,coupling.unweighted_pde,c=COLORS[n],alpha=.3,lw=.65)
            axes[0,1].plot(free.step,free.unweighted_pde,c=COLORS[n],alpha=.3,lw=.65)
            axes[0,2].scatter(run.training_seconds,run.spinor_rel_l2_gauge_aligned,c=COLORS[n],s=22,
                edgecolors="black" if run.cohort_shared_cpu_core else "none",linewidths=.8)
        i=names.index(n)
        for ax,key in ((axes[1,0],"peak_rss_mb"),(axes[1,1],"parameter_count"),(axes[1,2],"residual_training_spinor_points")):
            v=group[key].to_numpy()
            ax.scatter(i+np.linspace(-.1,.1,len(v)),v,c=COLORS[n],s=19)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=0),c="black",fmt="_",capsize=3)
            ax.set_xticks(range(len(names)),[SHORT[m] for m in names])
    axes[0,0].set(title="Coupled unweighted PDE histories",xlabel="Total step (pretraining included)",ylabel="PDE power / L")
    axes[0,0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axes[0,1].set(title="Separate unit free-carrier histories",xlabel="Pretraining step",ylabel="Unit-IVP PDE MSE")
    axes[0,2].set(title="Observed cost (black edge: cohort core conflict)",xlabel="Observed training time / s",ylabel="Spinor relative L2")
    for ax in axes[0]:
        ax.set_yscale("symlog",linthresh=1e-8)
    axes[1,0].set(title="Training peak process memory",ylabel="Peak RSS / MiB")
    axes[1,1].set(title="All stored inference parameters",ylabel="Parameters (including frozen)")
    axes[1,2].set(title="Training spinor-residual points",ylabel="Unit branch IVPs + coupled states")
    fig.legend(handles=[Line2D([],[],c=COLORS[n],label=LABELS[n]) for n in names],loc="outside lower center",ncol=4,fontsize=8)
    return finish(fig,"08_learning_cost",{"history":pd.concat(rows),"costs":validation},"Equal steps do not imply equal cost or capacity. Some mixed-seed cohorts share CPU cores: time is observed wall cost, not evidence of algorithmic speedup.","all five validation seeds")


def conditional_comparison():
    ablation,rescue=read_stage("ablation"),read_stage("rescue")
    pairs=pd.read_csv(RAW/"paired_comparisons.csv")
    table=pairs[pairs.comparison.str.startswith("ablation/" if not ablation.empty else "exploratory_rescue/")]
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),layout="constrained")
    names=list(table.comparison.unique())
    for ax,metric,label in zip(axes,("spinor_rel_l2_gauge_aligned","current_rel_l2","sideband_complex_coefficient_rel_l2"),("Spinor paired difference","Current paired difference","Sideband coefficient paired difference")):
        for i,n in enumerate(names):
            v=table[table.comparison==n][metric+"_difference"].to_numpy()
            ax.scatter(i+np.linspace(-.1,.1,len(v)),v,c="#326E97",s=22)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=0),fmt="_",c="black",capsize=3)
        ax.axhline(0,c="black",lw=.7)
        labels=[n.split("/",1)[1].replace("_minus_","\n−\n").replace("_"," ") for n in names]
        ax.set(title=label,xticks=range(len(names)),xticklabels=labels,ylabel="Target minus single-factor ablation")
        ax.tick_params(axis="x",labelsize=6)
    scope="five paired validation seeds, conditional mechanism ablations" if not ablation.empty else "three exploratory seeds, one residual-balancing rescue (not independent validation)"
    fig.suptitle(scope,fontsize=9)
    return finish(fig,"09_conditional_comparison",{"paired":table,"ablation_runs":ablation if not ablation.empty else rescue},"Predeclared single-factor comparison; negative differences favor target. Rescue remains exploratory.",scope)


def extensions():
    frame=read_stage("extension")
    if frame.empty:
        return None
    fig,axes=plt.subplots(2,2,figsize=(10.7,6.6),layout="constrained")
    for ax,case in zip(axes.flat,("T8","V01","V04","free_packet")):
        group=frame[frame.case==case]
        names=list(group.model.unique())
        extents=[]
        for i,n in enumerate(names):
            rows=group[group.model==n]
            for key,marker,offset in (("spinor_rel_l2_gauge_aligned","o",-.08),("current_rel_l2","^",.08)):
                v=rows[key].to_numpy()
                extents.extend([*v,v.mean()-v.std(ddof=0),v.mean()+v.std(ddof=0)])
                ax.scatter(i+offset+np.linspace(-.06,.06,len(v)),v,marker=marker,c=COLORS[n],s=21)
                ax.errorbar(i+offset,v.mean(),yerr=v.std(ddof=0),c="black",fmt="_",capsize=2)
        ax.set(title=case+" | n=3 new seeds",xticks=range(len(names)),xticklabels=[SHORT[n] for n in names],ylabel="Spinor (circle) / current (triangle) L2")
        if min(extents)>0:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog",linthresh=1e-4)
        ax.axhline(.1,c="#326E97",ls=":",lw=.8)
        ax.axhline(.15,c="#C56965",ls=":",lw=.8)
    return finish(fig,"10_extensions",{"all_runs":frame},"Independent new-case three-seed scope diagnostics; no general-potential claim.","all three extension seeds 200–202 per case")


def free_packet_regression():
    if not (RAW/"free_packet_regression_series.csv").exists():
        return None
    series=pd.read_csv(RAW/"free_packet_regression_series.csv")
    metrics=pd.read_csv(RAW/"free_packet_regression_metrics.csv")
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.5),layout="constrained")
    first=series[(series.model==series.model.iloc[0])&(series.seed==200)]
    for row,quantity in enumerate(("center","width")):
        for col,branch in enumerate(("plus","minus")):
            key=f"branch_{branch}_{quantity}"
            axes[row,col].plot(first.t,first[key+"_exact"],c="black",lw=1.1,label="reference")
            for name in series.model.unique():
                for seed in (200,201,202):
                    values=series[(series.model==name)&(series.seed==seed)]
                    axes[row,col].plot(values.t,values[key],c=COLORS[name],alpha=.5,lw=.9)
            axes[row,col].set(title=f"Free Gaussian {branch} branch {quantity}",xlabel="t",ylabel="Circular "+quantity)
    fig.legend(handles=[Line2D([],[],c="black",label="reference")]+[
        Line2D([],[],c=COLORS[name],label=LABELS[name]) for name in series.model.unique()],
        loc="outside lower center",ncol=3,fontsize=8)
    return finish(fig,"11_free_packet_regression",{"series":series,"metrics":metrics},
        "R5 circular center/width definitions, reconstructed from projected total fields. Low-norm controls are failed packets, not reliable propagation centers.",
        "all three new free-regression seeds 200–202")


def window_conditioning():
    table=pd.read_csv(RAW/"window_coordinate_conditioning.csv")
    summary=pd.read_csv(RAW/"window_coordinate_summary.csv")
    valid=table[(table.mode_offset.abs()==1)&table.reference_active&table.initial_nonzero]
    fig,axes=plt.subplots(2,2,figsize=(10.7,6.7),layout="constrained")
    for window,values in valid.groupby("window"):
        x=window+np.linspace(-.16,.16,len(values))
        axes[0,0].scatter(x,values.predicted_initial_population,c=COLORS["C_release_windows"],s=13,alpha=.5)
        reference=values[values.seed==100]
        axes[0,0].scatter(np.full(len(reference),window),reference.reference_initial_population,c="black",s=17,marker="_")
        axes[0,1].scatter(x,values.local_velocity_coordinate_condition,c=COLORS["C_release_windows"],s=13,alpha=.5)
        axes[1,1].scatter(x,values.local_initial_derivative_abs_error,c=COLORS["C_release_windows"],s=13,alpha=.5)
    axes[1,0].scatter(valid.required_q_output_by_pde.abs(),valid.actual_q_output.abs(),
        c=COLORS["C_release_windows"],s=16,alpha=.5)
    q_values=np.r_[valid.required_q_output_by_pde.abs(),valid.actual_q_output.abs()]
    lower,limit=float(q_values.min()),float(q_values.max())
    axes[1,0].plot([lower,limit],[lower,limit],c="black",ls=":",lw=.8)
    axes[0,0].set(title="Reference-active first-sideband handoffs",xlabel="Window",ylabel="Initial branch population")
    axes[0,1].set(title="Local velocity-coordinate conditioning",xlabel="Window",ylabel="max(tangent scale) / min(scale)")
    axes[1,0].set(title="Required vs learned local output coordinates",xlabel="PDE-required |q| (not labels)",ylabel="Learned |q| at handoff")
    axes[1,1].set(title="Actual initial derivative error",xlabel="Window",ylabel="Absolute branch derivative error")
    for ax in (axes[0,0],axes[0,1]):
        ax.set_yscale("log")
    if lower > 0:
        axes[1,0].set_xscale("log")
        axes[1,0].set_yscale("log")
    else:
        axes[1,0].set_xscale("symlog",linthresh=1e-10)
        axes[1,0].set_yscale("symlog",linthresh=1e-10)
    axes[1,1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    fig.suptitle("C: all five seeds | output-coordinate diagnostic, not physical phase errors or training supervision",fontsize=9)
    return finish(fig,"12_window_conditioning",{"all_channels":table,"per_seed_summary":summary},
        "Only reference-active, predicted-nonzero first-sideband handoffs are shown. Small polar amplitudes suppress the phase tangent; local-output conditioning is not the full network Hessian or proof of the failure cause.",
        "all five independent validation seeds 100–104")


def plot_all():
    selection=json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
    selected=selection["selected_candidates"]
    exploration,validation=read_stage("exploration"),read_stage("validation")
    if len(exploration)!=18 or len(validation)!=20:
        raise RuntimeError("Plotting requires complete exploration and independent validation")
    names=selected+list(CONTROLS)
    figures=[primary(exploration,list(CANDIDATES+CONTROLS),"01_exploration","exploration, n=3 paired seeds"),
             primary(validation,names,"02_validation","independent validation, n=5 new paired seeds"),
             modal_dynamics(selected,validation),branch_coherence(selected,validation),fields(selected),
             carrier_gradient(validation),residual_bounds(validation),learning_cost(validation),conditional_comparison()]
    ext=extensions()
    if ext:
        figures.append(ext)
    packet=free_packet_regression()
    if packet:
        figures.append(packet)
    figures.append(window_conditioning())
    write_json(FIGURES/"figure_manifest.json",dict(backend="Python/matplotlib",figures=figures,
        fixed_main_dynamics_seed=100, all_seed_statistics=True, no_image_manipulation=True))
    print(f"Exported {len(figures)} figures, each PNG/SVG/PDF and source CSV",flush=True)


if __name__ == "__main__":
    plot_all()
