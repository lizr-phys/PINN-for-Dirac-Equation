from __future__ import annotations
import json
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from .r11_config import RAW, FIGURES, GROUPS, SEEDS, THRESHOLDS, run_directory
from .freeze_r7_phase_release import write_json, sha256

COLORS = dict(zip(GROUPS,("#376F91","#B17C48","#776B9B","#737B80","#428A78")))
STYLES = ("-","--",":","-.",(0,(6,2)))
MARKERS = ("o","s","^","D","v")
SHORT = ("A","B","C","D","E*")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
    "axes.labelsize":8,"legend.fontsize":7,"svg.fonttype":"none","pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})

def save(fig,name,sources,claim,*,seeds=SEEDS):
    FIGURES.mkdir(parents=True,exist_ok=True)
    (FIGURES/"source_data").mkdir(exist_ok=True)
    paths=[]
    for suffix,table in sources.items():
        path=FIGURES/"source_data"/f"{name}_{suffix}.csv"
        table.to_csv(path,index=False); paths.append(path.relative_to(FIGURES).as_posix())
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); width,height=fig.canvas.get_width_height()
    undrawn=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            low,high=sorted(axis.get_view_interval())
            tol=16*np.finfo(float).eps*max(abs(low),abs(high),abs(high-low),np.finfo(float).tiny)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<low-tol or tick.get_loc()>high+tol:
                    undrawn.update((id(tick.label1),id(tick.label2)))
    outside=[]
    for item in fig.findobj(Text):
        if id(item) not in undrawn and item.get_visible() and item.get_text().strip():
            b=item.get_window_extent(renderer)
            if b.x0 < -2 or b.y0 < -2 or b.x1>width+2 or b.y1>height+2:
                outside.append(item.get_text())
    exports={}
    for suffix in ("png","svg","pdf"):
        path=FIGURES/f"{name}.{suffix}"
        fig.savefig(path,dpi=300)
        exports[suffix]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    plt.close(fig)
    return dict(figure=name,claim=claim,seeds=list(seeds),source_csv=paths,exports=exports,
        bounding_box_pass=not outside,text_outside_canvas=outside,
        svg_editable_text="<text" in (FIGURES/f"{name}.svg").read_text(encoding="utf-8"),
        statistics="n=5 fresh paired seeds; population SD ddof=0" if len(seeds)==5 else "fixed seed400",
        analytic_diagnostic="E*: exact free-carrier prior, excluded from learned claims",
        backend="Python/matplotlib")

def letters(axes):
    for letter,ax in zip(string.ascii_lowercase,np.asarray(axes).ravel()):
        ax.text(-.045,1.035,letter,transform=ax.transAxes,weight="bold",fontsize=10)

def seed_legend(ax,*,lines=False,ncol=5,loc="best",bbox_to_anchor=None):
    handles=[Line2D([0],[0],color="black",ls=STYLES[j] if lines else "None",
        marker=MARKERS[j],markersize=4,lw=.8,label=str(seed)) for j,seed in enumerate(SEEDS)]
    return ax.legend(handles=handles,title="Seeds",ncol=ncol,fontsize=6,
        loc=loc,bbox_to_anchor=bbox_to_anchor)

def source_series(filename):
    records=[]
    for group in GROUPS:
        for seed in SEEDS:
            p=run_directory(group,seed)/filename
            if p.is_file():
                f=pd.read_csv(p,float_precision="round_trip")
                f["model"],f["seed"]=group,seed; records.append(f)
    return pd.concat(records,ignore_index=True)

def dots(ax,frame,key):
    for i,g in enumerate(GROUPS):
        part=frame[frame.model==g].set_index("seed")
        values=part.loc[list(SEEDS),key]
        if values.notna().any():
            for j,s in enumerate(SEEDS):
                if pd.notna(values[s]):
                    ax.scatter(i+(j-2)*.055,values[s],color=COLORS[g],marker=MARKERS[j],s=22,zorder=3)
            ax.errorbar(i+.20,values.mean(),yerr=values.std(ddof=0),fmt="_",color="black",capsize=3)
    ax.set(xticks=range(5),xticklabels=SHORT,xlim=(-.35,4.45))
    ax.grid(axis="y",alpha=.15)

def make_figures():
    frame=pd.read_csv(RAW/"per_seed_metrics.csv",float_precision="round_trip")
    final=frame[frame.endpoint=="final"]
    qa=[]
    for endpoint,label in (("final","Fixed final endpoint: A-D total 26000; E* coupled 20000"),
                           ("coupled_20000","Equal 20000 coupled steps; A includes 6000 auxiliary steps")):
        f=frame[frame.endpoint==endpoint]
        fig,axes=plt.subplots(2,3,figsize=(10.5,5.7),layout="constrained")
        titles=("Spinor relative L2","Current relative L2","Mean fidelity","Modal population relative L2","Maximum norm drift")
        for ax,key,title in zip(axes.ravel(),THRESHOLDS,titles):
            dots(ax,f,key); ax.axhline(THRESHOLDS[key],ls="--",color="black",lw=.8)
            ax.set_title(title)
            if key!="fidelity_mean": ax.set_yscale("log")
        axes[1,2].axis("off")
        for j,s in enumerate(SEEDS):
            axes[1,2].scatter(.05,.90-j*.08,marker=MARKERS[j],color="black",s=22)
            axes[1,2].text(.12,.90-j*.08,str(s),va="center")
        axes[1,2].text(.05,.40,"All five seeds retained\nBlack: mean +/- population SD\nDashed: preregistered criterion\nE*: analytic diagnostic only\nA pretrained; B random frozen\nC joint; D capacity control",va="top",fontsize=8)
        axes[1,2].set(xlim=(0,1),ylim=(0,1))
        fig.suptitle(label,fontsize=11); letters(axes.ravel()[:5])
        qa.append(save(fig,f"primary_metrics_{endpoint}",{"values":f[["endpoint","model","seed"]+list(THRESHOLDS)]},
            "Absolute replication and comparison are distinct; all five fresh seeds and fixed endpoints."))
    comparisons=pd.read_csv(RAW/"paired_comparisons.csv",float_precision="round_trip")
    fig,axes=plt.subplots(2,2,figsize=(10.5,6),layout="constrained")
    pairlist=((GROUPS[0],GROUPS[1]),(GROUPS[0],GROUPS[2]),(GROUPS[0],GROUPS[3]),(GROUPS[4],GROUPS[0]),(GROUPS[2],GROUPS[1]))
    labels=("A-B","A-C","A-D","E*-A","C-B")
    for row,endpoint in enumerate(("final","coupled_20000")):
        for col,key in enumerate(("spinor_rel_l2_gauge_aligned","current_rel_l2")):
            ax=axes[row,col]
            for i,(target,base) in enumerate(pairlist):
                part=comparisons[(comparisons.endpoint==endpoint)&(comparisons.target==target)&
                    (comparisons.baseline==base)&(comparisons.metric==key)].set_index("seed")
                v=part.loc[list(SEEDS),"paired_difference"]
                for j,s in enumerate(SEEDS):
                    ax.scatter(i+(j-2)*.055,v[s],color=COLORS[target],marker=MARKERS[j],s=20)
                ax.errorbar(i+.20,v.mean(),yerr=v.std(ddof=0),fmt="_",color="black",capsize=3)
            ax.axhline(0,color="gray",lw=.8)
            ax.set(xticks=range(5),xticklabels=labels,ylabel="Target - baseline",
                title=f"{endpoint}: "+("spinor" if col==0 else "current"))
    fig.suptitle("Actual paired differences; no post hoc superiority threshold",fontsize=11)
    seed_legend(axes[0,1])
    letters(axes); qa.append(save(fig,"paired_ablation_differences",{"pairs":comparisons},
        "Paired contrasts separate budgets; A-C does not isolate freezing alone."))
    curves=pd.read_csv(RAW/"error_cost_curves.csv",float_precision="round_trip")
    fig,axes=plt.subplots(2,2,figsize=(10.5,6),layout="constrained")
    for row,key in enumerate(("spinor_rel_l2_gauge_aligned","current_rel_l2")):
        for col,cost in enumerate(("training_spinor_residual_points","training_seconds")):
            ax=axes[row,col]
            for g in GROUPS:
                for j,s in enumerate(SEEDS):
                    p=curves[(curves.model==g)&(curves.seed==s)].sort_values("coupled_steps")
                    ax.plot(p[cost]/(1e6 if col==0 else 60),p[key],color=COLORS[g],ls=STYLES[j],
                        marker=MARKERS[j],ms=3,lw=.8,alpha=.8,label=SHORT[GROUPS.index(g)] if j==0 else None)
            ax.set(yscale="log",ylabel="Spinor relative L2" if row==0 else "Current relative L2",
                xlabel="Cumulative spinor residual points (million)" if col==0 else "Optimization wall time (min)")
            ax.grid(alpha=.15)
    axes[0,0].legend(ncol=5,title="E*: exact diagnostic",loc="upper right",bbox_to_anchor=(1,.86))
    seed_legend(axes[1,1],lines=True,loc="upper right",bbox_to_anchor=(1,.80))
    fig.suptitle("All 115 fixed cost points; A cost includes auxiliary pretraining",fontsize=11)
    letters(axes); qa.append(save(fig,"error_vs_total_cost",{"curves":curves},
        "Equal optimizer steps do not mean equal residual work or measured wall time."))
    series=source_series("time_series.csv")
    fig,axes=plt.subplots(4,5,figsize=(14,9),layout="constrained")
    for i,g in enumerate(GROUPS):
        for row,(b,measure) in enumerate((("plus","center"),("minus","center"),("plus","width"),("minus","width"))):
            ax=axes[row,i]
            for j,s in enumerate(SEEDS):
                p=series[(series.model==g)&(series.seed==s)]
                ax.plot(p.t,p[f"{b}_{measure}_predicted"],color=COLORS[g],ls=STYLES[j],lw=.8,label=str(s))
            ax.plot(p.t,p[f"{b}_{measure}_reference"],color="black",lw=1.1,label="Reference")
            if row==0: ax.set_title(SHORT[i])
            if i==0: ax.set_ylabel(f"{b} {measure}")
            if row==3: ax.set_xlabel("t")
            ax.grid(alpha=.12)
    axes[0,0].legend(ncol=2,fontsize=6)
    fig.suptitle("Both branch trajectories, all seeds; low-norm failed predictions are diagnostic",fontsize=11)
    letters(axes)
    qa.append(save(fig,"branch_trajectories_all_seeds",{"series":series},
        "Branch center and width are compared to the same coupled reference, without timewise phase alignment."))
    fig,axes=plt.subplots(2,5,figsize=(13,5.6),layout="constrained")
    balances=source_series("norm_balance.csv")
    for i,g in enumerate(GROUPS):
        for j,s in enumerate(SEEDS):
            p=balances[(balances.model==g)&(balances.seed==s)]
            axes[0,i].plot(p.t,p.norm_change_from_initial,color=COLORS[g],ls=STYLES[j],lw=.8,label=str(s))
            axes[0,i].plot(p.t,p.residual_projected_integral,color="black",ls=":",lw=.4,alpha=.35)
            axes[1,i].plot(p.t,p.numerical_integration_discrepancy,color=COLORS[g],ls=STYLES[j],lw=.8)
        axes[0,i].axhline(.02,ls="--",color="gray",lw=.7)
        axes[0,i].axhline(-.02,ls="--",color="gray",lw=.7)
        axes[0,i].set_title(SHORT[i]); axes[1,i].set_xlabel("t")
        axes[1,i].ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
    axes[0,0].set_ylabel("N(t) - N(0)"); axes[1,0].set_ylabel("Numerical integration discrepancy")
    axes[0,0].legend(ncol=3,fontsize=6,loc="upper right",bbox_to_anchor=(1,.90))
    fig.suptitle("Norm balance: raw prediction and integral of 2 Im <psi,R>; numerical, not a certificate",fontsize=10)
    letters(axes); qa.append(save(fig,"norm_balance_all_seeds",{"balance":balances},
        "Full residual projection closes norm numerically, including all failed trajectories."))
    keys=("branch_complex_coefficient_rel_l2","active_coherence_rel_l2","carrier_free_pde_residual_rms",
          "residual_integral_fine","exterior_residual_integral","norm_balance_integration_max_abs_discrepancy")
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout="constrained")
    for ax,key,title in zip(axes.ravel(),keys,("Complex branch coefficients","Reference-active coherence",
        "Unit carrier defect (D: N/A)","Retained residual integral","Exterior residual integral","Norm integration discrepancy")):
        dots(ax,final,key); ax.set(yscale="log",title=title)
    fig.suptitle("Independent diagnostics, all seeds; E* carrier defect is floating-point scale",fontsize=10)
    seed_legend(axes[0,1])
    letters(axes); qa.append(save(fig,"complex_carrier_residual_diagnostics",{"values":final[["model","seed"]+list(keys)]},
        "Exact-carrier E is a diagnostic prior; residual quadrature is not continuous certification."))
    fig,axes=plt.subplots(4,5,figsize=(14,9),layout="constrained")
    modal_rows=[]
    for i,g in enumerate(GROUPS):
        for j,s in enumerate(SEEDS):
            p=series[(series.model==g)&(series.seed==s)]
            with np.load(run_directory(g,s)/"dense_grid.npz") as dense:
                n=dense["n"]; predicted=dense["modal_occupations_predicted"][-1]
                exact=dense["modal_occupations_reference"][-1]
                branches=dense["branch_occupations_predicted"][-1]
                modal_rows.append(pd.DataFrame(dict(model=g,seed=s,n=n,total_predicted=predicted,total_reference=exact,
                    plus_predicted=branches[:,0],minus_predicted=branches[:,1])))
                axes[0,i].plot(n,predicted,color=COLORS[g],ls=STYLES[j],lw=.8)
            axes[1,i].plot(p.t,p.plus_norm_predicted,color=COLORS[g],ls=STYLES[j],lw=.8)
            axes[2,i].plot(p.t,p.minus_norm_predicted,color=COLORS[g],ls=STYLES[j],lw=.8)
            axes[3,i].plot(p.t,p.occupation_outside_initial_support_predicted,color=COLORS[g],ls=STYLES[j],lw=.8)
        axes[0,i].plot(n,exact,color="black",lw=1)
        axes[0,i].set(title=SHORT[i],yscale="log",xlabel="n at T=5")
        axes[1,i].plot(p.t,p.plus_norm_reference,color="black",lw=1)
        axes[2,i].plot(p.t,p.minus_norm_reference,color="black",lw=1)
        axes[3,i].plot(p.t,p.occupation_outside_initial_support_reference,color="black",lw=1)
        for row in range(1,4): axes[row,i].set_xlabel("t")
        for row in range(4): axes[row,i].grid(alpha=.1)
    for ax,label in zip(axes[:,0],("Total modal occupation","Plus branch occupation","Minus branch occupation",
            "Occupation outside n=0..32")):
        ax.set_ylabel(label)
    seed_legend(axes[0,0],lines=True,ncol=2)
    fig.suptitle("All seeds: modal and branch occupations; new support is outside the original n=0..32",fontsize=10)
    letters(axes); qa.append(save(fig,"modal_branch_occupations_all_seeds",{"series":series,
        "final_modal_profiles":pd.concat(modal_rows,ignore_index=True)},
        "Occupations include both branches and all retained modes; new support respects the original broad Gaussian."))
    fields=[]; density=[]; current=[]
    for g in GROUPS:
        with np.load(run_directory(g,400)/"dense_grid.npz") as d:
            t,x=d["t"],d["x"]
            if not density:
                density.append(d["density_reference"]); current.append(d["current_reference"])
            density.append(d["density_predicted"]); current.append(d["current_predicted"])
    fig,axes=plt.subplots(4,6,figsize=(15,8),layout="constrained")
    vmax=max(a.max() for a in density); jmax=max(abs(a).max() for a in current)
    derr=[abs(a-density[0]) for a in density]; jerr=[abs(a-current[0]) for a in current]
    for row,values,title,cmap,low,high in ((0,density,"Density","viridis",0,vmax),
        (1,current,"Current","RdBu_r",-jmax,jmax),(2,derr,"Density absolute error","magma",0,max(a.max() for a in derr)),
        (3,jerr,"Current absolute error","magma",0,max(a.max() for a in jerr))):
        for i,a in enumerate(values):
            im=axes[row,i].pcolormesh(x,t,a,shading="auto",cmap=cmap,vmin=low,vmax=high,rasterized=True)
            if row==0: axes[row,i].set_title((("Reference",)+SHORT)[i])
            if i==0: axes[row,i].set_ylabel(title+"\nt")
            if row==3: axes[row,i].set_xlabel("x")
        fig.colorbar(im,ax=list(axes[row]),fraction=.014,pad=.01)
    tt,xx=np.meshgrid(t,x,indexing="ij")
    for i,label in enumerate(("reference",)+GROUPS):
        fields.append(pd.DataFrame(dict(model=label,seed=400,t=tt.ravel(),x=xx.ravel(),density=density[i].ravel(),
            current=current[i].ravel(),density_absolute_error=derr[i].ravel(),current_absolute_error=jerr[i].ravel())))
    fig.suptitle("Fixed seed400 dynamics; common color limits within each row; E* exact diagnostic",fontsize=11)
    letters(axes)
    qa.append(save(fig,"dynamics_seed400",{"physical_grid":pd.concat(fields,ignore_index=True)},
        "All five groups and reference at a preregistered seed, with failures visible.",seeds=(400,)))
    write_json(FIGURES/"FIGURE_QA.json",dict(status="PENDING_VISUAL_INSPECTION",
        figures=qa,visual_inspection_pending=True,backend="Python/matplotlib",all_seeds=list(SEEDS)))
    return qa

if __name__=="__main__":
    print(json.dumps(make_figures(),indent=2))
