from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .physics import ROOT,potential,save_json

COLORS=["#39749A","#C78242","#518C79"]
RESULTS=ROOT/"results"
OUT=RESULTS/"figures"


def style():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7,
        "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
        "pdf.fonttype":42,"axes.linewidth":0.7,"lines.linewidth":1.2,"savefig.facecolor":"white","legend.frameon":False})


def save(fig,name,sources,caption,manifest):
    fig.set_constrained_layout(True)
    fig.canvas.draw()
    for ext in ["png","svg","pdf"]:
        fig.savefig(OUT/f"{name}.{ext}",dpi=240)
    manifest.append(dict(name=name,sources=sources,caption=caption,
                         inches=list(fig.get_size_inches()),formats=["png","svg","pdf"]))
    plt.close(fig)


def data(kind,seed):
    return np.load(RESULTS/"formal"/kind/f"seed_{seed}"/"evaluation.npz")


def panel(ax,label,title):
    ax.set_title(f"{label}  {title}",loc="left",pad=8)
    ax.set_xlabel("x")


def main():
    OUT.mkdir(parents=True,exist_ok=True);style();manifest=[]
    summary=pd.read_csv(RESULTS/"formal_summary.csv")
    ref=json.loads((RESULTS/"reference"/"summary.json").read_text())
    if len(summary)!=6: raise RuntimeError("Figures require all six formal runs.")
    fig,axs=plt.subplots(1,2,figsize=(7.35,2.8))
    x=np.linspace(-20,20,1001)
    axs[0].plot(x,potential(x),color="#495057",label="V(x)")
    axs[0].plot(x,1+potential(x),color="#969DA4",ls="--",label="m + V(x)")
    for e,label,clr in [(ref["dirac"]["energy"],"Dirac E",COLORS[0]),(ref["kg"]["energy"],"KG E",COLORS[1])]:
        axs[0].axhline(e,color=clr,ls=":" if "KG" in label else "-",label=label)
    for e in [-1,1]: axs[0].axhline(e,color="0.6",lw=.6,ls="--")
    panel(axs[0],"a","Vector well and rest-energy thresholds");axs[0].set_ylabel("Energy (m = 1)")
    axs[0].legend(ncol=2,loc="lower right");axs[0].set_ylim(-1.1,1.15)
    for j,kind in enumerate(["dirac","kg"]):
        axs[1].hlines(ref[kind]["energy"],j-.3,j+.3,color="0.25",label="Reference" if j==0 else None)
        for seed in [0,1,2]:
            e=summary.query("kind == @kind and seed == @seed").energy.iloc[0]
            axs[1].scatter(j+(seed-1)*.13,e,s=22,color=COLORS[seed],label=f"Seed {seed}" if j==0 else None,zorder=3)
    axs[1].set_xticks([0,1],["Dirac","KG"]);axs[1].set_ylabel("Lowest positive energy")
    axs[1].set_title("b  Distinct ground levels",loc="left");axs[1].legend(loc="center right")
    save(fig,"01_potential_energy",["formal_summary.csv","reference/summary.json"],
         "Gaussian time-component vector potential and target energies. Black horizontal marks: reference; colored dots: all three independent initializations. Rest-energy thresholds are ±1.",manifest)

    fig,axs=plt.subplots(2,2,figsize=(7.35,4.6))
    d0=data("dirac",0);x=d0["x"]
    for j,label in enumerate([r"u = Re($\psi_1$)",r"v = Im($\psi_2$)"]):
        axs[0,j].plot(x,d0["reference"][:,j],color="0.25",lw=2,label="Reference")
        for s,c in enumerate(COLORS):
            d=data("dirac",s)
            axs[0,j].plot(x,d["y"][:,j],ls="--",color=c,label=f"PINN {s}")
            axs[1,j].plot(x,d["point_error"][:,j],color=c,label=f"Seed {s}")
        panel(axs[0,j],"ab"[j],label);axs[0,j].set_xlim(-9,9);axs[0,j].set_ylabel("Amplitude")
        panel(axs[1,j],"cd"[j],"Signed point error");axs[1,j].set_xlim(-12,12);axs[1,j].set_ylabel("PINN - reference")
        axs[1,j].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    axs[0,0].legend(ncol=1,loc="upper left")
    save(fig,"02_dirac_components",["formal/dirac/seed_*/evaluation.csv"],
         "Dirac spinor is (u, i v). Each run is aligned by one shared constant sign. No post-training renormalization. Central region shown; full-domain tails appear in Figure 4.",manifest)

    fig,axs=plt.subplots(2,2,figsize=(7.35,4.6));d0=data("kg",0);x=d0["x"]
    for j,(key,refkey,title) in enumerate([("y","reference",r"KG wavefunction $\phi$"),("charge_density","reference_charge_density",r"Charge density $2(E-V)|\phi|^2$")]):
        yr=d0[refkey].reshape(-1)
        axs[0,j].plot(x,yr,color="0.25",lw=2,label="Reference")
        for s,c in enumerate(COLORS):
            d=data("kg",s);y=d[key].reshape(-1)
            axs[0,j].plot(x,y,color=c,ls="--",label=f"PINN {s}")
            axs[1,j].plot(x,y-yr,color=c,label=f"Seed {s}")
        panel(axs[0,j],"ab"[j],title);axs[0,j].set_xlim(-9,9);axs[0,j].set_ylabel("Amplitude" if j==0 else "Charge / length")
        panel(axs[1,j],"cd"[j],"Signed point error");axs[1,j].set_xlim(-12,12);axs[1,j].set_ylabel("PINN - reference")
        axs[1,j].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    axs[0,0].legend(ncol=1,loc="upper left")
    save(fig,"03_kg_wavefunction_charge",["formal/kg/seed_*/evaluation.csv"],
         "KG L2 normalization fixes amplitude only. Charge density uses the explicit 2(E−V)|phi|² convention and is positive for these target states.",manifest)

    fig,axs=plt.subplots(2,2,figsize=(7.35,4.6))
    for row,kind in enumerate(["dirac","kg"]):
        d0=data(kind,0);x=d0["x"];half=x>=0
        for s,c in enumerate(COLORS):
            d=data(kind,s)
            axs[row,0].semilogy(x,np.maximum(np.linalg.norm(d["residual"],axis=1),1e-16),color=c,label=f"Seed {s}")
            axs[row,1].semilogy(x[half],np.linalg.norm(d["y"],axis=1)[half],color=c,ls="--",label=f"PINN {s}")
        axs[row,1].semilogy(x[half],np.linalg.norm(d0["reference"],axis=1)[half],color="0.25",lw=1.7,label="Reference")
        panel(axs[row,0],"ac"[row],f"{kind.upper()} continuous residual")
        panel(axs[row,1],"bd"[row],f"{kind.upper()} tail decay")
        axs[row,0].set_ylabel("Pointwise residual norm");axs[row,1].set_ylabel("Wavefunction amplitude norm")
        axs[row,1].set_xlim(0,20)
    axs[0,0].legend(ncol=3);axs[0,1].legend(ncol=2)
    save(fig,"04_residuals_tails",["formal/*/seed_*/evaluation.csv"],
         "Residual derivatives use automatic differentiation on 8001 independent uniform test points. Tails are shown over the full positive half-domain; negative tails follow the exact imposed parity.",manifest)

    fig,axs=plt.subplots(2,2,figsize=(7.35,4.6))
    for j,kind in enumerate(["dirac","kg"]):
        for s,c in enumerate(COLORS):
            h=pd.read_csv(RESULTS/"formal"/kind/f"seed_{s}"/"history.csv")
            axs[0,j].semilogy(h.evaluations,h.loss,color=c,label=f"Seed {s}")
            axs[1,j].semilogy(h.evaluations,np.maximum(abs(h.energy-ref[kind]["energy"]),1e-16),color=c)
        for a in axs[:,j]:
            a.axvline(2001,color="0.5",ls=":",lw=.8);a.set_xlabel("Loss / closure evaluations")
        axs[0,j].set_title(f'{"ab"[j]}  {kind.upper()} physics-only loss',loc="left")
        axs[1,j].set_title(f'{"cd"[j]}  Energy error (post hoc)',loc="left")
        axs[0,j].set_ylabel("Unscaled total loss");axs[1,j].set_ylabel("Absolute energy error")
    axs[0,0].legend(ncol=3)
    save(fig,"05_training",["formal/*/seed_*/history.csv","reference/summary.json"],
         "Adam followed by L-BFGS; dotted line is the optimizer switch. L-BFGS closure evaluations can include trial steps. Reference energy appears only in post hoc plots and was never a regression target.",manifest)

    conv=pd.read_csv(RESULTS/"reference"/"convergence.csv")
    fig,axs=plt.subplots(2,2,figsize=(7.35,4.6))
    for i,kind in enumerate(["dirac","kg"]):
        c=COLORS[i]
        grid=conv.query("kind == @kind and method == 'fourier' and L == 20").sort_values("N")
        axs[0,0].loglog(grid.N,np.maximum(grid.abs_delta,1e-16),"o-",color=c,label=kind.upper())
        domain=conv.query("kind == @kind and method == 'fourier' and N == 257").sort_values("L")
        axs[0,1].semilogy(domain.L,np.maximum(domain.abs_delta,1e-16),"o-",color=c)
        axs[1,0].semilogy(domain.L,np.maximum(domain.wavefunction_l2,1e-16),"o-",color=c)
    doubled=pd.read_csv(RESULTS/"reference"/"naive_dirac_doubling.csv")
    for j,(n,group) in enumerate(doubled.groupby("N")):
        offset=np.resize(np.array([-.07,.07]),len(group))
        axs[1,1].scatter(j+offset,group.energy,s=25,marker="x",color=COLORS[j],label=f"N={n}")
    axs[1,1].set_xticks([0,1,2],["128","256","512"])
    for e in ref["dirac"]["positive_spectrum"]:
        axs[1,1].axhline(e,color="0.6",ls=":",lw=.7)
    axs[0,0].set(xlabel="Fourier grid N at L=20",ylabel="Absolute energy difference")
    axs[0,1].set(xlabel="Half-domain L (N=257)",ylabel="Absolute energy difference")
    axs[1,0].set(xlabel="Half-domain L (N=257)",ylabel="Relative wavefunction L2")
    axs[1,1].set(xlabel="Central-difference grid N",ylabel="Candidate Dirac energies")
    for a,title in zip(axs.flat,["a  Spatial resolution","b  Domain extension","c  Periodic-image effect","d  Spurious doublets (offset for visibility)"]):a.set_title(title,loc="left")
    axs[0,0].legend()
    save(fig,"06_reference_convergence",["reference/convergence.csv","reference/naive_dirac_doubling.csv"],
         "Differences are measured against the L=28 high-tolerance BVP target. At fixed L the Fourier error can reach a periodic-image plateau. Central-difference pairs have a small horizontal display offset within each grid-size category, without changing energy. They have large continuous residuals and are not accepted as physical states.",manifest)
    save_json(OUT/"manifest.json",manifest)
    print(f"Saved {len(manifest)} figure groups in PNG, SVG, PDF.")


if __name__=="__main__":main()
