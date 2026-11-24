"""Source-grounded report figures; every seed is retained."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .physics import ROOT,config,potential,save_json

OUT=ROOT/"figures"
COLORS={"dirac":"#355D7B","kg":"#B77749"}
VARIANTS=["original","legacy","independent","derivative"]
LABELS=["Original\n3 seeds","Soft + GN\n3 seeds","Hard, 2 outputs\n3 Dirac seeds","Improved\n10 seeds"]
CAPTIONS={}
SOURCES={}

def setup():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":.7,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42})
    OUT.mkdir(exist_ok=True)

def save(fig,name,caption,sources):
    fig.canvas.draw()
    for suffix in ["png","svg","pdf"]:
        fig.savefig(OUT/f"{name}.{suffix}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[name]=caption
    canonical=[]
    for source in sources:
        path=(OUT/source).resolve() if source.startswith("../") or "/" not in source else (ROOT/source).resolve()
        canonical.append(str(path.relative_to(ROOT)).replace("\\","/"))
    SOURCES[name]=canonical

def panel(ax,label):
    ax.text(-.12,1.06,label,transform=ax.transAxes,fontweight="bold",fontsize=10)

def array(row):
    return np.load(ROOT/row["run"]/"evaluation.npz")

def comparison(formal,ablation):
    old=pd.read_csv(ROOT.parent/"results"/"formal_summary.csv")
    old["variant"]="original";old["case"]="base";old["state"]=0
    base=formal[(formal["case"]=="base")&(formal.state==0)]
    table=pd.concat([old,ablation,base],ignore_index=True)
    table.to_csv(OUT/"comparison_source.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout="constrained")
    for ax,key,title in zip(axes.flat,["energy_abs_error","wavefunction_rel_l2","normalization_error","seconds"],
                            ["Absolute energy difference","Wavefunction relative L2 error","Independent-grid normalization error","Training time (s)"]):
        for ki,kind in enumerate(["dirac","kg"]):
            for vi,variant in enumerate(VARIANTS):
                rows=table[(table.kind==kind)&(table.variant==variant)]
                if rows.empty:continue
                y=np.maximum(rows[key].to_numpy(),1e-16)
                xpos=vi+(ki-.5)*.24
                jitter=np.linspace(-.07,.07,len(y))
                ax.scatter(xpos+jitter,y,s=18,marker="o" if kind=="dirac" else "^",color=COLORS[kind],alpha=.8)
                ax.plot([xpos-.10,xpos+.10],[np.mean(y)]*2,color=COLORS[kind],lw=1.7)
        ax.set_xticks(range(4),LABELS,fontsize=7)
        ax.set_title(title,fontsize=9)
        ax.set_yscale("log")
    for ax,label in zip(axes.flat,"abcd"):panel(ax,label)
    axes[0,0].legend([Line2D([],[],marker="o",ls="",color=COLORS["dirac"]),Line2D([],[],marker="^",ls="",color=COLORS["kg"])],["Dirac","KG"],fontsize=7)
    save(fig,"01_comparison","所有种子散点；短线为算术均值，无显著性检验或置信区间。原始组为三种子且曾并发计时；改进基准为版本 6 十种子，版本 7 的新软约束及硬双输出对照各三种子、串行计时。架构、先验、数值保护和预算差异见方法，不能把原始时间作严格等硬件速度比或完全等协议因果试验。归一化在独立 8001 点网格测量，显示下限 1e-16；原始数据未截断。",["comparison_source.csv"])

def base_profiles(formal,kind):
    rows=formal[(formal["case"]=="base")&(formal.kind==kind)&(formal.state==0)]
    data=[array(r) for r in rows.to_dict("records")]
    first=data[0];x=first["x"]
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),layout="constrained")
    if kind=="dirac":
        for j in [0,1]:
            ax=axes[0,j]
            ax.plot(x,first["reference"][:,j],color="#222222",lw=1.7,label="Reference")
            for i,d in enumerate(data):ax.plot(x,d["y"][:,j],color=COLORS[kind],lw=.8,alpha=.5,label="PINN, 10 seeds" if i==0 else None)
            ax.set_xlim(-9,9);ax.set_ylabel("u = Re(psi1)" if j==0 else "v = Im(psi2)")
            ax=axes[1,j]
            for d in data:ax.semilogy(x,np.maximum(abs(d["y"][:,j]-d["reference"][:,j]),1e-18),color=COLORS[kind],lw=.7,alpha=.65)
            ax.set_ylabel("Absolute component error");ax.set_xlim(-20,20)
        name="02_dirac_components"
        caption="Dirac 上、下分量和逐点绝对误差，十个正式种子全部绘出。只对齐一个共同整体符号；未重归一化 PINN。上排放大核心区域，下排含完整区间；对数误差显示下限 1e-18。"
    else:
        quantities=[("y","reference","phi"),("charge_density","reference_charge_density","Charge density: 2(E-V)phi^2")]
        for j,(key,refkey,label) in enumerate(quantities):
            rf=first[refkey][:,0] if j==0 else first[refkey]
            axes[0,j].plot(x,rf,color="#222222",lw=1.7,label="Reference")
            for i,d in enumerate(data):
                y=d[key][:,0] if j==0 else d[key]
                axes[0,j].plot(x,y,color=COLORS[kind],lw=.8,alpha=.5,label="PINN, 10 seeds" if i==0 else None)
                r=d[refkey][:,0] if j==0 else d[refkey]
                axes[1,j].semilogy(x,np.maximum(abs(y-r),1e-18),color=COLORS[kind],lw=.7,alpha=.65)
            axes[0,j].set_ylabel(label);axes[0,j].set_xlim(-9,9)
            axes[1,j].set_ylabel("Absolute point error")
        name="03_kg_wave_charge"
        caption="KG 波形、所选约定的电荷密度及逐点误差；十个正式种子全部绘出。L2 归一化用于固定幅度，不把 phi^2 解释为一般概率密度。只对齐整体符号，对数误差显示下限 1e-18。"
    axes[0,0].legend(fontsize=7)
    for ax,label in zip(axes.flat,"abcd"):panel(ax,label);ax.set_xlabel("x (natural units)")
    save(fig,name,caption,[r["run"]+"/evaluation.npz" for r in rows.to_dict("records")])

def coverage(formal):
    cfg=config()
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout="constrained")
    names={"base":"Base","shallow":"Shallow","narrow":"Narrow","deep_wide":"Deep/wide"}
    palette=["#355D7B","#888D94","#8E80AB","#63988B"]
    x=np.linspace(-12,12,801)
    for color,(case_id,case) in zip(palette,cfg["cases"].items()):axes[0,0].plot(x,potential(x,case),color=color,label=names[case_id])
    axes[0,0].set(xlabel="x",ylabel="Vector potential V(x)");axes[0,0].legend(fontsize=7)
    for ki,kind in enumerate(["dirac","kg"]):
        for state in [0,1,2]:
            row=formal[(formal["case"]=="deep_wide")&(formal.kind==kind)&(formal.state==state)].iloc[0]
            e=row.reference_energy
            axes[0,1].plot([ki-.3,ki+.3],[e,e],color=COLORS[kind],lw=2)
            axes[0,1].text(ki+.32,e,f"n={state}",fontsize=7,va="center")
    axes[0,1].axhline(1,color=".5",ls="--",lw=.7)
    axes[0,1].set(xticks=[0,1],xticklabels=["Dirac","KG"],ylabel="Validated reference energy E",ylim=(.2,1.05))
    sectors=[("base",0),("shallow",0),("narrow",0),("deep_wide",0),("deep_wide",1),("deep_wide",2)]
    for ax,key,label in [(axes[1,0],"wavefunction_rel_l2","Wavefunction relative L2"),(axes[1,1],"residual_l2","Residual L2 / m (Dirac), / m^2 (KG)")]:
        for j,(case_id,state) in enumerate(sectors):
            for ki,kind in enumerate(["dirac","kg"]):
                rows=formal[(formal["case"]==case_id)&(formal.kind==kind)&(formal.state==state)]
                xpos=j+(ki-.5)*.22+np.linspace(-.06,.06,len(rows))
                good=rows.reliable.to_numpy()
                ax.scatter(xpos[good],rows.loc[good,key],color=COLORS[kind],s=18,marker="o" if kind=="dirac" else "^")
                ax.scatter(xpos[~good],rows.loc[~good,key],color="#B54444",s=32,marker="x")
        ax.set_yscale("log");ax.set_ylabel(label)
        ax.set_xticks(range(len(sectors)),[names[c]+f"\nn={s}" for c,s in sectors],fontsize=7)
    axes[1,0].legend([Line2D([],[],ls="",marker="o",color=COLORS["dirac"]),
                      Line2D([],[],ls="",marker="^",color=COLORS["kg"]),
                      Line2D([],[],ls="",marker="x",color="#B54444")],
                     ["Dirac","KG","Failed attempt"],fontsize=6,loc="upper left")
    for ax,label in zip(axes.flat,"abcd"):panel(ax,label)
    save(fig,"04_parameter_state_coverage","四个矢量势参数组；深宽阱三个已验证能级。误差/残差散点包含全部正式种子，失败尝试以红叉标示；基准最低态每模型样本数 10，其余扇区每模型样本数 3。保留的 38 次最低/第一激发态为版本 6，其余 12 次为版本 7，逐运行协议在源 CSV 中明列。Dirac/KG 残差量纲不同，不按其数值跨模型排名。四个参数分别训练，不是参数化算子或未训练参数泛化。",["../results/formal_summary.csv","../config.json"])

def deep_profiles(formal,kind):
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8),layout="constrained")
    sources=[]
    for state in [0,1,2]:
        rows=formal[(formal["case"]=="deep_wide")&(formal.kind==kind)&(formal.state==state)]
        data=[array(r) for r in rows.to_dict("records")]
        row_records=rows.to_dict("records")
        has_failure=not rows.reliable.all()
        for row in rows.to_dict("records"):sources.append(row["run"]+"/evaluation.npz")
        x=data[0]["x"]
        for j in [0,1]:
            key="y" if kind=="dirac" or j==0 else "charge_density"
            refkey="reference" if key=="y" else "reference_charge_density"
            reference=data[0][refkey][:,j if kind=="dirac" else 0] if key=="y" else data[0][refkey]
            ax=axes[j,state]
            ax.plot(x,reference,color=".15",lw=1.6,label="Reference")
            for i,d in enumerate(data):
                y=d[key][:,j if kind=="dirac" else 0] if key=="y" else d[key]
                bad=not row_records[i]["reliable"]
                ax.plot(x,y,color="#B54444" if bad else COLORS[kind],ls="--" if bad else "-",lw=.9,alpha=.7,
                        label=f"Failed seed {row_records[i]['seed']}" if bad else ("PINN" if i==0 else None))
            ax.set_xlim((x[0],x[-1]) if has_failure else (-12,12));ax.set_xlabel("x")
            if has_failure:ax.legend(fontsize=6)
            if j==0:ax.set_title(f"State n={state}")
            if state==0:ax.set_ylabel(["u","v"][j] if kind=="dirac" else ["phi","Charge density"][j])
    axes[0,0].legend(fontsize=7)
    for ax,label in zip(axes.flat,"abcdef"):panel(ax,label)
    save(fig,"05_deep_dirac" if kind=="dirac" else "06_deep_kg",f"深宽阱 {kind.upper()} 的 n=0,1,2。各态三个种子全部绘出；Dirac 逐分量、KG 波形与电荷分别绘图。失败种子用红色虚线单独标示，该态显示完整区间；全部成功的态放大核心区。原始文件包含完整区间，不能把失败尝试当作可靠本征态。",sources)

def diagnostics(formal):
    ortho=pd.read_csv(ROOT/"results"/"orthogonality.csv")
    cfg=config()
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout="constrained")
    q=ortho[(ortho.kind=="kg")&(ortho.state_n==0)&(ortho.state_m==2)]
    for j,metric in enumerate(["ordinary_l2","normalized_physical_overlap"]):
        for offset,source in [(-.10,"reference"),(.10,"pinn")]:
            sub=q[q.source==source]
            axes[0,0].scatter(j+offset+np.linspace(-.03,.03,len(sub)),np.maximum(abs(sub[metric]),1e-16),
                              s=22,color=".3" if source=="reference" else COLORS["kg"],label=source if j==0 else None)
    axes[0,0].set_yscale("log");axes[0,0].set_xticks([0,1],["Ordinary L2\n(not required zero)","Charge-weighted\nnormalized overlap"],fontsize=7)
    axes[0,0].set_ylabel("Absolute overlap, KG states 0 and 2");axes[0,0].legend(fontsize=7)
    rs=[]
    for name in cfg["cases"]:
        df=pd.read_csv(ROOT/"results"/"reference"/name/"convergence.csv")
        df["step"]=np.where(df.N==257,0,np.where(df.L==cfg["cases"][name]["reference_length"],1,2))
        rs.append(df)
    references=pd.concat(rs,ignore_index=True);references.to_csv(OUT/"reference_convergence_source.csv",index=False)
    for kind in ["dirac","kg"]:
        worst=references[references.kind==kind].groupby("step").wavefunction_l2.max()
        axes[0,1].semilogy(worst.index,np.maximum(worst,1e-16),"o-",color=COLORS[kind],label=kind)
    axes[0,1].set_xticks([0,1,2],["257, Lref","513, Lref","513, Lref+8"],fontsize=7)
    axes[0,1].set_ylabel("Worst reference wavefunction discrepancy");axes[0,1].legend(fontsize=7)
    for ki,kind in enumerate(["dirac","kg"]):
        values=[]
        values.append(float(formal[(formal["case"]=="base")&(formal.kind==kind)&(formal.seed==0)].iloc[0].wavefunction_rel_l2))
        for group in ["grid_refined","domain_extended"]:
            df=pd.read_csv(ROOT/"results"/f"{group}_summary.csv")
            values.append(float(df[df.kind==kind].iloc[0].wavefunction_rel_l2))
        axes[1,0].semilogy(range(3),values,"o-",color=COLORS[kind],label=kind)
        base=formal[(formal["case"]=="base")&(formal.kind==kind)]
        axes[1,1].semilogy(base.initial_energy,base.residual_l2,"o" if kind=="dirac" else "^",color=COLORS[kind],label=kind)
    axes[1,0].set_xticks([0,1,2],["Seed 0 baseline","Double quadrature","L = 24"],fontsize=7)
    axes[1,0].set_ylabel("Wavefunction relative L2");axes[1,0].legend(fontsize=7)
    axes[1,1].set(xlabel="Initial energy E/m",ylabel="Residual L2 / m (Dirac), / m^2 (KG)")
    axes[1,1].legend(fontsize=7)
    for ax,label in zip(axes.flat,"abcd"):panel(ax,label)
    save(fig,"07_validation_diagnostics","KG 同宇称 n=0 与 n=2 的非平凡重叠；参考样本一对，PINN 三种子，显示绝对值但有符号原始值保留。参考收敛为十二个目标态中的最大波形差。敏感性仅为每模型单种子，不能据此声称训练误差单调收敛。独立网格残差包含全部十个基准种子，已按自然质量单位区分量纲。",["../results/orthogonality.csv","reference_convergence_source.csv","../results/formal_summary.csv","../results/grid_refined_summary.csv","../results/domain_extended_summary.csv"])

def training_cost(formal):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout="constrained")
    for kind in ["dirac","kg"]:
        rows=formal[(formal["case"]=="base")&(formal.kind==kind)]
        for seed,style in zip([0,4,8],["-","--","-."]):
            row=rows[rows.seed==seed].iloc[0]
            history=pd.read_csv(ROOT/row.run/"history.csv")
            axes[0,0].semilogy(history.seconds,np.maximum(history.loss,1e-18),color=COLORS[kind],ls=style,lw=.9,alpha=.8)
            axes[0,1].semilogy(history.seconds,np.maximum(abs(history.energy-row.reference_energy),1e-16),color=COLORS[kind],ls=style,lw=.9,alpha=.8)
            if "attempt" in history:
                for transition in history.loc[history.attempt.diff()>0,"seconds"]:
                    for ax in axes[0]:ax.axvline(transition,color=COLORS[kind],ls=":",lw=.7,alpha=.6)
        axes[1,0].loglog(rows.seconds,rows.wavefunction_rel_l2,"o" if kind=="dirac" else "^",color=COLORS[kind],ms=4,label=kind+" PINN")
        counts=rows.stop_reason.value_counts()
        ki=0 if kind=="dirac" else 1
        for j,status in enumerate(["physics_tolerance","iteration_budget"]):
            axes[1,1].bar(j+(ki-.5)*.3,counts.get(status,0),width=.27,color=COLORS[kind],label=kind if j==0 else None)
    bvp=pd.read_csv(ROOT/"results"/"bvp_benchmark.csv")
    for kind in ["dirac","kg"]:
        rows=bvp[bvp.kind==kind]
        axes[1,0].loglog(rows.seconds,np.maximum(rows.wavefunction_rel_l2,1e-16),"x",color=COLORS[kind],ms=5,label=kind+" BVP")
    axes[0,0].set(xlabel="Training wall time (s)",ylabel="Physics objective")
    axes[0,1].set(xlabel="Training wall time (s)",ylabel="Energy difference, evaluated afterwards")
    axes[0,0].legend([Line2D([],[],color=COLORS[k]) for k in ["dirac","kg"]],["Dirac","KG"],fontsize=7)
    axes[0,1].legend([Line2D([],[],color=".25",ls=s) for s in ["-","--","-."]],["Seed 0","Seed 4","Seed 8"],fontsize=7)
    axes[1,0].set(xlabel="Complete solve/training time (s)",ylabel="Wavefunction relative L2");axes[1,0].legend(fontsize=7)
    axes[1,1].set_xticks([0,1],["Physics tolerance","Budget exhausted"],fontsize=7);axes[1,1].set_ylabel("Number of base runs");axes[1,1].legend(fontsize=7)
    for ax,label in zip(axes.flat,"abcd"):panel(ax,label)
    save(fig,"08_training_cost","训练曲线使用预先指定的种子 0/4/8，点线标示重启开始，累计时间包含失败尝试；能量差在训练后计算，没有进入损失或早停。成本图包含两模型各十种子、BVP 每模型三容差各五次计时；BVP 评价不计入求解时间，PINN 评价亦不计入训练时间。预算结束不是辨态失败；更严格物理早停与事后可靠性验收分开。",["../results/formal_summary.csv","../results/bvp_benchmark.csv"])

def main():
    setup()
    formal=pd.read_csv(ROOT/"results"/"formal_summary.csv")
    ablation=pd.read_csv(ROOT/"results"/"ablations_summary.csv")
    comparison(formal,ablation)
    base_profiles(formal,"dirac");base_profiles(formal,"kg")
    coverage(formal)
    deep_profiles(formal,"dirac");deep_profiles(formal,"kg")
    diagnostics(formal);training_cost(formal)
    save_json(OUT/"manifest.json",dict(backend="python/matplotlib",archetype="quantitative grid",
                                     statistics="All seeds; arithmetic means where shown; sample SD in report, no p-values.",
                                     figures={n:dict(caption=CAPTIONS[n],sources=SOURCES[n],formats=["png","svg","pdf"]) for n in CAPTIONS}))

if __name__=="__main__":main()
