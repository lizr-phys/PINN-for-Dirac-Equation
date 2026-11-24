from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from .r12_config import ROOT,RAW,FIGURES,CASES,GROUPS,SEEDS,THRESHOLDS,CASE_LABELS,run_directory
from .freeze_r7_phase_release import write_json,sha256

COLORS=dict(zip(GROUPS,('#2471a3','#d96b16','#727b83','#8e55ab')))
SHORT=dict(zip(GROUPS,('A','C','D','E*')))
QA=[]
MARKERS=('o','s','^','D','v')
STYLES=('-', '--', ':', '-.', (0,(5,2)))

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':11,
        'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'lines.linewidth':1.2,'figure.facecolor':'white'})

def source(name,frame):
    d=FIGURES/'source_data';d.mkdir(parents=True,exist_ok=True);path=d/f'{name}.csv'
    frame.to_csv(path,index=False);return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),rows=len(frame))

def finish(fig,name,data_sources,claim,*,all_seeds=True):
    for ax in fig.axes:
        if ax.get_yscale()=='symlog':
            low,high=sorted(ax.get_ylim())
            linear_threshold=ax.yaxis._scale.linthresh
            ticks=ax.yaxis.get_major_locator().tick_values(low,high)
            # Keep zero and logarithmic ticks; crowded interior linear ticks
            # obscure one another without adding evidence to these panels.
            ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if t==0 or abs(t)>=linear_threshold]))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    bad=[];undrawn=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<low or tick.get_loc()>high:
                    undrawn.update((id(tick.label1),id(tick.label2)))
    for artist in fig.findobj():
        if id(artist) not in undrawn and hasattr(artist,'get_text') and artist.get_visible() and artist.get_text():
            extent=artist.get_window_extent(renderer)
            if extent.width and extent.height and (extent.x0 < bounds.x0-2 or extent.y0 < bounds.y0-2 or extent.x1 > bounds.x1+2 or extent.y1 > bounds.y1+2):
                bad.append(artist.get_text())
    for ext in ('png','svg','pdf'):fig.savefig(FIGURES/f'{name}.{ext}',facecolor='white')
    svg=(FIGURES/f'{name}.svg').read_text(encoding='utf-8')
    QA.append(dict(figure=name,claim=claim,source_data=data_sources,all_seeds=all_seeds,
        seeds=list(SEEDS) if all_seeds else [500],bounding_box_pass=not bad,bounding_box_issues=bad,
        svg_editable_text='<text' in svg,exports={ext:sha256(FIGURES/f'{name}.{ext}') for ext in ('png','svg','pdf')}))
    plt.close(fig)

def legend(fig,extra=()):
    handles=[Line2D([],[],color=COLORS[g],marker='o',label=SHORT[g]+(' exact analytic diagnostic' if g==GROUPS[-1] else ''),linestyle='-') for g in GROUPS]
    seeds=[Line2D([],[],color='black',marker=MARKERS[i],ls=STYLES[i],label='Seed '+str(s),ms=4) for i,s in enumerate(SEEDS)]
    fig.legend(handles=handles+seeds+list(extra),loc='outside upper center',ncol=5,frameon=False)

def primary(frame,endpoint):
    data=frame[frame.endpoint==endpoint];name='primary_metrics_'+endpoint
    keys=list(THRESHOLDS);titles=['Spinor relative L2','Current relative L2','Infidelity (1 - F)','Modal population relative L2','Maximum norm drift']
    fig,axes=plt.subplots(4,5,figsize=(18,11),layout='constrained')
    for ci,case in enumerate(CASES):
        for j,key in enumerate(keys):
            ax=axes[ci,j]
            for gi,g in enumerate(GROUPS):
                rows=data[(data.case==case)&(data.model==g)].set_index('seed').loc[list(SEEDS)]
                values=rows[key].to_numpy();values=1-values if key=='fidelity_mean' else values
                for si,value in enumerate(values):ax.scatter(gi+(si-2)*.07,value,s=21,marker=MARKERS[si],color=COLORS[g],alpha=.82,zorder=3)
                ax.errorbar(gi,values.mean(),yerr=values.std(ddof=0),fmt='s',color=COLORS[g],ms=5,capsize=3,zorder=4)
            bound=1-THRESHOLDS[key] if key=='fidelity_mean' else THRESHOLDS[key]
            ax.axhline(bound,color='#b83636',ls='--',lw=.9)
            ax.set_yscale('symlog',linthresh=.005);ax.set_xticks(range(4),[SHORT[g] for g in GROUPS]);ax.set_title(titles[j] if ci==0 else '')
            if j==0:ax.set_ylabel(CASE_LABELS[case]+'\nError (symlog)')
            ax.grid(axis='y',which='major',alpha=.15)
    legend(fig);finish(fig,name,[source(name+'_values',data)],'Per-condition fixed endpoints; dots all seeds, squares mean +/- population SD, red line absolute threshold.')

def paired(pairs):
    selected=pairs[pairs.metric.isin(['spinor_rel_l2_gauge_aligned','current_rel_l2','max_norm_drift'])]
    contrasts=((0,1),(0,2),(1,2),(3,0),(3,1))
    fig,axes=plt.subplots(4,3,figsize=(13,12),layout='constrained')
    for ci,c in enumerate(CASES):
        for j,key in enumerate(('spinor_rel_l2_gauge_aligned','current_rel_l2','max_norm_drift')):
            ax=axes[ci,j]
            for pi,(ai,bi) in enumerate(contrasts):
                for endpoint,marker,dx in (('final','o',-.1),('coupled_20000','^',.1)):
                    a=selected[(selected.case==c)&(selected.metric==key)&(selected.target==GROUPS[ai])&
                        (selected.baseline==GROUPS[bi])&(selected.endpoint==endpoint)].set_index('seed').loc[list(SEEDS)]
                    ax.scatter(pi+dx+np.linspace(-.045,.045,5),a.paired_difference,color=COLORS[GROUPS[ai]],marker=marker,s=18,alpha=.85)
            ax.axhline(0,color='black',ls='--',lw=.8);ax.set_yscale('symlog',linthresh=.002)
            ax.set_xticks(range(len(contrasts)),[SHORT[GROUPS[a]]+'-'+SHORT[GROUPS[b]] for a,b in contrasts])
            if ci==0:ax.set_title(('Spinor difference','Current difference','Norm drift difference')[j])
            if j==0:ax.set_ylabel(CASE_LABELS[c]+'\nTarget - baseline (symlog)')
    fig.legend(handles=[Line2D([],[],marker=m,color='black',ls='',label=l) for m,l in [('o','Final total 26k'),('^','Coupled 20k')]],loc='outside upper center',ncol=2,frameon=False)
    finish(fig,'paired_differences',[source('paired_differences_values',pairs)],'All paired differences, no posthoc superiority gate; A-C mixes pretraining, updates and budget allocation.')

def costs(curves,x,name,label):
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for ci,case in enumerate(CASES):
        for j,key in enumerate(('spinor_rel_l2_gauge_aligned','current_rel_l2')):
            ax=axes[ci,j]
            for group in GROUPS:
                for seed in SEEDS:
                    a=curves[(curves.case==case)&(curves.model==group)&(curves.seed==seed)].sort_values('steps')
                    xv=a[x]/3600 if x=='training_seconds' else a[x]/1e6
                    ax.plot(xv,a[key],color=COLORS[group],marker=MARKERS[SEEDS.index(seed)],ls=STYLES[SEEDS.index(seed)],ms=2.8,alpha=.65,lw=1)
            ax.set_yscale('log');ax.axhline(.1 if j==0 else .15,color='#b83636',ls='--',lw=.8)
            ax.set_ylabel(CASE_LABELS[case]+'\nRelative L2' if j==0 else 'Relative L2')
            ax.set_xlabel(label);ax.set_title(('Spinor','Current')[j] if ci==0 else '')
            ax.grid(alpha=.15)
    legend(fig);finish(fig,name,[source(name+'_curves',curves)],'All fixed cost points; auxiliary pretraining included. Concurrent one-core timings are not isolated acceleration evidence.')

def series_frame():
    parts=[]
    for case in CASES:
        for group in GROUPS:
            for seed in SEEDS:
                table=pd.read_csv(run_directory(case,group,seed)/'time_series.csv')
                table['case'],table['model'],table['seed']=case,group,seed;parts.append(table)
    return pd.concat(parts,ignore_index=True)

def norm_coherence_support(series):
    fig,axes=plt.subplots(4,3,figsize=(14,12),layout='constrained')
    pairs=(('norm_predicted','norm_reference'),('coherence_relative_error',None),('occupation_outside_initial_support_predicted','occupation_outside_initial_support_reference'))
    for ci,c in enumerate(CASES):
        for j,(pred,ref) in enumerate(pairs):
            ax=axes[ci,j]
            for group in GROUPS:
                for seed in SEEDS:
                    a=series[(series.case==c)&(series.model==group)&(series.seed==seed)]
                    y=a[pred]-1 if j==0 else a[pred]
                    ax.plot(a.t,y,color=COLORS[group],ls=STYLES[SEEDS.index(seed)],lw=.9,alpha=.65)
            a=series[(series.case==c)&(series.model==GROUPS[0])&(series.seed==500)]
            if ref and j!=0:ax.plot(a.t,a[ref],color='black',ls='--',lw=1.3)
            if j==0:
                for y in (-.02,.02):ax.axhline(y,color='#b83636',ls='--',lw=.8)
            if j==1:ax.set_yscale('symlog',linthresh=.001)
            ax.set_xlabel('Time');ax.set_title(('Norm - 1','Active coherence relative error','Occupation outside n = 0...32')[j] if ci==0 else '')
            if j==0:ax.set_ylabel(CASE_LABELS[c])
    legend(fig);finish(fig,'norm_coherence_support',[source('norm_coherence_support_series',series)],'All seeds, collapse retained in coherence, new support defined by original Gaussian index interval.')

def trajectories(series):
    fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained')
    quantities=('plus_center','minus_center','plus_width','minus_width')
    for ci,c in enumerate(CASES):
        for j,q in enumerate(quantities):
            ax=axes[ci,j]
            for g in GROUPS:
                for seed in SEEDS:
                    a=series[(series.case==c)&(series.model==g)&(series.seed==seed)]
                    ax.plot(a.t,a[q+'_predicted']-a[q+'_reference'],color=COLORS[g],ls=STYLES[SEEDS.index(seed)],alpha=.65,lw=.9)
            ax.axhline(0,color='black',ls='--',lw=.8);ax.set_xlabel('Time')
            if ci==0:ax.set_title(q.replace('_',' ')+' error')
            if j==0:ax.set_ylabel(CASE_LABELS[c]+'\nPosition error')
    legend(fig);finish(fig,'branch_trajectories',[source('branch_trajectories_series',series)],'Circular centers and widths relative to each coupled reference; collapsed branches are failure diagnostics.')

def diagnostics(frame):
    data=frame[frame.endpoint=='final']
    keys=('carrier_free_pde_residual_rms','weighted_carrier_defect_spinor_rms','residual_integral_fine','exterior_residual_integral')
    fig,axes=plt.subplots(4,4,figsize=(15,11),layout='constrained')
    for ci,c in enumerate(CASES):
        for j,key in enumerate(keys):
            ax=axes[ci,j]
            for gi,g in enumerate(GROUPS):
                a=data[(data.case==c)&(data.model==g)].set_index('seed').loc[list(SEEDS)]
                values=a[key].to_numpy()
                if np.isfinite(values).any():
                    for si,value in enumerate(values):ax.scatter(gi+(si-2)*.07,value,color=COLORS[g],marker=MARKERS[si],s=20)
            ax.set_yscale('log');ax.set_xticks(range(4),[SHORT[g] for g in GROUPS])
            if ci==0:ax.set_title(('Unit free-carrier defect','Weighted carrier defect','Retained residual integral','Exterior residual integral')[j])
            if j==0:ax.set_ylabel(CASE_LABELS[c])
    legend(fig);finish(fig,'carrier_residual_diagnostics',[source('carrier_residual_diagnostics_values',data)],'C has internal phase gauge freedom; carrier defects do not alone judge complete dynamics; quadrature is not continuous certification.')

def dynamics(case):
    fields=[];rows=[]
    for group in GROUPS:
        with np.load(run_directory(case,group,500)/'dense_grid.npz') as a:
            fields.append({key:a[key] for key in ('t','x','density_predicted','density_reference','current_predicted','current_reference')})
    ref=fields[0];fig,axes=plt.subplots(4,5,figsize=(17,10),layout='constrained')
    rho_max=max(np.max(f['density_predicted']) for f in fields+[dict(density_predicted=ref['density_reference'])])
    current_max=max(np.max(abs(f['current_predicted'])) for f in fields+[dict(current_predicted=ref['current_reference'])])
    drho=max(np.max(abs(f['density_predicted']-ref['density_reference'])) for f in fields)
    dj=max(np.max(abs(f['current_predicted']-ref['current_reference'])) for f in fields)
    ti,xi=np.indices(ref['density_reference'].shape)
    for col in range(5):
        f=ref if col==0 else fields[col-1]
        values=(ref['density_reference'],ref['current_reference'],np.zeros_like(ref['density_reference']),np.zeros_like(ref['current_reference'])) if col==0 else (
            f['density_predicted'],f['current_predicted'],f['density_predicted']-ref['density_reference'],f['current_predicted']-ref['current_reference'])
        for row,z in enumerate(values):
            ax=axes[row,col];vmax=(rho_max,current_max,max(drho,1e-15),max(dj,1e-15))[row]
            im=ax.pcolormesh(ref['x'],ref['t'],z,shading='auto',rasterized=True,cmap='viridis' if row==0 else 'RdBu_r',vmin=0 if row==0 else -vmax,vmax=vmax)
            if row==0:ax.set_title('Reference' if col==0 else SHORT[GROUPS[col-1]])
            if col==0:ax.set_ylabel(('Density\nTime','Current\nTime','Density error\nTime','Current error\nTime')[row])
            if row==3:ax.set_xlabel('x')
            ax.set_xlim(ref['x'][0],ref['x'][-1])
            if col==4:fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.87,pad=.012)
        rows.append(pd.DataFrame(dict(case=case,model='reference' if col==0 else GROUPS[col-1],seed=500,
            t=ref['t'][ti.ravel()],x=ref['x'][xi.ravel()],density=values[0].ravel(),current=values[1].ravel(),
            density_error=values[2].ravel(),current_error=values[3].ravel())))
    fig.suptitle(CASE_LABELS[case]+'; predetermined seed 500; full periodic spatial domain')
    finish(fig,'dynamics_'+case+'_seed500',[source('dynamics_'+case+'_seed500_grid',pd.concat(rows,ignore_index=True))],
        'Fixed seed 500 for all groups, shared physical and error color scales; full-domain predictions preserved.',all_seeds=False)

def plot():
    FIGURES.mkdir(parents=True,exist_ok=True);style();QA.clear()
    frame=pd.read_csv(RAW/'per_seed_metrics.csv',float_precision='round_trip')
    curves=pd.read_csv(RAW/'error_cost_curves.csv',float_precision='round_trip')
    primary(frame,'final');primary(frame,'coupled_20000')
    paired(pd.read_csv(RAW/'paired_comparisons.csv',float_precision='round_trip'))
    costs(curves,'training_seconds','error_vs_total_cost','Measured cumulative training time (core-hours)')
    costs(curves,'training_spinor_residual_points','error_vs_residual_work','Cumulative spinor residual points (millions)')
    series=series_frame();norm_coherence_support(series);trajectories(series);diagnostics(frame)
    for c in CASES:dynamics(c)
    write_json(FIGURES/'FIGURE_QA.json',dict(status='PENDING_VISUAL_INSPECTION',visual_inspection_pending=True,
        backend='Python/matplotlib',figures=QA,all_seeds=list(SEEDS),dynamics_seed=500,
        population_sd_ddof=0,no_data_smoothing=True,no_seed_selection=True))
    print('Exported',len(QA),'R12 figures',flush=True)

if __name__=='__main__':plot()
