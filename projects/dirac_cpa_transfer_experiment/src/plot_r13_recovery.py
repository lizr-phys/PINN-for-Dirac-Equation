from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r13_config import ROOT,RAW,FIGURES,THRESHOLDS
from .freeze_r7_phase_release import write_json,sha256

COLORS={'A_original':'#3975A6','original':'#3975A6','scaled_phase':'#D49A49',
    'static_frequency':'#5A8C7C','pde_only':'#82699D','norm_regularized':'#C5A15D',
    'hard_total_norm':'#668B79','carrier_only':'#D49A49','envelope_only':'#668B79',
    'combined':'#3F897D','selected_combination':'#3F897D','E_pde_only':'#82699D',
    'E_pde_only_alias':'#82699D','E_repaired_envelope_alias':'#A78BBE'}
LABELS={'A_original':'A original','original':'Original phase','scaled_phase':'Scaled phase',
    'static_frequency':'Static frequency','pde_only':'E PDE only (analytic)',
    'norm_regularized':'E norm penalty (analytic)','hard_total_norm':'E total norm (analytic)',
    'carrier_only':'Carrier repair','envelope_only':'Envelope repair','combined':'Both repairs',
    'selected_combination':'Selected combination','E_pde_only':'E PDE only (analytic)',
    'E_pde_only_alias':'E PDE only (alias)','E_repaired_envelope_alias':'E same norm repair (alias)'}
SHORT={'spinor_rel_l2_gauge_aligned':'Spinor error','current_rel_l2':'Current error',
    'fidelity_mean':'Infidelity (1 - F)','modal_population_rel_l2':'Modal population error','max_norm_drift':'Max norm drift'}

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})

def save(fig,name,source,contract):
    FIGURES.mkdir(parents=True,exist_ok=True);data=FIGURES/'source_data';data.mkdir(exist_ok=True)
    csv=data/(name+'.csv')
    existing=[p for p in [csv]+[FIGURES/(name+'.'+s) for s in ('png','svg','pdf')] if p.exists()]
    if existing:
        import shutil
        from datetime import datetime,timezone
        archive=FIGURES/'export_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')/name
        archive.mkdir(parents=True)
        for p in existing:shutil.copyfile(p,archive/p.name)
        if (FIGURES/'figure_registry.json').exists():shutil.copyfile(FIGURES/'figure_registry.json',archive/'registry_before.json')
    source.to_csv(csv,index=False)
    paths=[]
    for suffix in ('png','svg','pdf'):
        path=FIGURES/(name+'.'+suffix);fig.savefig(path,bbox_inches='tight',facecolor='white');paths.append(path)
    plt.close(fig)
    return dict(name=name,source_csv=csv.relative_to(ROOT).as_posix(),source_sha256=sha256(csv),
        exports={p.suffix[1:]:dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in paths},contract=contract)

def metric_grid(frame,name,cases):
    fig,axes=plt.subplots(len(cases),5,figsize=(12,2.8*len(cases)),squeeze=False,layout='constrained')
    for row,case in enumerate(cases):
        data=frame[frame.case==case];groups=list(data.group.unique())
        for col,(key,bound) in enumerate(THRESHOLDS.items()):
            ax=axes[row,col]
            for pos,group in enumerate(groups):
                values=data[data.group==group].sort_values('seed')[key].to_numpy()
                if key=='fidelity_mean':values=1-values
                offsets=np.linspace(-.13,.13,len(values))
                ax.scatter(pos+offsets,values,c=COLORS.get(group,'#777777'),s=20,alpha=.85,zorder=3)
                if len(values):ax.errorbar(pos,float(values.mean()),yerr=float(values.std(ddof=0)),fmt='_',
                    color=COLORS.get(group,'#777777'),capsize=3,linewidth=1.3)
            actual_bound=1-bound if key=='fidelity_mean' else bound
            ax.axhline(actual_bound,color='#555555',ls='--',lw=.8)
            ax.set_yscale('symlog',linthresh=1e-4 if key!='max_norm_drift' else 1e-5)
            ax.set_xticks(range(len(groups)),[LABELS.get(g,g) for g in groups],rotation=55,ha='right',fontsize=6)
            ax.set_title(SHORT[key]);ax.set_ylabel(f'{case}; dimensionless')
            ax.grid(axis='y',alpha=.16)
    return save(fig,name,frame,'All fixed seeds; points and population SD(ddof0); dashed preregistered limits; infidelity displayed explicitly')

def phase_time(carriers,case):
    data=carriers[(carriers.stage=='free_carriers')&(carriers.case==case)];records=[]
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    keys=('initial_weighted_unwrapped_phase_rmse','initial_weighted_carrier_chord_rmse','full_spectrum_unit_spinor_residual_rms')
    for _,row in data.iterrows():
        series=pd.read_csv(ROOT/row.directory/'time_series.csv');series=series.assign(case=case,group=row.group,seed=row.seed)
        records.append(series)
        for ax,key in zip(axes,keys):
            ax.plot(series.t,series[key],color=COLORS[row.group],lw=.9,alpha=.65,label=LABELS[row.group] if row.seed==600 else None)
    for ax,title in zip(axes,('Initial weighted phase error (rad)','Initial weighted carrier chord error','All-mode unit spinor defect')):
        ax.set_xlabel('Physical time');ax.set_title(title);ax.set_yscale('symlog',linthresh=1e-4);ax.grid(alpha=.16)
    axes[0].legend(fontsize=7)
    return save(fig,'free_phase_time_'+case,pd.concat(records,ignore_index=True),'No phase alignment; each of the three seeds is a separate line; weights fixed by original Gaussian')

def free_training(snapshots):
    data=snapshots[snapshots.job_type=='carrier'];data=data[data.stage=='free_carriers']
    fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
    for col,case in enumerate(('T10','m10')):
        for (group,seed),f in data[data.case==case].groupby(['group','seed']):
            f=f.sort_values('executed_steps')
            axes[0,col].plot(f.training_seconds/3600,f.initial_weighted_unwrapped_phase_rms,'o-',ms=3,color=COLORS[group],alpha=.65,label=LABELS[group] if seed==600 else None)
            axes[1,col].plot(f.executed_steps,f.carrier_free_pde_residual_rms,'o-',ms=3,color=COLORS[group],alpha=.65)
        axes[0,col].set_title(case);axes[0,col].set_xlabel('Carrier training hours');axes[0,col].set_ylabel('Initial weighted phase RMS (rad)')
        axes[1,col].set_xlabel('Prescheduled free training step');axes[1,col].set_ylabel('Full-spectrum unit residual RMS')
        for ax in axes[:,col]:ax.set_yscale('log');ax.grid(alpha=.16)
    axes[0,0].legend(fontsize=7)
    return save(fig,'free_phase_training',data,'Only prescribed 1000/3000/6000; all seeds; no best-checkpoint selection')

def cost_curves(snapshots):
    data=snapshots[snapshots.job_type=='coupled'].copy();panels=list(data[['stage','case']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(int(np.ceil(len(panels)/2)),2,figsize=(10,2.8*int(np.ceil(len(panels)/2))),layout='constrained',squeeze=False)
    for ax,(stage,case) in zip(axes.ravel(),panels):
        for (group,seed),f in data[(data.stage==stage)&(data.case==case)].groupby(['group','seed']):
            f=f.sort_values('executed_steps')
            ax.plot(f.full_attributed_pipeline_seconds/3600,f.spinor_rel_l2_gauge_aligned,'o-',color=COLORS.get(group,'#777777'),ms=2.5,alpha=.6,
                label=LABELS.get(group,group) if seed in (600,700) else None)
        ax.axhline(.10,color='#555555',ls='--',lw=.7);ax.set_title(stage+' / '+case)
        ax.set_xlabel('Attributed pipeline hours (training, evaluation, I/O)');ax.set_ylabel('Spinor relative error')
        ax.set_yscale('log');ax.legend(fontsize=6,frameon=True,facecolor='white',edgecolor='none',framealpha=1);ax.grid(alpha=.16)
    for ax in axes.ravel()[len(panels):]:ax.set_visible(False)
    return save(fig,'error_vs_total_cost',data,'All fixed checkpoints and seeds; carrier source cost fully attributed; concurrent times are not isolated hardware speed')

def norm_current(coupled):
    data=coupled[coupled.stage=='independent_validation'];records=[]
    fig,axes=plt.subplots(3,2,figsize=(10,8),layout='constrained')
    for col,case in enumerate(('T10','m10')):
        for _,row in data[data.case==case].iterrows():
            series=pd.read_csv(ROOT/row.directory/'time_series.csv')
            with np.load(ROOT/row.directory/'dense_grid.npz') as dense:
                numerator=np.linalg.norm(dense['current_predicted']-dense['current_reference'],axis=1)
                denominator=np.linalg.norm(dense['current_reference'],axis=1)
                series['current_relative_spatial_l2']=numerator/np.maximum(denominator,1e-300)
            series=series.assign(case=case,group=row.group,seed=row.seed);records.append(series)
            for ax,key in zip(axes[:,col],('norm_predicted','current_relative_spatial_l2','coherence_relative_error')):
                ax.plot(series.t,series[key],color=COLORS[row.group],alpha=.6,lw=.8,label=LABELS[row.group] if row.seed==700 else None)
        axes[0,col].axhline(1,color='#555555',ls='--',lw=.8);axes[0,col].set_title(case);axes[0,col].legend(fontsize=7)
        for ax,label in zip(axes[:,col],('Total probability','Current spatial relative error','Reference-mask coherence error')):
            ax.set_xlabel('Physical time');ax.set_ylabel(label);ax.grid(alpha=.16)
        for ax in axes[1:,col]:ax.set_yscale('symlog',linthresh=1e-4)
    return save(fig,'norm_current_coherence',pd.concat(records,ignore_index=True),'All five independent seeds; reference-defined coherence mask retains collapsed predictions; no posthoc norm correction')

def dynamics(coupled,stage,case,seed):
    data=coupled[(coupled.stage==stage)&(coupled.case==case)&(coupled.seed==seed)]
    groups=list(data.group.unique());values={};source={};times=None;xs=None
    for _,row in data.iterrows():
        with np.load(ROOT/row.directory/'dense_grid.npz') as d:
            times,xs=d['t'],d['x']
            values[row.group]=(d['density_predicted'].copy(),d['current_predicted'].copy())
            values['Reference']=(d['density_reference'].copy(),d['current_reference'].copy())
    if not values:return None
    ordered=['Reference']+groups;fig,axes=plt.subplots(4,len(ordered),figsize=(max(8,2.1*len(ordered)),8),layout='constrained',squeeze=False)
    vmax_density=max(float(v[0].max()) for v in values.values());vmax_current=max(float(abs(v[1]).max()) for v in values.values())
    de=max(float(abs(values[g][0]-values['Reference'][0]).max()) for g in groups)
    je=max(float(abs(values[g][1]-values['Reference'][1]).max()) for g in groups)
    tt,xx=np.meshgrid(times,xs,indexing='ij');source=dict(t=tt.ravel(),x=xx.ravel())
    for col,group in enumerate(ordered):
        density,current=values[group];difference_density=density-values['Reference'][0];difference_current=current-values['Reference'][1]
        source.update({group+'_density':density.ravel(),group+'_current':current.ravel(),
            group+'_density_error':difference_density.ravel(),group+'_current_error':difference_current.ravel()})
        for row,(matrix,cmap,maximum) in enumerate(((density,'viridis',vmax_density),(current,'RdBu_r',vmax_current),
            (difference_density,'RdBu_r',de),(difference_current,'RdBu_r',je))):
            ax=axes[row,col]
            image=ax.imshow(matrix,extent=(xs[0],xs[-1],times[-1],times[0]),aspect='auto',interpolation='nearest',
                cmap=cmap,vmin=0 if row==0 else -max(maximum,1e-15),vmax=max(maximum,1e-15),rasterized=True)
            if col==0:ax.set_ylabel(('Density','Current','Density error','Current error')[row]+'; t')
            ax.set_xlabel('x')
            if row==0:ax.set_title('Reference' if group=='Reference' else LABELS.get(group,group),fontsize=7)
            if col==len(ordered)-1:fig.colorbar(image,ax=axes[row,:],shrink=.8,location='right',fraction=.02)
    return save(fig,f'dynamics_{stage}_{case}_seed{seed}',pd.DataFrame(source),
        'Preregistered seed; shared physical ranges include every shown prediction; signed errors separately scaled; no smoothing')

def plot_all():
    style();coupled=pd.read_csv(RAW/'per_seed_metrics.csv');carriers=pd.read_csv(RAW/'carrier_per_seed_metrics.csv');snapshots=pd.read_csv(RAW/'snapshot_index.csv')
    registry=[]
    for stage,name,cases in (('bridge','bridge_metrics',('bridge',)),('exact_envelopes','exact_envelope_metrics',('T10','m10')),
        ('T10_factorial','factorial_T10',('T10',)),('independent_validation','validation_metrics',('T10','m10'))):
        frame=coupled[coupled.stage==stage]
        if len(frame):registry.append(metric_grid(frame,name,cases))
    for case in ('T10','m10'):registry.append(phase_time(carriers,case))
    registry.extend((free_training(snapshots),cost_curves(snapshots),norm_current(coupled)))
    for case in ('T10','m10'):
        entry=dynamics(coupled,'independent_validation',case,700)
        if entry:registry.append(entry)
    entry=dynamics(coupled,'T10_factorial','T10',600)
    if entry:registry.append(entry)
    write_json(FIGURES/'figure_registry.json',dict(backend='Python matplotlib',all_seeds_shown=True,
        fixed_dynamics_seeds=[600,700],figures=registry))
    return registry

if __name__=='__main__':plot_all()
