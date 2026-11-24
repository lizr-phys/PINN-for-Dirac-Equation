"""Fixed all-seed evidence figures. Generate only after prescribed attempts and audits."""
from __future__ import annotations
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.fft import ifft
from plan import BASE,CORE,EXTENSION,SEEDS,THRESHOLDS,write,sha
from .expected_jobs import expected_jobs

COLORS=dict(original_pde='#8c9cab',static_pde='#526b7c',original_hard='#779059',learned_hard='#287d82',
    analytic_hard='#c47d3e',joint_hard='#a76891',no_carrier_hard='#7768a3',siren='#315b91',fourier='#b94c4c')
LABELS=dict(original_pde='O/P',static_pde='S/P',original_hard='O/H',learned_hard='S/H',analytic_hard='A/H',
    joint_hard='J/H',no_carrier_hard='N/H',siren='SIREN',fourier='Fourier')
METRICS=('spinor_rel_l2_gauge_aligned','current_rel_l2','fidelity_mean','modal_population_rel_l2','max_norm_drift')
NAMES=('Spinor error / 0.10','Current error / 0.15','Fidelity deficit / 0.02','Modal error / 0.10','Norm drift / 0.02')
FIGURES=[]

def normalized(record,key):
    value=record.get(key,np.nan)
    if pd.isna(value):return np.nan
    return (1-float(value))/.02 if key=='fidelity_mean' else float(value)/THRESHOLDS[key]

def save(fig,name,data,caption,conclusion):
    out=BASE/'figures';source=out/'source_data'/f'{name}.csv';source.parent.mkdir(exist_ok=True)
    data.to_csv(source,index=False)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=220)
    plt.close(fig);(out/f'{name}_caption.txt').write_text(caption,encoding='utf-8')
    write(out/f'{name}_metadata.json',dict(status='GENERATED_REQUIRES_ACTUAL_VISUAL_CHECK',conclusion=conclusion,
        script_sha256=sha(__file__),source_path=source.relative_to(BASE).as_posix(),source_sha256=sha(source),
        scientific_source_sha256=sha(BASE/'frozen/source_manifest.json'),design_sha256=sha(BASE/'frozen/design.json'),
        selection_sha256=sha(BASE/'frozen/baseline_selection.json'),
        analysis_input_sha256={p:sha(BASE/p) for p in (
            'analysis_outputs/all_snapshots_metrics.csv','analysis_outputs/paired_statistics.csv',
            'analysis_outputs/factor_interactions.csv','analysis_outputs/algorithm_costs_with_search_overhead.csv',
            'analysis_outputs/independent_direct_carrier_phase.csv','zero_shot/summary_metrics.csv',
            'zero_shot/independent_per_seed_metrics.csv','spectral_benchmark/error_cost.csv',
            'spectral_reference_check/spectral_error_cost_wide_reference.csv') if (BASE/p).exists()},
        exports={ext:sha(out/f'{name}.{ext}') for ext in ('png','svg','pdf')},
        failure_rows_not_removed=True,five_frozen_seeds=[int(s) for s in SEEDS],actual_visual_check_complete=False))
    FIGURES.append(name)

def endpoint(data,case,group,step):
    present=data[(data.case==case)&(data.group==group)&(data.step==step)]
    rows=[]
    for seed in SEEDS:
        found=present[present.seed==seed]
        if len(found)>1:raise RuntimeError('Duplicate metric endpoint')
        if len(found):row=found.iloc[0].to_dict();row['endpoint_observed']=True
        else:
            failure=BASE/'formal/coupled'/case/group/f'seed_{seed}'/'failure.json'
            if not failure.exists():raise RuntimeError('Unexplained missing endpoint')
            record=json.loads(failure.read_text(encoding='utf-8'))
            if record['status']!='DEPENDENCY_FAILED' and record.get('occurred_step',0)>step:raise RuntimeError('Failure after a missing earlier endpoint')
            row=dict(case=case,group=group,seed=seed,step=step,endpoint_observed=False,seed_joint_pass=False,
                failure_path=failure.relative_to(BASE).as_posix(),failure_status=record['status'])
        rows.append(row)
    return pd.DataFrame(rows)

def mark(ax,x,value,color,passed=True):
    if np.isfinite(value):
        ax.scatter(x,max(float(value),1e-18),s=16,edgecolors=color,facecolors=color if passed else 'none',lw=.7,zorder=3)
    else:ax.plot(x,.97,'x',color='#b1272c',ms=5,transform=ax.get_xaxis_transform(),clip_on=False)

def legend(groups,extra=True):
    handles=[Line2D([],[],color=COLORS[g],marker='o',ms=4,lw=1,label=LABELS[g]) for g in groups]
    if extra:handles.extend((Line2D([],[],color='#555555',marker='o',mfc='none',lw=0,label='Fails joint accuracy rule'),
        Line2D([],[],color='#b1272c',marker='x',lw=0,label='No valid endpoint: source/numerical failure')))
    return handles

def same_structure(data):
    fig,axes=plt.subplots(2,4,figsize=(12,6.3),layout='constrained');sources=[]
    statistics=pd.read_csv(BASE/'analysis_outputs/paired_statistics.csv')
    for row,case in enumerate(('T10','m10')):
        a=endpoint(data,case,'analytic_hard',20000);b=endpoint(data,case,'learned_hard',20000)
        for column,(key,label) in enumerate((('spinor_rel_l2_gauge_aligned','Spinor relative error'),
            ('current_rel_l2','Current relative error'),('attributed_training_wall_seconds','Training wall time including carrier (s)'))):
            ax=axes[row,column]
            for index,seed in enumerate(SEEDS):
                aa=a[a.seed==seed].iloc[0];bb=b[b.seed==seed].iloc[0];values=[aa.get(key,np.nan),bb.get(key,np.nan)]
                color=plt.get_cmap('viridis')(index/5)
                if all(pd.notna(v) for v in values):ax.plot([0,1],values,color=color,lw=.9,alpha=.8)
                for x,record,v in zip((0,1),(aa,bb),values):mark(ax,x+(index-2)*.015,float(v) if pd.notna(v) else np.nan,color,bool(record.seed_joint_pass))
                sources.append(dict(case=case,seed=seed,metric=key,analytic=values[0],learned=values[1],
                    difference=values[1]-values[0] if all(pd.notna(v) for v in values) else None,
                    analytic_source=aa.get('path',aa.get('failure_path')),learned_source=bb.get('path',bb.get('failure_path'))))
            ax.set_xticks((0,1),('Analytic A/H','Learned S/H'));ax.set_ylabel(label);ax.set_title(case,loc='left');ax.set_xlim(-.15,1.15)
            if column<2:ax.axhline(THRESHOLDS[key],color='#777777',lw=.7,ls='--')
        ax=axes[row,3];differences=[];key='spinor_rel_l2_gauge_aligned'
        for index,seed in enumerate(SEEDS):
            aa=a[a.seed==seed].iloc[0];bb=b[b.seed==seed].iloc[0]
            av,bv=aa.get(key,np.nan),bb.get(key,np.nan)
            value=bv-av if pd.notna(av) and pd.notna(bv) else np.nan;differences.append(value)
            color=plt.get_cmap('viridis')(index/5)
            if np.isfinite(value):ax.scatter(index,value,color=color,s=18,zorder=3)
            else:ax.plot(index,.96,'x',color='#b1272c',ms=5,transform=ax.get_xaxis_transform(),clip_on=False)
        cohort=statistics[(statistics.case==case)&(statistics.target=='learned_hard')&(statistics.baseline=='analytic_hard')&
            (statistics.target_step==20000)&(statistics.baseline_step==20000)&(statistics.metric==key)]
        estimate={}
        if np.isfinite(differences).all():
            if len(cohort)!=1 or cohort.iloc[0]['status']!='ALL_FIVE_FINITE_PAIRS':raise RuntimeError('Missing prescribed five-pair interval')
            r=cohort.iloc[0];estimate={k:float(r[k]) for k in ('paired_difference_mean','CI95_lower','CI95_upper')}
            center=estimate['paired_difference_mean'];lo=estimate['CI95_lower'];hi=estimate['CI95_upper']
            ax.errorbar(6,center,yerr=[[center-lo],[hi-center]],fmt='D',color='#252525',ms=4,lw=1,capsize=3)
        else:ax.text(.03,.04,'No five-pair interval',transform=ax.transAxes,fontsize=7)
        for seed,value in zip(SEEDS,differences):sources.append(dict(case=case,seed=seed,metric='paired_spinor_difference',
            difference=value,analytic_source=a[a.seed==seed].iloc[0].get('path',a[a.seed==seed].iloc[0].get('failure_path')),
            learned_source=b[b.seed==seed].iloc[0].get('path',b[b.seed==seed].iloc[0].get('failure_path')),**estimate))
        ax.axhline(0,color='#777777',lw=.7,ls='--');ax.set_xlim(-.5,6.8)
        ax.set_xticks([0,1,2,3,4,6],[str(s) for s in SEEDS]+['Mean'],rotation=35,fontsize=7)
        ax.set_ylabel('Paired spinor difference (S/H − A/H)');ax.set_title(case+' · descriptive 95% t interval',loc='left',fontsize=9)
    fig.legend([Line2D([],[],color=plt.get_cmap('viridis')(i/5),lw=1,label=f'Seed {s}') for i,s in enumerate(SEEDS)],
        loc='outside lower center',ncol=5,fontsize=7)
    save(fig,'same_structure_carriers',pd.DataFrame(sources),
        'T10 and m10, all five frozen paired seeds810–814, both at20000 coupled steps with identical hard-norm complex envelopes. '
        'Lines connect the same seed. Analytic A/H uses the known free energy; learned S/H includes its full6000-step carrier expense. '
        'Hollow points fail at least one joint endpoint criterion; crosses denote absent valid endpoints with preserved scientific failures. '
        'Cost is observed optimization wall time, separate from setup/evaluation/I/O. Five-pair descriptive t intervals and sign-flip results '
        'are in analysis_outputs/paired_statistics.csv. The rightmost panel shows all seed differences and a Student-t df4 95% interval '
        'at the mean; negative favors learned. No interval is computed for incomplete pairs; n=5 has limited power.',
        'Direct same-envelope carrier comparison; the direction and cost of differences are determined by all prescribed paired results.')

def mechanisms(data):
    groups=tuple(LABELS)[:7];fig,axes=plt.subplots(5,3,figsize=(11.2,12),layout='constrained');sources=[]
    for column,case in enumerate(CORE):
        for group_index,group in enumerate(groups):
            records=endpoint(data,case,group,20000);sources.append(records)
            for metric_index,key in enumerate(METRICS):
                ax=axes[metric_index,column];values=[]
                for seed_index,(_,record) in enumerate(records.sort_values('seed').iterrows()):
                    v=normalized(record,key);values.append(v);mark(ax,group_index+(seed_index-2)*.055,v,COLORS[group],bool(record.seed_joint_pass))
                if np.isfinite(values).all():ax.plot([group_index-.22,group_index+.22],[max(np.mean(values),1e-18)]*2,color='#262626',lw=1.2)
        for metric_index,key in enumerate(METRICS):
            ax=axes[metric_index,column];ax.set_yscale('log');ax.axhline(1,color='#777777',lw=.7,ls='--')
            ax.set_xticks(range(7),[LABELS[g] for g in groups],fontsize=7);ax.set_xlim(-.5,6.5)
            if column==0:ax.set_ylabel(NAMES[metric_index])
            if metric_index==0:ax.set_title(case,loc='left',weight='bold')
            if metric_index==4:ax.text(.02,.06,'Hard norm: structural identity',transform=ax.transAxes,fontsize=7)
    fig.legend(handles=legend(groups),loc='outside lower center',ncol=5,fontsize=7)
    save(fig,'mechanism_dynamics',pd.concat(sources,ignore_index=True),
        'All seven mechanism groups, three core cases and seeds810–814 at20000 coupled steps. Each point is a prescribed seed; '
        'bars are the five-seed mean only when all five finite values exist. Values are divided by each frozen threshold; '
        'the fidelity panel uses(1-F)/0.02. O=original time phase, S=repaired static frequency, P=unconstrained complex envelope, '
        'H=full-spectrum hard norm, A=analytic carrier, J=joint training and N=no carrier. '
        'Hollow points fail the overall joint accuracy rule; red crosses preserve unavailable scientific-failure endpoints. '
        'Values below1e-18 appear at a display floor; CSV values remain unchanged. Hard norm is a parameterization identity, '
        'so dynamics require spinor/current/fidelity/modal evidence. The2x2 interaction and independent-training-strategy limitations '
        'are reported separately in analysis_outputs/factor_interactions.csv and verification_notes_2026_10_11.md.',
        'Distinguish phase representation, norm structure and training-strategy effects without treating norm conservation as accurate dynamics.')

def scope(data,family,zero=False):
    groups=('learned_hard','analytic_hard',family);cases=EXTENSION if zero else CORE+EXTENSION
    fig,axes=plt.subplots(3,2,figsize=(10.5,9),layout='constrained');sources=[]
    for index,case in enumerate(cases):
        ax=axes[index,0] if zero else axes.flat[index]
        for group_index,group in enumerate(groups):
            if zero:
                records=data[(data.case==case)&(data.group==group)].copy()
                records['endpoint_observed']=records.diagnostic_executed & records.pass_check
            else:records=endpoint(data,case,group,26000 if group in ('siren','fourier') else 20000)
            sources.append(records)
            for metric_index,key in enumerate(METRICS[:4]):
                for seed_index,(_,record) in enumerate(records.sort_values('seed').iterrows()):
                    value=normalized(record,key) if record.endpoint_observed else np.nan
                    mark(ax,metric_index+(group_index-1)*.2+(seed_index-2)*.02,value,COLORS[group],bool(record.seed_joint_pass))
            if zero:
                initial=axes[index,1]
                for seed_index,(_,record) in enumerate(records.sort_values('seed').iterrows()):
                    v=record.get('initial_condition_error_max',np.nan) if record.endpoint_observed else np.nan
                    mark(initial,group_index+(seed_index-2)*.045,float(v) if pd.notna(v) else np.nan,COLORS[group])
                initial.set_xticks(range(3),[LABELS[g] for g in groups]);initial.set_yscale('log')
                initial.set_ylabel('Raw spectral initial-condition max error');initial.set_title(case+' at t=0',loc='left')
        ax.set_xticks(range(4),('Spinor','Current','1-F','Modal'));ax.set_yscale('log');ax.axhline(1,color='#777777',lw=.7,ls='--')
        ax.set_ylabel('Error / frozen threshold');ax.set_title(case+(' — zero-shot diagnostic' if zero else ' — prescribed endpoint'),loc='left')
    fig.legend(handles=legend(groups),loc='outside lower center',ncol=3,fontsize=7)
    caption=('All five frozen seeds810–814 per case and group. Values are divided by their frozen dynamics thresholds; '
        'norm is part of the joint rule and is separately reported as a structural identity. No case pooling. '
        'S/H=learned carrier, A/H=analytic free carrier with identical hard envelope, and the neural family was selected '
        'from development data only; this does not identify the best family at the long formal budget. '
        'Hollow points fail a joint criterion; crosses denote scientifically failed sources without valid predictions. '
        'A1e-18 display floor retains original CSV values. ')
    caption+=('Unchanged complete m5T5 models,45 prescribed diagnostics; weights and initial buffers are unchanged. '
        'Initial-condition mismatch is displayed, without initial-state correction. These are diagnostics, distinct from retraining and carrier reuse.' if zero else
        'Core models are retrained; extension S/H reuses the same-seed m5T5 carrier and retrains its envelope. '
        'S/H and A/H use20000 coupled steps; neural performance endpoints use26000. Total auxiliary and search expenses are retained in cost tables.')
    save(fig,'scope_zero_shot' if zero else 'scope_retrained',pd.concat(sources,ignore_index=True),caption,
        'Each applicability case and inference mode is judged separately; an initial-condition mismatch limits unchanged-model generalization.' if zero else
        'Finite-case applicability at prespecified endpoints with retraining and carrier reuse explicitly distinguished.')

def curves(data,costs,work=False):
    joined=data.merge(costs,on=['case','group','seed','step'],validate='one_to_one')
    xkey='standalone_unit_spinor_residual_points_plus_relevant_search' if work else 'standalone_optimization_plus_relevant_search_seconds'
    for key in (xkey,'spinor_rel_l2_gauge_aligned','seed_joint_pass'):
        if key not in joined:joined[key]=np.nan
    fig,axes=plt.subplots(3,2,figsize=(10.5,9),layout='constrained')
    for case,ax in zip(CORE+EXTENSION,axes.flat):
        subset=joined[joined.case==case]
        for group in LABELS:
            table=subset[subset.group==group]
            for seed in SEEDS:
                t=table[table.seed==seed].sort_values('step')
                ax.plot(t[xkey],t.spinor_rel_l2_gauge_aligned,color=COLORS[group],lw=.6,alpha=.45)
                for _,record in t.iterrows():mark(ax,record[xkey],record.spinor_rel_l2_gauge_aligned,COLORS[group],bool(record.seed_joint_pass))
            complete=[]
            for step,t in table.groupby('step'):
                if len(t)==5 and set(t.seed)==set(SEEDS):complete.append((float(t[xkey].mean()),float(t.spinor_rel_l2_gauge_aligned.mean())))
            if complete:
                x,y=zip(*complete);ax.plot(x,y,color=COLORS[group],lw=1.6,marker='s',ms=3)
        if not work:
            classical=pd.read_csv(BASE/'spectral_reference_check/spectral_error_cost_wide_reference.csv')
            for grid,t in classical[classical.case==case].groupby('grid'):
                t=t.sort_values('cost_median_seconds');ax.plot(t.cost_median_seconds,t.primary161_spinor_error,
                    color='#555555',ls={1024:'--',2048:':',4096:'-.'}[int(grid)],lw=.9,marker='x',ms=3)
        failures=[j for j in expected_jobs() if j['type']=='coupled' and j['stage']!='development' and j['case']==case and (BASE/j['directory']/'failure.json').exists()]
        if failures:ax.text(.02,.98,'Preserved failures: '+', '.join(LABELS[j['group']]+'/'+str(j['seed']) for j in failures),
            transform=ax.transAxes,va='top',color='#b1272c',fontsize=6)
        ax.set_xscale('log');ax.set_yscale('log');ax.axhline(.1,color='#777777',ls='--',lw=.65);ax.set_title(case,loc='left')
        ax.set_ylabel('Spinor relative error');ax.set_xlabel('Unit-spinor residual points, including search' if work else 'Optimization + relevant search wall time (s)')
    handles=legend(tuple(LABELS))
    if not work:handles.extend(Line2D([],[],color='#555555',ls=style,label=f'TSFP N={grid}') for grid,style in ((1024,'--'),(2048,':'),(4096,'-.')))
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=7)
    caption=('Every prescribed snapshot and seed is retained. Thin curves connect fixed checkpoints for one seed; squares and thick curves '
        'are five-seed means only when all five values exist. Learning costs include the complete6000-step source; neural costs include '
        'the prespecified relevant family search once per standalone solve. Separate no-search, pipeline, CPU, memory, inference and '
        'ensemble-allocation columns remain in the source cost ledger. A scientific failure is annotated and not replaced by a best earlier checkpoint. ')
    caption+=('The workload unit is a unit-spinor residual row, not a common FLOP proxy for spectral integration.' if work else
        'TSFP curves show every one of six fixed step sizes at each of three spatial grids, five-repeat median integration cost, '
        'against the immutable primary161-mode reference to match the neural target. Its narrower-reference floor is visible; '
        'the independently calibrated wide-reference TSFP accuracy and all108 configurations are also shown in TSFP_error_cost. '
        'Concurrent CPU wall observations are not an isolated hardware speedup measurement.')
    save(fig,'error_cost_work' if work else 'error_cost_wall',joined,caption,
        'Accuracy versus full auxiliary/search expense at fixed budgets; neural residual workload and mature spectral integration costs have distinct units.')

def carrier_figures():
    rows=[];direct=pd.read_csv(BASE/'analysis_outputs/independent_direct_carrier_phase.csv')
    for job in expected_jobs():
        if job['type']!='carrier':continue
        for step in (1000,3000,6000):
            directory=BASE/job['directory']/'snapshots'/f'{step:05d}';metric=directory/'metrics.json'
            if metric.exists():
                m=json.loads(metric.read_text(encoding='utf-8'));c=json.loads((directory/'cost.json').read_text(encoding='utf-8'))
                independent=direct[(direct.case==job['case'])&(direct.group==job['kind'])&(direct.seed==job['seed'])&(direct.step==step)]
                if len(independent)!=1:raise RuntimeError('Missing direct phase audit for a plotted source')
                phase=independent.iloc[0]
                rows.append(dict(case=job['case'],kind=job['kind'],seed=job['seed'],step=step,
                    metric_sha256=sha(metric),carrier_free_residual_rms=m['carrier_free_residual_rms'],
                    archived_initial_weighted_free_phase_rms=m['initial_weighted_free_phase_rms'],
                    initial_weighted_free_phase_rms=phase.direct_initial_weighted_unwrapped_phase_rms,
                    archived_phase_diagnostic_may_alias=phase.archived_phase_diagnostic_may_alias,
                    phase_checkpoint_sha256=phase.checkpoint_sha256,training_wall_seconds=c['training_wall_seconds']))
            else:rows.append(dict(case=job['case'],kind=job['kind'],seed=job['seed'],step=step,scientific_failure=True,
                carrier_free_residual_rms=np.nan,initial_weighted_free_phase_rms=np.nan,training_wall_seconds=np.nan))
    data=pd.DataFrame(rows);fig,axes=plt.subplots(3,2,figsize=(10.5,8.5),layout='constrained')
    for index,case in enumerate(CORE):
        for column,(key,label) in enumerate((('carrier_free_residual_rms','Free unit-spinor PDE residual RMS'),
            ('initial_weighted_free_phase_rms','Initial-spectrum weighted unwrapped phase RMS'))):
            ax=axes[index,column]
            for kind,color,label_group in (('original','#526b7c','Original time phase'),('static_frequency','#287d82','Static frequency')):
                for seed_index,seed in enumerate(SEEDS):
                    d=data[(data.case==case)&(data.kind==kind)&(data.seed==seed)].sort_values('step')
                    ax.plot(d.step,d[key],color=color,alpha=.65,lw=.7,marker=('o' if kind=='original' else 's'),ms=3,
                        label=label_group if seed_index==0 else None)
                    for _,r in d[d[key].isna()].iterrows():mark(ax,r.step,np.nan,color)
            ax.set_yscale('log');ax.set_xticks((1000,3000,6000));ax.set_xlabel('Prescribed free-training step');ax.set_ylabel(label);ax.set_title(case,loc='left')
    axes[0,0].legend(fontsize=7)
    save(fig,'carrier_pretraining_diagnostics',data,
        'Both carrier parameterizations, all core cases and seeds810–814 at the frozen1000/3000/6000 checkpoints. '
        'Every seed is shown, with unavailable scientific-failure points marked. Exact free energies are used for evaluation only; '
        'both methods already use known free eigenspinors. Unwrapped phase errors use the independent direct real-network output '
        'and the initial spectrum; archived diagnostics and any unwrap alias remain in the CSV. '
        'These free diagnostics do not establish coupled-dynamics accuracy. No extra training is selected from these observations.',
        'Free pretraining diagnostics quantify representation repair while remaining separate from coupled accuracy and practical value.')

def dynamic_data(pred,ref,n,times,mass):
    alpha=float(np.angle(np.vdot(ref.ravel(),pred.ravel())));aligned=pred*np.exp(-1j*alpha)
    modal_p,modal_r=np.sum(abs(pred)**2,-1),np.sum(abs(ref)**2,-1)
    def physical(spec):
        pad=np.zeros((len(times),1024,2),dtype=np.complex128);pad[:,n%1024]=spec*np.exp(-1j*np.pi*n)[None,:,None]
        return ifft(pad,axis=1,norm='ortho',workers=1)*np.sqrt(1024/(16*np.pi))
    p,r=physical(pred),physical(ref);jp=2*(p[...,0].conj()*p[...,1]).real;jr=2*(r[...,0].conj()*r[...,1]).real
    current_scale=float(np.sqrt(np.mean(np.sum(jr**2,axis=1))))
    theta=.5*np.arctan2(n/8,mass);up=np.stack((np.cos(theta),np.sin(theta)),-1);um=np.stack((-np.sin(theta),np.cos(theta)),-1)
    def branches(s):return np.stack((np.sum(s*up[None],-1),np.sum(s*um[None],-1)),-1)
    cp,cr=branches(pred),branches(ref);coherence=[]
    for a,b in zip(cp,cr):
        a,b=a.ravel(),b.ravel();ids=np.flatnonzero(abs(b)**2>=1e-6);a,b=a[ids],b[ids]
        aa=a[:,None]*a.conj()[None];bb=b[:,None]*b.conj()[None];mask=~np.eye(len(ids),dtype=bool)
        coherence.append(np.sqrt(np.sum(abs(aa[mask]-bb[mask])**2)/max(np.sum(abs(bb[mask])**2),1e-300)))
    return pd.DataFrame(dict(t=times,norm_drift=abs(np.sum(abs(pred)**2,(1,2))-1),
        spinor_raw_error=np.sqrt(np.sum(abs(pred-ref)**2,(1,2))),spinor_aligned_error=np.sqrt(np.sum(abs(aligned-ref)**2,(1,2))),
        current_normalized_error=np.sqrt(np.sum((jp-jr)**2,axis=1))/max(current_scale,1e-300),
        modal_relative_error=np.linalg.norm(modal_p-modal_r,axis=1)/np.maximum(np.linalg.norm(modal_r,axis=1),1e-300),
        coherence_relative_error=coherence))

def dynamics(data,family):
    fields=('norm_drift','spinor_raw_error','spinor_aligned_error','current_normalized_error','modal_relative_error','coherence_relative_error')
    labels=('Norm drift (structural for H)','Raw spinor L2 error','Constant-phase-aligned spinor L2 error',
        'Current L2 error / reference time-RMS','Modal population relative error','Active-channel coherence relative error')
    for case in CORE+EXTENSION:
        groups=tuple(LABELS) if case in CORE else ('learned_hard','analytic_hard',family)
        with np.load(BASE/'references'/case/'test201.npz') as f:ref,times=f['state'],f['t']
        fig,axes=plt.subplots(3,2,figsize=(10.5,9),layout='constrained');sources=[];failed=[]
        for group in groups:
            step=26000 if group in ('joint_hard','no_carrier_hard','siren','fourier') else 20000
            row=endpoint(data,case,group,step);record=row[row.seed==810].iloc[0]
            if not record.endpoint_observed:failed.append(group);continue
            raw=BASE/record['path'];raw=raw.parent/'prediction.npz'
            with np.load(raw) as f:pred,n=f['prediction'],f['n']
            table=dynamic_data(pred,ref,n,times,10. if case=='m10' else 5.)
            table.insert(0,'case',case);table.insert(1,'group',group);table.insert(2,'seed',810);table.insert(3,'step',step);sources.append(table)
            for ax,key in zip(axes.flat,fields):ax.plot(times,np.maximum(table[key],1e-18),color=COLORS[group],lw=1)
        for ax,label in zip(axes.flat,labels):
            ax.set_yscale('log');ax.set_ylabel(label);ax.set_xlabel('t');ax.set_xlim(times[0],times[-1])
            if failed:ax.text(.02,.97,'No valid seed810 endpoint: '+', '.join(LABELS[g] for g in failed),transform=ax.transAxes,
                color='#b1272c',va='top',fontsize=6)
        fig.suptitle(case+' — fixed seed810 at prescribed terminal endpoints',fontsize=10)
        fig.legend(handles=legend(groups,extra=False),loc='outside lower center',ncol=5,fontsize=7)
        source=pd.concat(sources,ignore_index=True) if sources else pd.DataFrame([dict(case=case,seed=810,scientific_failure_group=g) for g in failed])
        save(fig,f'dynamics_seed810_{case}',source,
            'Seed810 was fixed before results; failed endpoints are reported without selecting another seed. Terminal performance checkpoints '
            'are26000 coupled steps for J/H,N/H,SIREN and Fourier, and20000 for the other groups, with full6000 auxiliary costs separately attributed. '
            'Raw spinor error retains the initial condition; aligned error removes one global spacetime constant phase only. Current error '
            'uses one reference time-RMS denominator so its time-RMS agrees with the overall current metric. Modal populations cover all '
            'retained modes; coherence includes predicted collapse of reference-active channels. Norm conservation is structural for H. '
            'A1e-18 plot floor retains exact original CSV zeros. Initial data and reference are not used for optimizer updates by this plotting code.',
            'Fixed-seed dynamics distinguish phase, current, modal transport and coherence from structural norm preservation.')

def main():
    attempts=json.loads((BASE/'analysis_outputs/attempt_completion_audit.json').read_text(encoding='utf-8'))
    audit=json.loads((BASE/'analysis_outputs/independent_metric_audit.json').read_text(encoding='utf-8'))
    zero=json.loads((BASE/'zero_shot/independent_audit.json').read_text(encoding='utf-8'))
    if not attempts['all_training_attempts_verified'] or audit['status']!='PASS_AVAILABLE_SNAPSHOTS' or not zero['entire_zero_shot_validation_complete']:
        print('FORMAL_FIGURES_WAIT_FOR_COMPLETE_ATTEMPTS_AND_INDEPENDENT_AUDITS');return
    data=pd.read_csv(BASE/'analysis_outputs/all_snapshots_metrics.csv');data=data[data.stage!='development']
    cost_path=BASE/'analysis_outputs/algorithm_costs_with_search_overhead.csv'
    costs=pd.read_csv(cost_path) if cost_path.stat().st_size>2 else pd.DataFrame(columns=['case','group','seed','step'])
    selection=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'));family=selection['strongest_family']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    same_structure(data);mechanisms(data);scope(data,family)
    scope(pd.read_csv(BASE/'zero_shot/independent_per_seed_metrics.csv'),family,zero=True)
    curves(data,costs);curves(data,costs,work=True);carrier_figures();dynamics(data,family)
    write(BASE/'figures/formal_generation_status.json',dict(status='GENERATED_REQUIRES_ACTUAL_VISUAL_CHECK',figures=FIGURES,
        expected_formal_figures=13,generated=len(FIGURES),whole_figure_package_complete=False,all_fixed_seeds_used=True,
        script_sha256=sha(__file__),actual_visual_check_still_required=True))
    print('FORMAL_FIGURES_GENERATED',len(FIGURES))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
