from __future__ import annotations
import json,math
from plan import BASE,CORE,EXTENSION,SEEDS,DEV_SEEDS,GROUPS,TUNING,THRESHOLDS,write,sha

def carrier(case,kind,seed):
    return dict(type='carrier',stage='core',case=case,kind=kind,seed=seed,budget=6000,snapshots=[1000,3000,6000],
        directory=f'formal/carriers/{case}/{kind}/seed_{seed}')

def coupled(case,group,seed,*,stage='core',options=None,candidate=None):
    dev=stage=='development'
    budget=4000 if dev else (26000 if group in ('siren','fourier','joint_hard','no_carrier_hard') else 20000)
    j=dict(type='coupled',stage=stage,case=case,group=group,seed=seed,budget=budget,
        snapshots=[4000] if dev else [s for s in (5000,10000,15000,20000,26000) if s<=budget],
        directory=f'{"development" if dev else "formal/coupled"}/{case}/{group}'+(f'/candidate_{candidate}' if dev else '')+f'/seed_{seed}')
    if options is not None:j['options']=dict(options)
    if candidate is not None:j['candidate']=candidate
    if group in ('original_pde','original_hard','static_pde','learned_hard'):
        kind='original' if group.startswith('original') else 'static_frequency'
        sourcecase='m5T5' if case in EXTENSION else case
        j.update(source=carrier(sourcecase,kind,seed)['directory'],reuse_source=case in EXTENSION)
    return j

def development_jobs():
    return [coupled(case,family,seed,stage='development',options=opt,candidate=i) for case in ('T10','m10')
        for family,candidates in TUNING.items() for i,opt in enumerate(candidates) for seed in DEV_SEEDS]

def select_baselines():
    path=BASE/'frozen/baseline_selection.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for job in development_jobs():
        d=BASE/job['directory'];m=d/'snapshots/04000/metrics.json'
        if m.exists():
            metric=json.loads(m.read_text(encoding='utf-8'));score=metric['spinor_rel_l2_gauge_aligned']/.10+metric['current_rel_l2']/.15+metric['modal_population_rel_l2']/.10+(1-metric['fidelity_mean'])/.02
        elif (d/'failure.json').exists():score=None
        else:raise RuntimeError('Development not finished: '+str(d))
        rows.append(dict(case=job['case'],family=job['group'],candidate=job['candidate'],seed=job['seed'],score=score,
            evidence=str(m.relative_to(BASE)) if m.exists() else str((d/'failure.json').relative_to(BASE)),sha256=sha(m if m.exists() else d/'failure.json')))
    def rank(family,case=None):
        scores=[]
        for i in range(len(TUNING[family])):
            r=[x for x in rows if x['family']==family and x['candidate']==i and (case is None or x['case']==case)]
            failed=sum(x['score'] is None for x in r);finite=[x['score'] for x in r if x['score'] is not None]
            scores.append((failed,sum(finite)/len(finite) if finite else float('inf'),i))
        return min(scores)
    percase={c:{f:rank(f,c)[2] for f in TUNING} for c in ('T10','m10')}
    globalpick={f:rank(f)[2] for f in TUNING}
    strongest=min(TUNING,key=lambda f:(rank(f)[0],rank(f)[1],list(TUNING).index(f)))
    result=dict(per_case_candidate=percase,global_candidate=globalpick,strongest_family=strongest,development_rows=rows,
        no_formal_seed_used=True,fixed_search_finished=True,no_extra_candidates=True)
    write(path,result);return result

def core_jobs(case,seed,selection):
    result=[carrier(case,kind,seed) for kind in ('original','static_frequency')]
    groups=list(GROUPS)+['siren','fourier'];offset=(CORE.index(case)+SEEDS.index(seed))%len(groups);groups=groups[offset:]+groups[:offset]
    for group in groups:
        opt=None
        if group in TUNING:
            i=selection['per_case_candidate'].get(case,selection['global_candidate'])[group];opt=TUNING[group][i]
        result.append(coupled(case,group,seed,options=opt))
    return result

def extension_jobs(case,seed,selection):
    groups=['learned_hard','analytic_hard',selection['strongest_family']]
    offset=(EXTENSION.index(case)+SEEDS.index(seed))%3;groups=groups[offset:]+groups[:offset]
    result=[]
    for group in groups:
        opt=TUNING[group][selection['global_candidate'][group]] if group in TUNING else None
        result.append(coupled(case,group,seed,stage='extension',options=opt))
    return result
