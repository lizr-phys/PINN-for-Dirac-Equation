"""Independent inventory from the frozen protocol; never schedules training."""
from __future__ import annotations
import json
from plan import BASE

def expected_jobs():
    d=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    selection_path=BASE/'frozen/baseline_selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else None
    jobs=[]
    def carrier(case,kind,seed):
        return dict(type='carrier',stage='core',case=case,kind=kind,seed=seed,budget=6000,
            snapshots=[1000,3000,6000],directory=f'formal/carriers/{case}/{kind}/seed_{seed}')
    def coupled(case,group,seed,stage,options=None,candidate=None):
        dev=stage=='development'
        budget=4000 if dev else (26000 if group in ('siren','fourier','joint_hard','no_carrier_hard') else 20000)
        job=dict(type='coupled',stage=stage,case=case,group=group,seed=seed,budget=budget,
            snapshots=[4000] if dev else [s for s in (5000,10000,15000,20000,26000) if s<=budget],
            directory=f'{"development" if dev else "formal/coupled"}/{case}/{group}'+
                (f'/candidate_{candidate}' if dev else '')+f'/seed_{seed}')
        if options is not None:job['options']=options
        if candidate is not None:job['candidate']=candidate
        if group in ('original_pde','original_hard','static_pde','learned_hard'):
            sourcecase='m5T5' if case in d['extension_cases'] else case
            kind='original' if group.startswith('original') else 'static_frequency'
            job.update(source=carrier(sourcecase,kind,seed)['directory'],reuse_source=case in d['extension_cases'])
        return job
    tuning=d['baselines']['tuning_configs']
    for case in d['baselines']['tuning_cases']:
        for family,candidates in tuning.items():
            for i,opt in enumerate(candidates):
                for seed in d['development_seeds']:
                    jobs.append(coupled(case,family,seed,'development',opt,i))
    for case in d['core_cases']:
        for seed in d['formal_seeds']:
            jobs.extend(carrier(case,kind,seed) for kind in ('original','static_frequency'))
            for group in list(d['mechanism_groups'])+d['baselines']['families']:
                opt=None
                if group in tuning and selection:
                    idx=selection['per_case_candidate'].get(case,selection['global_candidate'])[group]
                    opt=tuning[group][idx]
                jobs.append(coupled(case,group,seed,'core',opt))
    for case in d['extension_cases']:
        for seed in d['formal_seeds']:
            # Before selection, retain an explicit unresolved family, rather than silently omitting jobs.
            family=selection['strongest_family'] if selection else 'UNSELECTED_NEURAL'
            for group in ('learned_hard','analytic_hard',family):
                opt=tuning[group][selection['global_candidate'][group]] if group in tuning and selection else None
                jobs.append(coupled(case,group,seed,'extension',opt))
    assert len(jobs)==234 and len({j['directory'] for j in jobs})==234
    return jobs

