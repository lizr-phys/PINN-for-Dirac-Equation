from __future__ import annotations

import json
from .r13_config import RAW, ROOT, EXPLORATION_SEEDS, VALIDATION_SEEDS, CARRIER_KINDS, ENVELOPE_KINDS, carrier_directory, run_directory

STAGES=('bridge','free_carriers','exact_envelopes','T10_factorial','independent_validation')

def _source(stage,case,kind,seed):
    return carrier_directory(stage,case,kind,seed).relative_to(ROOT).as_posix()

def jobs_for(stage,seed):
    if stage not in STAGES:raise ValueError(stage)
    seeds=VALIDATION_SEEDS if stage=='independent_validation' else EXPLORATION_SEEDS
    if seed not in seeds:raise ValueError(seed)
    rotation=seeds.index(seed)
    jobs=[]
    def carrier(case,kind):
        jobs.append(dict(type='carrier',stage=stage,case=case,carrier_kind=kind,seed=seed))
    def coupled(case,group,kind,envelope,source_stage=None):
        jobs.append(dict(type='coupled',stage=stage,case=case,group=group,carrier_kind=kind,envelope_kind=envelope,seed=seed,
            **({'pretrain_source':_source(source_stage or stage,case,kind,seed)} if kind!='exact' else {})))
    if stage=='bridge':
        carrier('bridge','original')
        groups=('A_original','E_pde_only')
        for group in groups[rotation%2:]+groups[:rotation%2]:
            coupled('bridge',group,'original' if group=='A_original' else 'exact','pde_only')
    elif stage=='free_carriers':
        for i,case in enumerate(('T10','m10')):
            r=(rotation+i)%3
            for kind in CARRIER_KINDS[r:]+CARRIER_KINDS[:r]:carrier(case,kind)
    elif stage=='exact_envelopes':
        for i,case in enumerate(('T10','m10')):
            r=(rotation+i)%3
            for envelope in ENVELOPE_KINDS[r:]+ENVELOPE_KINDS[:r]:coupled(case,envelope,'exact',envelope)
    elif stage=='T10_factorial':
        ck=json.loads((RAW/'carrier_selection.json').read_text(encoding='utf-8'))['kind']
        ek=json.loads((RAW/'envelope_selection.json').read_text(encoding='utf-8'))['kind']
        groups=(('A_original','original','pde_only'),('carrier_only',ck,'pde_only'),
            ('envelope_only','original',ek),('combined',ck,ek))
        r=rotation%4
        for group,kind,envelope in groups[r:]+groups[:r]:coupled('T10',group,kind,envelope,'free_carriers')
    else:
        selection=json.loads((RAW/'combination_selection.json').read_text(encoding='utf-8'))
        for i,case in enumerate(('T10','m10')):
            carrier(case,'original')
            if selection['selected']:carrier(case,selection['carrier_kind'])
            groups=[('A_original','original','pde_only'),('E_pde_only','exact','pde_only')]
            if selection['selected']:groups.insert(1,('selected_combination',selection['carrier_kind'],selection['envelope_kind']))
            r=(rotation+i)%len(groups)
            for group,kind,envelope in groups[r:]+groups[:r]:coupled(case,group,kind,envelope)
    return jobs

def directory_for(job, *, smoke=False):
    if job['type']=='carrier':
        return carrier_directory(job['stage'],job['case'],job['carrier_kind'],job['seed'],smoke=smoke)
    return run_directory(job['stage'],job['case'],job['group'],job['seed'],smoke=smoke)

def validate_job(job):
    if job not in jobs_for(job['stage'],job['seed']):
        raise RuntimeError('Job outside frozen/adaptively preregistered R13 plan')
    if job['stage']=='independent_validation':
        selection=json.loads((RAW/'combination_selection.json').read_text(encoding='utf-8'))
        if not selection['no_validation_seed_used']:raise RuntimeError('Contaminated validation selection')
    return True
