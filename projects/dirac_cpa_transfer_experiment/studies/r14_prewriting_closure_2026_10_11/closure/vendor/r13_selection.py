from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from .r13_config import RAW, ROOT, THRESHOLDS, EXPLORATION_SEEDS, VALIDATION_SEEDS, CARRIER_KINDS, ENVELOPE_KINDS, carrier_directory, run_directory
from .freeze_r7_phase_release import write_json,sha256

def absolute_success(frame, *, seeds=VALIDATION_SEEDS, required_joint=4):
    complete=len(frame)==len(seeds) and sorted(frame.seed.tolist())==list(seeds)
    means={key:float(frame[key].mean()) if key in frame and len(frame) else None for key in THRESHOLDS}
    checks={key:means[key] is not None and math.isfinite(means[key]) and
        (means[key]>=bound if key=='fidelity_mean' else means[key]<=bound) for key,bound in THRESHOLDS.items()}
    joint=np.ones(len(frame),dtype=bool)
    for key,bound in THRESHOLDS.items():
        if key not in frame:
            joint[:]=False;continue
        v=frame[key].to_numpy();joint&=np.isfinite(v)&(v>=bound if key=='fidelity_mean' else v<=bound)
    return dict(complete=complete,means=means,mean_checks=checks,joint_pass_count=int(joint.sum()),
        joint_pass_seeds=frame.seed.to_numpy()[joint].tolist() if len(frame) else [],
        pass_check=bool(complete and all(checks.values()) and joint.sum()>=required_joint))

def rows(paths):
    records=[]
    for path in paths:
        if (path/'metrics.json').is_file():
            records.append(json.loads((path/'metrics.json').read_text(encoding='utf-8')))
    return pd.DataFrame(records)

def evidence(paths):
    return {str((p/'metrics.json').relative_to(ROOT).as_posix()):sha256(p/'metrics.json') for p in paths if (p/'metrics.json').exists()}

def _write_once(name, record):
    path=RAW/name
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        for p,digest in old['evidence_sha256'].items():
            if sha256(ROOT/p)!=digest:raise RuntimeError('Selection evidence changed: '+p)
        return old
    write_json(path,record);return record

def select_carrier():
    paths=[carrier_directory('free_carriers',c,k,s) for c in ('T10','m10') for k in CARRIER_KINDS for s in EXPLORATION_SEEDS]
    data=rows(paths);records=[]
    for kind in CARRIER_KINDS[1:]:
        checks={};scores=[]
        for case in ('T10','m10'):
            a=data[(data.case==case)&(data.carrier_kind==kind)].set_index('seed') if len(data) else pd.DataFrame()
            b=data[(data.case==case)&(data.carrier_kind=='original')].set_index('seed') if len(data) else pd.DataFrame()
            complete=len(a)==3 and len(b)==3
            if complete:
                a,b=(v.loc[list(EXPLORATION_SEEDS)] for v in (a,b))
                checks[case]=dict(complete=True,
                    phase_mean=float(a.initial_weighted_unwrapped_phase_rms.mean()),
                    original_phase_mean=float(b.initial_weighted_unwrapped_phase_rms.mean()),
                    phase_paired_improvements=int((a.initial_weighted_unwrapped_phase_rms<b.initial_weighted_unwrapped_phase_rms).sum()),
                    full_residual_mean=float(a.carrier_free_pde_residual_rms.mean()),
                    original_full_residual_mean=float(b.carrier_free_pde_residual_rms.mean()))
                if case=='T10':scores=[float(a.initial_weighted_unwrapped_phase_rms.mean()),float(a.initial_weighted_unit_spinor_residual_rms.mean())]
            else:checks[case]=dict(complete=False)
        eligible=all(r.get('complete') and r['phase_mean']<r['original_phase_mean'] and
            r['full_residual_mean']<=r['original_full_residual_mean'] for r in checks.values()) and checks['T10'].get('phase_paired_improvements',0)>=2
        records.append(dict(kind=kind,eligible=bool(eligible),per_case=checks,rank_scores=scores or [None,None]))
    candidates=[r for r in records if r['eligible']] or records
    selected=min(candidates,key=lambda r:tuple(math.inf if x is None else x for x in r['rank_scores'])+(CARRIER_KINDS.index(r['kind']),))
    return _write_once('carrier_selection.json',dict(kind=selected['kind'],eligible=selected['eligible'],
        diagnostic_fallback=not selected['eligible'],candidates=records,exploration_only=True,
        fixed_endpoint=6000,no_validation_seed_used=True,evidence_sha256=evidence(paths)))

def dynamic_score(means):
    if any(means[k] is None for k in THRESHOLDS):return None
    return means['spinor_rel_l2_gauge_aligned']/.10+means['current_rel_l2']/.15+means['modal_population_rel_l2']/.10+(1-means['fidelity_mean'])/.02

def select_envelope():
    paths=[run_directory('exact_envelopes',c,e,s) for c in ('T10','m10') for e in ENVELOPE_KINDS for s in EXPLORATION_SEEDS]
    data=rows(paths);records=[]
    for kind in ENVELOPE_KINDS[1:]:
        checks={}
        for case in ('T10','m10'):
            frame=data[(data.case==case)&(data.envelope_kind==kind)] if len(data) else pd.DataFrame()
            checks[case]=absolute_success(frame,seeds=EXPLORATION_SEEDS,required_joint=2)
        base=data[(data.case=='T10')&(data.envelope_kind=='pde_only')] if len(data) else pd.DataFrame()
        norm_better=bool(len(base)==3 and checks['T10']['means']['max_norm_drift'] is not None and
            checks['T10']['means']['max_norm_drift']<float(base.max_norm_drift.mean()))
        eligible=all(r['pass_check'] for r in checks.values()) and norm_better
        records.append(dict(kind=kind,eligible=eligible,per_case=checks,
            norm_mean_improves_T10=norm_better,rank_scores=[dynamic_score(checks['T10']['means']),checks['T10']['means']['max_norm_drift']]))
    candidates=[r for r in records if r['eligible']] or records
    selected=min(candidates,key=lambda r:tuple(math.inf if x is None else x for x in r['rank_scores'])+(ENVELOPE_KINDS.index(r['kind']),))
    return _write_once('envelope_selection.json',dict(kind=selected['kind'],eligible=selected['eligible'],
        diagnostic_fallback=not selected['eligible'],candidates=records,exploration_only=True,
        fixed_endpoint=20000,no_validation_seed_used=True,evidence_sha256=evidence(paths)))

def select_combination():
    carrier=json.loads((RAW/'carrier_selection.json').read_text(encoding='utf-8'))
    envelope=json.loads((RAW/'envelope_selection.json').read_text(encoding='utf-8'))
    paths=[run_directory('T10_factorial','T10','combined',s) for s in EXPLORATION_SEEDS]
    check=absolute_success(rows(paths),seeds=EXPLORATION_SEEDS,required_joint=2)
    selected=bool(carrier['eligible'] and envelope['eligible'] and check['pass_check'])
    hashes=evidence(paths)|{(RAW/n).relative_to(ROOT).as_posix():sha256(RAW/n) for n in ('carrier_selection.json','envelope_selection.json')}
    return _write_once('combination_selection.json',dict(selected=selected,carrier_kind=carrier['kind'],envelope_kind=envelope['kind'],
        independent_combination_count=1 if selected else 0,exploratory_absolute_check=check,
        component_gates=dict(carrier=carrier['eligible'],envelope=envelope['eligible']),
        no_validation_seed_used=True,no_alternative_combination=True,evidence_sha256=hashes))
