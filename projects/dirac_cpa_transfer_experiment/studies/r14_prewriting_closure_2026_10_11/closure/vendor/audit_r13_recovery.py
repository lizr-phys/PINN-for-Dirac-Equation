from __future__ import annotations

import json
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd
import torch
from .r13_config import ROOT,RAW,FIGURES,DOCS,ARTIFACTS,configuration
from .r13_plan import STAGES,jobs_for,directory_for
from .freeze_r13_recovery import verify_freeze,verify_prior_results
from .freeze_r7_phase_release import write_json,sha256

def audit():
    checks=[]
    def check(name,passed,**detail):checks.append(dict(name=name,pass_check=bool(passed),**detail))
    frozen=verify_freeze();check('frozen_scientific_code',True,sha256=frozen)
    history=verify_prior_results();check('all_prior_history_unchanged',history['status']=='PASS',**history)
    attempts=json.loads((RAW/'attempt_index.json').read_text(encoding='utf-8'))['attempts']
    streams={};sources={}
    for attempt in attempts:
        directory=ROOT/attempt['directory'];tag=attempt['directory']
        if attempt['status']!='COMPLETE':
            check(tag+'/preserved_failed_or_incomplete',attempt['status'] in ('NUMERICAL_FAILURE','DEPENDENCY_FAILED'))
            continue
        metrics=json.loads((directory/'metrics.json').read_text(encoding='utf-8'));cfg=configuration(attempt['case'])
        check(tag+'/six_and_recovery',all((directory/n).exists() for n in ARTIFACTS+('recovery_state.pt','execution_cost.json')))
        check(tag+'/fixed_budget',metrics['executed_steps']==(6000 if attempt['type']=='carrier' else 20000))
        check(tag+'/precision_environment',json.loads((directory/'config.json').read_text(encoding='utf-8'))['environment']['default_dtype']=='torch.float64')
        state=torch.load(directory/'model_state.pt',weights_only=False)
        recovery=torch.load(directory/'recovery_state.pt',weights_only=False)
        check(tag+'/checkpoint_equals_recovery',all(torch.equal(state['state_dict'][k],recovery['state_dict'][k]) for k in state['state_dict']))
        check(tag+'/no_future_labels_in_checkpoint',not any(any(w in k for w in ('target','reference','eigenvalues')) for k in state['state_dict']))
        check(tag+'/sampling_length',len(recovery['sampler']['trace'])==metrics['executed_steps'])
        check(tag+'/sampler_prefix',metrics['normalized_time_stream_sha256']==state['normalized_time_stream_sha256'])
        check(tag+'/sampling_rng_unchanged_by_evaluation',metrics['evaluation_does_not_change_training_rng'])
        check(tag+'/full_cost_includes_training',metrics['full_attributed_pipeline_seconds']>=metrics['training_seconds'])
        check(tag+'/double_precision_checkpoint',all(v.dtype in (torch.float64,torch.complex128) for v in state['state_dict'].values()))
        if attempt['type']=='carrier':
            sources[tag]=state;continue
        streams.setdefault((attempt['case'],attempt['seed']),set()).add(metrics['normalized_time_stream_sha256'])
        pre=6000 if attempt['carrier_kind']!='exact' else 0
        check(tag+'/residual_work',(2*pre+20000)*cfg.spectral_collocation_points==metrics['training_spinor_residual_points'])
        check(tag+'/active_parameters',metrics['active_parameters']==12932)
        expected_total=12932 if attempt['carrier_kind']=='exact' else (25670 if attempt['carrier_kind']=='static_frequency' else 25734)
        check(tag+'/total_parameters',metrics['total_parameters']==expected_total)
        if pre:
            source=ROOT/attempt['pretrain_source'];source_state=torch.load(source/'model_state.pt',weights_only=False)
            check(tag+'/frozen_carrier_unchanged',all(torch.equal(state['state_dict']['carrier.'+k],v) for k,v in source_state['state_dict'].items()))
            check(tag+'/source_hash',metrics['pretrain_source_sha256']==sha256(source/'model_state.pt'))
        with np.load(directory/'dense_grid.npz') as dense:
            raw,ref=dense['spectral_predicted_raw'],dense['spectral_reference']
            norm=np.sum(abs(raw)**2,(1,2));phase=float(np.angle(np.vdot(ref.ravel(),raw.ravel())))
            spinor=float(np.linalg.norm(raw*np.exp(-1j*phase)-ref)/np.linalg.norm(ref))
            current=float(np.linalg.norm(dense['current_predicted']-dense['current_reference'])/np.linalg.norm(dense['current_reference']))
            check(tag+'/spinor_recomputed',abs(spinor-metrics['spinor_rel_l2_gauge_aligned'])<2e-13)
            check(tag+'/current_recomputed',abs(current-metrics['current_rel_l2'])<2e-13)
            check(tag+'/norm_recomputed',abs(float(abs(norm-1).max())-metrics['max_norm_drift'])<2e-13)
            check(tag+'/reference_coherence_mask',np.array_equal(dense['phase_reference_active_mask'],dense['branch_occupations_reference']>=1e-6))
            check(tag+'/signed_exterior',np.array_equal(dense['exterior_residual_complex'],-dense['exterior_positive_coupling_source_complex']))
            check(tag+'/residual_archive',dense['residual_complex_unweighted'].shape==(cfg.residual_time_points,161,2))
            check(tag+'/no_false_continuous_certificate',not metrics['residual_numerical_estimate_is_continuous_certificate'])
            check(tag+'/hard_initial',metrics['initial_condition_max_abs_error']<1e-14)
            check(tag+'/no_aliasing',metrics['fourier_integral_no_aliasing'] and max(metrics['density_quadrature_double_grid_difference'],metrics['current_quadrature_double_grid_difference'])<1e-12)
            if attempt['envelope_kind']=='hard_total_norm':check(tag+'/structural_norm',metrics['max_norm_drift']<1e-13)
    for key,digests in streams.items():check('paired_full_coupled_stream/'+str(key),len(digests)==1)
    runtime=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    for batch in runtime['batches']:
        masks=[s['mask'] for s in batch['slots']]+[runtime['control_mask']]
        check('disjoint_cpu_masks/'+batch['stage'],all(not(a&b) for i,a in enumerate(masks) for b in masks[i+1:]))
        for slot in batch['slots']:
            result=json.loads((ROOT/'results/logs/r13_long_time_recovery'/f"{batch['stage']}_seed{slot['seed']}_worker_result.json").read_text(encoding='utf-8'))
            check('actual_cpu_mask/'+batch['stage']+'/'+str(slot['seed']),result['actual_affinity_mask']==slot['mask'])
    selection=json.loads((RAW/'combination_selection.json').read_text(encoding='utf-8'))
    check('at_most_one_combination',selection['independent_combination_count']<=1 and selection['no_validation_seed_used'])
    for name in ('carrier_selection.json','envelope_selection.json','combination_selection.json'):
        record=json.loads((RAW/name).read_text(encoding='utf-8'))
        check('selection_evidence_hashes/'+name,all(sha256(ROOT/p)==d for p,d in record['evidence_sha256'].items()))
    links=[]
    for path in [ROOT/'R13_LONG_TIME_RECOVERY_REPORT.md']+list(DOCS.glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            target=target.strip('<>').split('#')[0]
            if re.match(r'^[a-z]+://',target):continue
            resolved=path.parent/unquote(target)
            links.append(dict(source=path.relative_to(ROOT).as_posix(),target=target,exists=resolved.exists()))
    check('local_links',all(row['exists'] for row in links),count=len(links));write_json(RAW/'link_check.json',dict(status='PASS' if all(r['exists'] for r in links) else 'FAIL',links=links))
    figure_qa=json.loads((FIGURES/'FIGURE_QA.json').read_text(encoding='utf-8'))
    check('automated_figure_checks',figure_qa['status'] in ('PASS','AUTOMATED_PASS_VISUAL_REVIEW_PENDING'))
    result=dict(status='PASS' if all(r['pass_check'] for r in checks) else 'FAIL',checks=checks,
        checked_count=len(checks),failed_count=sum(not r['pass_check'] for r in checks),figure_visual_status=figure_qa['status'])
    write_json(RAW/'audit.json',result)
    return result

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
