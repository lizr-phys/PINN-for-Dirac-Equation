from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r12_config import ROOT,RAW,DOCS,FIGURES,LOGS,CASES,GROUPS,SEEDS,ARTIFACTS,CURVE_STEPS,budget,configuration,run_directory,group_order
from .r12_models import R12Model,hashes,parameter_counts
from .r12_decision import decision
from .aggregate_r12_robustness import original_metrics
from .freeze_r12_robustness import verify_freeze,verify_prior_results
from .freeze_r7_phase_release import write_json,sha256
from .audit_r11_ablation import forbidden_keys,check_links

def audit(*,final=False):
    checks=[]
    def add(name,ok,**details):checks.append(dict(check=name,pass_check=bool(ok),**details))
    frozen=verify_freeze();add('science_freeze_unchanged',True,sha256=frozen)
    print('R12 AUDIT verifying protected historical hashes',flush=True)
    history=verify_prior_results();add('r1_through_r11_unchanged',history['status']=='PASS',**history)
    plan=json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    masks=plan['masks']+[plan['control_mask']]
    add('physical_cpu_allocation_disjoint',len(masks)==6 and all(not(a&b) for i,a in enumerate(masks) for b in masks[i+1:]),masks=masks)
    add('all80_formal_attempts_completed',plan['status']=='COMPLETE' and len(plan['batches'])==4 and not plan['failures'])
    add('batch_and_group_order_preregistered',[b['case'] for b in plan['batches']]==list(CASES) and all(
        s['group_order']==list(group_order(b['case'],s['seed'])) for b in plan['batches'] for s in b['slots']))
    observed=json.loads((RAW/'initial_runtime_affinity_observation.json').read_text(encoding='utf-8'))
    add('initial_live_OS_affinity_verified',observed['status']=='PASS' and len(observed['workers'])==5)
    frame=original_metrics();recorded=json.loads((RAW/'decision.json').read_text(encoding='utf-8'))
    # Compare JSON representations: tuples are serialized as lists, with no
    # tolerance or alteration of numeric decisions and success thresholds.
    canonical=lambda value:json.loads(json.dumps(value,allow_nan=False))
    add('decision_recomputed_from_original_JSON',recorded==canonical(decision(frame)))
    csv=pd.read_csv(RAW/'per_seed_metrics.csv',float_precision='round_trip')
    add('CSV_roundtrip_decision',canonical(decision(csv))==recorded)
    add('160_endpoint_rows140_unique_snapshots',len(frame)==160 and frame.snapshot_identity.nunique()==140)
    curves=pd.read_csv(RAW/'error_cost_curves.csv',float_precision='round_trip')
    index=pd.read_csv(RAW/'snapshot_index.csv');costs=pd.read_csv(RAW/'cost_and_residual_work.csv',float_precision='round_trip')
    add('380_cost_points140_full240_light',len(curves)==380 and len(index)==380 and index.full_residual.sum()==140)
    cross_case_initial={};cross_case_uniforms={}
    optimizer_steps=0;residual_points=0
    for case in CASES:
        print('R12 AUDIT checking',case,'fixed runs and snapshots',flush=True)
        cfg=configuration(case);batch=next(b for b in plan['batches'] if b['case']==case)
        slots={s['seed']:s for s in batch['slots']}
        ref_record=json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
        row=next(r for r in ref_record['comparisons'] if r['case']==case and r['smaller']==ref_record['selected_common_truncation'])
        add(case+'/independent_reference_converged',row['pass_check'] and row['hermiticity_max_abs']==0)
        for seed in SEEDS:
            initials={};traces={}
            for group in GROUPS:
                prefix=f'{case}/{group}/seed{seed}';directory=run_directory(case,group,seed);pre,coupled=budget(group)
                m=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
                c=json.loads((directory/'config.json').read_text(encoding='utf-8'))
                state=torch.load(directory/'model_state.pt',weights_only=False)
                rec=torch.load(directory/'recovery_state.pt',weights_only=False)
                initial=torch.load(directory/'initial_state.pt',weights_only=False)['state_dict'];initials[group]=initial
                add(prefix+'/six_pack_restore_complete',all((directory/p).is_file() for p in ARTIFACTS+('recovery_state.pt',)))
                add(prefix+'/fixed_budget_Adam_steps',state['steps']==m['steps']==rec['step']==26000 and
                    {int(v['step']) for v in rec['optimizer']['state'].values()}=={coupled})
                add(prefix+'/checkpoint_restore_equal',all(torch.equal(v,rec['state_dict'][k]) for k,v in state['state_dict'].items()))
                add(prefix+'/no_target_or_reference_keys',not forbidden_keys(state))
                add(prefix+'/frozen_protocol_affinity_threads',c['freeze_manifest_sha256']==frozen and
                    int(c['cpu_affinity_mask'])==slots[seed]['mask'] and c['environment']['torch_num_threads']==1)
                worker=json.loads((LOGS/f'{case}_seed{seed}_worker_result.json').read_text(encoding='utf-8'))
                add(prefix+'/observed_worker_affinity',worker['status']=='PASS' and worker['actual_affinity_mask']==slots[seed]['mask'])
                free=rec['free_sampler'];stream=rec['coupled_sampler'];traces[group]=stream['normalized_trace']
                add(prefix+'/private_streams_and_exact_physical_mapping',tuple(free['trace'].shape)==(pre,32,1) and
                    tuple(stream['trace'].shape)==(coupled,32,1) and torch.equal(stream['trace'],cfg.t_max*stream['normalized_trace']))
                model=R12Model(cfg,group,seed);model.load_state_dict(state['state_dict'])
                if group==GROUPS[0]:model.carrier.freeze()
                add(prefix+'/parameter_counts',all(m[k]==v for k,v in parameter_counts(model).items()))
                add(prefix+'/hard_initial_and_initialization',m['initial_condition_max_abs_error']<1e-14 and c['fresh_initialization'] and seed in SEEDS)
                if group==GROUPS[0]:
                    pretrained=torch.load(directory/'pretrained_carrier_state.pt',weights_only=False)['state_dict']
                    add(prefix+'/carrier_frozen_preserved',all(torch.equal(v,pretrained[k]) for k,v in model.carrier.state_dict().items()))
                if group==GROUPS[1]:
                    add(prefix+'/carrier_and_envelope_updated',any(not torch.equal(initial[k],state['state_dict'][k]) for k in initial if k.startswith('carrier.')) and
                        any(not torch.equal(initial[k],state['state_dict'][k]) for k in initial if k.startswith('networks.')))
                add(prefix+'/analytic_phase_onlyE',(model.carrier is None or bool(getattr(model.carrier,'analytic_prior',False))==(group==GROUPS[-1])) and m['analytic_prior']==(group==GROUPS[-1]))
                points=(2*pre+coupled)*cfg.spectral_collocation_points;residual_points+=points;optimizer_steps+=pre+coupled
                cost=costs[(costs.case==case)&(costs.model==group)&(costs.seed==seed)].iloc[0]
                add(prefix+'/total_cost_and_work',m['training_spinor_residual_points']==points==cost.training_spinor_residual_points and
                    cost.training_seconds>0 and cost.evaluation_seconds>0 and cost.peak_rss_mib>0 and
                    abs(cost.training_seconds-cost.carrier_training_seconds-cost.coupled_training_seconds)<1e-8)
                snaps=sorted((directory/'snapshots').iterdir())
                expected=[s for s in CURVE_STEPS if s<=coupled]
                add(prefix+'/all_fixed_snapshots',[int(s.name.split('_')[1]) for s in snaps]==expected)
                for snap in snaps:
                    sm=json.loads((snap/'metrics.json').read_text(encoding='utf-8'))
                    add(prefix+'/'+snap.name+'/six_pack',all((snap/p).is_file() for p in ARTIFACTS))
                    if not sm['evaluation_full_residual']:continue
                    add(prefix+'/'+snap.name+'/restore', (snap/'recovery_state.pt').is_file())
                    with np.load(snap/'dense_grid.npz') as d:
                        active=d['phase_reference_active_mask'];power=d['branch_occupations_reference']
                        add(prefix+'/'+snap.name+'/reference_only_masks_and_finite',np.array_equal(active,power>=1e-6) and
                            all(np.isfinite(d[k]).all() for k in d.files))
                        add(prefix+'/'+snap.name+'/spectral_dimensions_and_signed_exterior',d['residual_complex_unweighted'].shape==(cfg.residual_time_points,cfg.mode_count,2) and
                            d['full_physical_residual_complex'].shape==(cfg.residual_time_points,cfg.mode_count+2*cfg.max_coupling_shift,2) and
                            np.array_equal(d['exterior_residual_complex'],-d['exterior_positive_coupling_source_complex']))
                        add(prefix+'/'+snap.name+'/original_support_and_coherence',np.array_equal(d['initial_support_mask'],(d['n']>=0)&(d['n']<=32)) and
                            np.array_equal(active.reshape(201,-1),d['coherence_reference_active_channel_mask']))
                    add(prefix+'/'+snap.name+'/quadrature_not_certificate',sm['fourier_integral_no_aliasing'] and
                        sm['fourier_norm_parseval_max_error']<1e-12 and sm['density_quadrature_double_grid_difference']<1e-12 and
                        sm['current_quadrature_double_grid_difference']<1e-12 and not sm['residual_numerical_estimate_is_continuous_certificate'])
                del model,rec,state
            add(f'{case}/seed{seed}/paired_initial_networks',all(torch.equal(v,initials[g][k])
                for k,v in initials[GROUPS[0]].items() if k.startswith('networks.0.') for g in GROUPS))
            add(f'{case}/seed{seed}/paired_carrier_AC',all(torch.equal(v,initials[GROUPS[1]][k])
                for k,v in initials[GROUPS[0]].items() if k.startswith('carrier.')))
            add(f'{case}/seed{seed}/paired_coupled_prefix',all(torch.equal(traces[GROUPS[0]],traces[g][:20000]) for g in GROUPS[1:]) and
                all(torch.equal(traces[GROUPS[1]],traces[g]) for g in GROUPS[2:]))
            if seed not in cross_case_initial:
                cross_case_initial[seed]=initials[GROUPS[0]];cross_case_uniforms[seed]=traces[GROUPS[0]]
            else:
                add(f'{case}/seed{seed}/cross_case_normalized_time_pair',torch.equal(traces[GROUPS[0]],cross_case_uniforms[seed]))
                add(f'{case}/seed{seed}/cross_case_network_pair',all(torch.equal(v,initials[GROUPS[0]][k])
                    for k,v in cross_case_initial[seed].items() if k.startswith(('networks.','carrier.'))))
    add('total2080000_steps11334400000_spinor_points',optimizer_steps==2080000 and residual_points==11334400000,
        optimizer_steps=optimizer_steps,spinor_residual_points=residual_points)
    if final:
        qa=json.loads((FIGURES/'FIGURE_QA.json').read_text(encoding='utf-8'))
        add('actual_visual_QA_all12_figures',qa['status']=='PASS' and not qa['visual_inspection_pending'] and len(qa['figures'])==12)
        for f in qa['figures']:
            add(f['figure']+'/all_exports_and_source_hashes',all((FIGURES/(f['figure']+'.'+ext)).is_file() for ext in ('png','svg','pdf')) and
                all(sha256(ROOT/s['path'])==s['sha256'] for s in f['source_data']))
        tests=json.loads((RAW/'pytest_final_result.json').read_text(encoding='utf-8'))
        add('complete_repository_pytest',tests['status']=='PASS' and not(tests['failures']+tests['errors']+tests['skipped']))
        links=check_links([ROOT/'R12_ROBUSTNESS_REPORT.md']+list(DOCS.glob('*.md'))+[FIGURES/'FIGURE_QA.md'])
        write_json(RAW/'link_check.json',links);add('all_report_links_exist',links['status']=='PASS',checked_links=links['checked_links'])
    result=dict(status='PASS' if all(c['pass_check'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),
        failed=[c for c in checks if not c['pass_check']],final=final)
    write_json(RAW/('audit.json' if final else 'audit_before_figures.json'),result)
    print('R12 AUDIT',result['status'],'checks=',len(checks),'failures=',len(result['failed']),flush=True)
    if result['status']!='PASS':raise RuntimeError(json.dumps(result['failed'],ensure_ascii=False))
    return result

def artifact_manifest():
    files={}
    for directory in (RAW,FIGURES,LOGS,DOCS):
        for p in directory.rglob('*'):
            if p.is_file() and p.name not in ('artifact_sha256.json','postprocessing_status.json'):
                files[p.relative_to(ROOT).as_posix()]=sha256(p)
    report=ROOT/'R12_ROBUSTNESS_REPORT.md';files[report.relative_to(ROOT).as_posix()]=sha256(report)
    write_json(RAW/'artifact_sha256.json',dict(files=files,all_result_data_preserved=True))
    return len(files)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    audit(final=args.final)
