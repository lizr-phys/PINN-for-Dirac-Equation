"""Archive the first complete T10 primary comparison; no training or selection."""
from __future__ import annotations
import csv,json,math,shutil,time
from pathlib import Path
import psutil,torch
from plan import BASE,write,sha

def table(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))

def main():
    psutil.Process().cpu_affinity([6]);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    out=BASE/'analysis_outputs';design=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    stats=json.loads((out/'independent_statistical_audit.json').read_text(encoding='utf-8'))
    assert stats['status'] in ('PASS_AVAILABLE_STATISTICAL_TABLES','ALL_PRESCRIBED_STATISTICS_VERIFIED') and not stats['errors']
    for name,digest in stats['source_table_sha256'].items():assert sha(out/name)==digest,name
    assessment=json.loads((out/'learning_advantage_assessment.json').read_text(encoding='utf-8'))
    case=next(r for r in assessment['cases'] if r['case']=='T10')
    assert case['all_five_paired_and_independently_audited'] and case['prescribed_primary_attempts_accounted']
    pairing=json.loads((out/'pairing_audit.json').read_text(encoding='utf-8'));assert not pairing['mismatches']
    controls=[]
    for seed in design['formal_seeds']:
        states=[];traces=[];initial_sources=[]
        for group in ('learned_hard','analytic_hard'):
            folder=BASE/'formal/coupled/T10'/group/f'seed_{seed}'
            assert (folder/'complete.json').exists()
            initial=next(r for r in pairing['initializer_checks'] if (r['case'],r['group'],r['seed'])==('T10',group,seed))
            trace=next(r for r in pairing['stream_replay_checks'] if (r['case'],r['group'],r['seed'])==('T10',group,seed))
            assert initial['pass_check'] and trace['pass_check']
            assert sha(BASE/initial['path'])==initial['initial_file_sha256']
            assert sha(BASE/trace['checkpoint'])==trace['checkpoint_sha256'] and sha(BASE/trace['terminal_recovery'])==trace['recovery_sha256']
            raw=torch.load(BASE/initial['path'],weights_only=False)['state_dict']
            states.append({k:v for k,v in raw.items() if not k.startswith('carrier.')})
            traces.append(trace);initial_sources.append(initial)
        assert states[0].keys()==states[1].keys() and all(torch.equal(states[0][k],states[1][k]) and states[0][k].dtype==states[1][k].dtype for k in states[0])
        for key in ('actual_normalized_prefix_sha256','actual_physical_prefix_sha256'):
            assert traces[0][key]==traces[1][key]
        controls.append(dict(seed=seed,actual_noncarrier_weights_and_buffers_bitwise_equal=True,private_20000_time_streams_bitwise_equal=True,
            initial_sources=initial_sources,independent_replayed_trace_sources=traces))
    pairs=[r for r in table(out/'paired_statistics.csv') if (r['case'],r['target'],r['baseline'],r['target_step'],r['baseline_step'])==('T10','learned_hard','analytic_hard','20000','20000')]
    assert pairs and all(r['status']=='ALL_FIVE_FINITE_PAIRS' and int(r['n'])==5 for r in pairs)
    spinor=next(r for r in pairs if r['metric']=='spinor_rel_l2_gauge_aligned')
    mean=case['spinor_paired_difference_mean'];assert math.isclose(mean,float(spinor['paired_difference_mean']),abs_tol=2e-14)
    directory=out/'primary_comparison_receipts/T10_20000_first';assert not directory.exists(),'First receipt already exists; do not overwrite'
    directory.mkdir(parents=True)
    names=['independent_metric_audit.json','independent_metrics_per_snapshot.csv','independent_cost_audit.json',
        'independent_statistical_audit.json','paired_statistics.csv','paired_per_seed.csv','summary_metrics.csv',
        'pairing_audit.json','learning_advantage_assessment.json']
    preserved={}
    for name in names:
        target=directory/name;shutil.copyfile(out/name,target);preserved[name]=sha(target)
    write(directory/'receipt.json',dict(assessed_utc=time.time(),status='FIRST_FULL_T10_PRIMARY_COMPARISON_INDEPENDENTLY_VERIFIED',
        case='T10',coupled_steps=20000,formal_seeds=design['formal_seeds'],assessment=case,paired_rows=pairs,strict_pairing_controls=controls,
        preserved_source_sha256=preserved,design_sha256=sha(BASE/'frozen/design.json'),archive_source=Path(__file__).relative_to(BASE).as_posix(),archive_source_sha256=sha(__file__),
        optimization_cost_includes_full_6000_auxiliary_steps=True,optimization_wall_excludes_setup_evaluation_and_IO=True,
        native_inference_and_RSS_comparison_rows_are_not_isolated_inference_measurements=True,
        exact_sign_flip_two_sided_p=float(spinor['exact_sign_flip_two_sided_p']),n5_t_interval_descriptive_not_general_significance=True,
        m10_comparison_pending=True,all_scope_and_strong_baseline_evidence_pending=True,whole_goal_complete=False))
    print(json.dumps(dict(receipt=(directory/'receipt.json').relative_to(BASE).as_posix(),strict_five_pairs_checked=len(controls),assessment=case,whole_goal_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
