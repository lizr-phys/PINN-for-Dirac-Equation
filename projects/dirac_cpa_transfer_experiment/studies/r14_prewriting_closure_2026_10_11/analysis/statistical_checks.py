"""Acceptance-relevant statistical checks: missing/failure handling and small-n limits."""
import numpy as np
import pandas as pd
from plan import BASE,SEEDS,THRESHOLDS,write
from .aggregate import paired_frame,summaries_with_missing

def main():
    metric='spinor_rel_l2_gauge_aligned'
    a=pd.DataFrame({'seed':SEEDS,metric:[.04]*5});b=pd.DataFrame({'seed':SEEDS,metric:[.05]*5})
    s,_=paired_frame(a,b,'T10','learned_hard','analytic_hard',20000,metric)
    assert abs(s['exact_sign_flip_two_sided_p']-.0625)<1e-15
    assert s['CI95_upper']<0 and s['favorable_pairs']==5
    assert paired_frame(a.iloc[:4],b,'T10','learned_hard','analytic_hard',20000,metric)[0] is None
    values=dict(zip(THRESHOLDS,(.02,.02,.999,.02,1e-15)))
    rows=[dict(case='T10',group='learned_hard',step=20000,seed=seed,path=f'metric_{seed}',metric_sha256=f'sha_{seed}',
        seed_joint_pass=True,**values) for seed in SEEDS]
    d=pd.DataFrame(rows);audits={r['path']:r['metric_sha256'] for r in rows}
    jobs=[dict(type='coupled',stage='core',case='T10',group='learned_hard',seed=seed,budget=20000,snapshots=[20000]) for seed in SEEDS]
    assert summaries_with_missing(d,jobs,[],audits).iloc[0]['status']=='PASS'
    missing=summaries_with_missing(d.iloc[:4],jobs,[],audits).iloc[0]
    assert missing['status']=='INCOMPLETE' and pd.isna(missing[metric+'_mean'])
    failure=dict(job=dict(case='T10',group='learned_hard',seed=814),status='NUMERICAL_FAILURE',occurred_step=18000)
    failed=summaries_with_missing(d.iloc[:4],jobs,[failure],audits).iloc[0]
    assert failed['status']=='FAIL' and pd.isna(failed[metric+'_mean']) and failed['failed_seeds']=='814'
    failures=[dict(failure,job=dict(failure['job'],seed=seed)) for seed in SEEDS]
    allfailed=summaries_with_missing(d.iloc[:0],jobs,failures,audits).iloc[0]
    assert allfailed['status']=='FAIL' and allfailed['n']==0 and allfailed['numerical_or_dependency_failures']==5
    threepass=d.copy();threepass.loc[3:,'current_rel_l2']=.16;threepass.loc[3:,'seed_joint_pass']=False
    r=summaries_with_missing(threepass,jobs,[],audits).iloc[0]
    assert r['current_rel_l2_mean']<.15 and r['joint_pass_count']==3 and r['status']=='FAIL'
    latefailure=dict(failure,occurred_step=22000);extended=[dict(j,budget=26000,snapshots=[20000,26000]) for j in jobs]
    late=summaries_with_missing(d,extended,[latefailure],audits)
    assert late[late.step==20000].iloc[0]['status']=='PASS' and late[late.step==26000].iloc[0]['status']=='FAIL'
    write(BASE/'engineering/statistical_acceptance_checks.json',dict(status='PASS',
        n5_two_sided_minimum_sign_flip_p=.0625,incomplete_pairs_do_not_generate_statistics=True,
        missing_seeds_do_not_generate_five_seed_means=True,numerical_failures_remain_in_denominator=True,
        all_failed_groups_remain_visible=True,three_of_five_cannot_pass_despite_acceptable_means=True,
        later_terminal_failure_does_not_rewrite_an_earlier_verified_primary_endpoint=True))
    print('STATISTICAL_ACCEPTANCE_CHECKS_PASS')

if __name__=='__main__':main()
