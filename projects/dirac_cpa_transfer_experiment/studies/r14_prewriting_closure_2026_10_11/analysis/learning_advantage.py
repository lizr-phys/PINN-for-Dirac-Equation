"""Apply the original learning-advantage gate without seed or checkpoint selection."""
import json,time
import pandas as pd
from plan import BASE,write

def main():
    out=BASE/'analysis_outputs';summary=pd.read_csv(out/'summary_metrics.csv');paired=pd.read_csv(out/'paired_statistics.csv');rows=[]
    attempts=json.loads((out/'attempt_completion_audit.json').read_text(encoding='utf-8'))['rows']
    for case in ('T10','m10'):
        comparison=paired[(paired.case==case)&(paired.target=='learned_hard')&(paired.baseline=='analytic_hard')&
            (paired.target_step==20000)&(paired.baseline_step==20000)]
        spinor=comparison[comparison.metric=='spinor_rel_l2_gauge_aligned'];cost=comparison[comparison.metric=='attributed_training_wall_seconds']
        absolute=summary[(summary.case==case)&(summary.group=='learned_hard')&(summary.step==20000)]
        baseline=summary[(summary.case==case)&(summary.group=='analytic_hard')&(summary.step==20000)]
        ready=len(spinor)==1 and spinor.iloc[0]['status']=='ALL_FIVE_FINITE_PAIRS' and len(cost)==1 and cost.iloc[0]['status']=='ALL_FIVE_FINITE_PAIRS'
        ready=ready and absolute.iloc[0]['status'] in ('PASS','FAIL') and baseline.iloc[0]['status'] in ('PASS','FAIL')
        cohort=[a for a in attempts if a.get('case')==case and a.get('group') in ('learned_hard','analytic_hard')]
        accounted=len(cohort)==10 and all(a['state'] in ('COMPLETE_VERIFIED','FAILURE_PRESERVED') for a in cohort)
        r=dict(case=case,primary_step=20000,status='INCOMPLETE',all_five_paired_and_independently_audited=bool(ready),prescribed_primary_attempts_accounted=accounted,
            accuracy_advantage_qualified=None,efficiency_claim_qualified=None)
        if ready:
            s,c=spinor.iloc[0],cost.iloc[0]
            gate=bool(s['paired_difference_mean']<0 and s['CI95_upper']<0 and s['favorable_pairs']>=4 and absolute.iloc[0]['absolute_success'])
            r.update(status='QUALIFIED_IN_THIS_CASE' if gate else 'NOT_QUALIFIED_IN_THIS_CASE',accuracy_advantage_qualified=gate,
                efficiency_claim_qualified=bool(gate and c['paired_difference_mean']<0),
                spinor_paired_difference_mean=float(s['paired_difference_mean']),spinor_CI95_upper=float(s['CI95_upper']),
                lower_error_pairs=int(s['favorable_pairs']),complete_auxiliary_cost_included=True,
                attributed_training_time_difference_mean=float(c['paired_difference_mean']))
        elif accounted and any(a['state']=='FAILURE_PRESERVED' for a in cohort):
            r.update(status='NOT_ESTABLISHED_WITH_PRESERVED_SCIENTIFIC_FAILURES',accuracy_advantage_qualified=False,
                efficiency_claim_qualified=False,paired_difference_not_estimated_from_surviving_subset=True)
        rows.append(r)
    complete=all(r['prescribed_primary_attempts_accounted'] and r['accuracy_advantage_qualified'] is not None for r in rows)
    accuracy=all(r['accuracy_advantage_qualified'] for r in rows) if complete else None
    efficiency=all(r['efficiency_claim_qualified'] for r in rows) if complete else None
    report=dict(updated_utc=time.time(),status='ASSESSED_FIXED_PRIMARY_COMPARISONS' if complete else 'INCOMPLETE',
        both_T10_and_m10_accuracy_gate_qualified=accuracy,both_T10_and_m10_efficiency_gate_qualified=efficiency,cases=rows,
        n5_exact_two_sided_minimum_p=.0625,descriptive_t_interval_assumptions_required=True,
        no_cross_case_pooling=True,no_accuracy_driven_extra_budget=True,entire_study_complete=False,
        paper_positioning_waits_for_all_scope_and_baseline_evidence=True)
    write(out/'learning_advantage_assessment.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
