from __future__ import annotations
import numpy as np
import pandas as pd
from .r12_config import CASES,GROUPS,SEEDS,THRESHOLDS,EXPERIMENT

def absolute_success(frame):
    complete=len(frame)==5 and sorted(frame.seed.tolist())==list(SEEDS)
    means={key:float(frame[key].mean()) for key in THRESHOLDS}
    checks={key:means[key]>=bound if key=='fidelity_mean' else means[key]<=bound for key,bound in THRESHOLDS.items()}
    joint=np.ones(len(frame),dtype=bool)
    for key,bound in THRESHOLDS.items():
        values=frame[key].to_numpy();joint &= np.isfinite(values)&(values>=bound if key=='fidelity_mean' else values<=bound)
    return dict(complete=complete,means=means,mean_checks=checks,joint_pass_count=int(joint.sum()),
        joint_pass_seeds=frame.seed.to_numpy()[joint].tolist(),pass_check=bool(complete and all(checks.values()) and joint.sum()>=4))

def decision(frame):
    records={c:{e:{g:absolute_success(frame[(frame.case==c)&(frame.endpoint==e)&(frame.model==g)])
        for g in GROUPS} for e in ('final','coupled_20000')} for c in CASES}
    return dict(experiment=EXPERIMENT,status='ALL_FIXED_RUNS_EVALUATED',per_case=records,
        learned_success_excludes_E=True,primary_groups=GROUPS[:3],thresholds=THRESHOLDS,
        population_sd_ddof=0,all_conditions_retrained=True,zero_shot=False,
        cross_case_success_averaging=False,posthoc_advantage_threshold=False,selective_budget_extension=False,
        scope='four specified conditions, known finite Fourier Gaussian initial spectrum')

def comparisons(frame):
    rows=[];summary=[]
    keys=tuple(THRESHOLDS)+('density_rel_l2','active_coherence_rel_l2','branch_complex_coefficient_rel_l2',
        'residual_integral_fine','predicted_max_occupation_outside_initial_support','plus_center_rmse','minus_center_rmse')
    for case in CASES:
        for endpoint in ('final','coupled_20000'):
            data=frame[(frame.case==case)&(frame.endpoint==endpoint)]
            for ai,bi in ((0,1),(0,2),(1,2),(3,0),(3,1),(3,2)):
                a,b=(data[data.model==GROUPS[i]].set_index('seed').loc[list(SEEDS)] for i in (ai,bi))
                for key in keys:
                    delta=a[key]-b[key];ratio=a[key]/b[key]
                    for seed in SEEDS:rows.append(dict(case=case,endpoint=endpoint,target=GROUPS[ai],baseline=GROUPS[bi],
                        metric=key,seed=seed,paired_difference=float(delta[seed]),actual_ratio=float(ratio[seed])))
                    summary.append(dict(case=case,endpoint=endpoint,target=GROUPS[ai],baseline=GROUPS[bi],metric=key,n=5,
                        paired_difference_mean=float(delta.mean()),paired_difference_population_sd=float(delta.std(ddof=0)),
                        ratio_of_means=float(a[key].mean()/b[key].mean()),mean_of_paired_ratios=float(ratio.mean()),
                        favorable_pair_count=int((delta>0 if key=='fidelity_mean' else delta<0).sum())))
    return pd.DataFrame(rows),pd.DataFrame(summary)
