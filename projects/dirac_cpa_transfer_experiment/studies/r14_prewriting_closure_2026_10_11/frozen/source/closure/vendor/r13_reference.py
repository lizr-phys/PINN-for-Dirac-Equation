from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pandas as pd
from .r13_config import ROOT, RAW, CASES, configuration, reference_policy
from .r12_reference import solve, compare_pair
from .freeze_r7_phase_release import write_json, sha256

def prepare():
    RAW.mkdir(parents=True,exist_ok=True)
    policy = RAW/'reference_selection_policy.json'
    if not policy.exists():
        write_json(policy,reference_policy())
    record = RAW/'reference_validation.json'
    if record.exists():
        return json.loads(record.read_text(encoding='utf-8'))
    rows=[]
    for case in CASES:
        cfg = configuration(case)
        times = np.linspace(0,cfg.t_max,cfg.residual_time_points)
        row, a, b = compare_pair(cfg,configuration(case,(-80,112)),times)
        rows.append(row)
        directory=RAW/'references'/case; directory.mkdir(parents=True,exist_ok=True)
        for data, bounds in ((a,(-64,96)),(b,(-80,112))):
            path=directory/f'reference_n{bounds[0]}_{bounds[1]}.npz'
            if not path.exists():
                np.savez_compressed(path,**data)
        print('REFERENCE',case,'difference',row['full_complex_zero_padded_relative_difference'],'pass',row['pass_check'],flush=True)
    result=dict(status='PASS' if all(r['pass_check'] for r in rows) else 'REFERENCE_CALIBRATION_FAILED',
        selected_common_truncation=[-64,96],comparisons=rows,before_learning=True,
        no_pinn_based_selection=True,original_gaussian_unchanged=True,no_renormalization=True,
        policy_sha256=sha256(policy))
    pd.DataFrame(rows).to_csv(RAW/'reference_convergence.csv',index=False)
    write_json(record,result)
    if result['status']!='PASS':
        return result
    from .r9_evaluate import field_metrics
    for case in CASES:
        cfg=configuration(case); directory=RAW/'references'/case
        fine=directory/'reference_n-64_96.npz'
        write_json(directory/'selected_reference.json',dict(fine_grid_path=fine.relative_to(ROOT).as_posix(),sha256=sha256(fine)))
        reference=solve(cfg,np.linspace(0,cfg.t_max,cfg.test_t_points))
        np.savez_compressed(directory/'reference_test_grid.npz',**reference)
        free=solve(replace(cfg,potential_harmonics=((1,0.),)),reference['t'])
        np.savez_compressed(directory/'diagnostic_ignoring_potential.npz',**free)
        metrics,_,_=field_metrics(free['state'],reference['state'],cfg,reference['t'])
        write_json(directory/'diagnostic_ignoring_potential.json',metrics|dict(diagnostic_only=True,no_learning_or_selection=True))
    return result
