"""Evaluate saved untrained carrier rates with NumPy; no trainer/model creation."""
from __future__ import annotations
import json,time
import numpy as np
import torch
from plan import BASE,CORE,SEEDS,write,sha


def mlp(state,x):
    indices=sorted(int(k.split('.')[2]) for k in state if k.startswith('mlp.net.') and k.endswith('.weight'))
    for j,i in enumerate(indices):
        w=state[f'mlp.net.{i}.weight'].numpy();b=state[f'mlp.net.{i}.bias'].numpy()
        x=x@w.T+b
        if j<len(indices)-1:x=np.tanh(x)
    return x


def main():
    rows=[];missing=[];errors=[];joint_missing=[]
    for case in CORE:
        duration=10. if case=='T10' else 5.
        for seed in SEEDS:
            root=BASE/'formal/carriers'/case;op=root/'original'/f'seed_{seed}'/'initial_state.pt'
            sp=root/'static_frequency'/f'seed_{seed}'/'initial_state.pt'
            if not op.exists() or not sp.exists():missing.append(dict(case=case,seed=seed));continue
            original=torch.load(op,weights_only=False)['state_dict'];static=torch.load(sp,weights_only=False)['state_dict']
            cfg=json.loads((op.parent/'config.json').read_text(encoding='utf-8'));bounds=cfg['bounds']
            k=np.arange(bounds[0],bounds[1]+1,dtype=float)/8;x=((k-2)/10)[:,None]
            r0=mlp(original,np.concatenate((x,np.zeros_like(x)),axis=1))/duration
            r1=-mlp(static,x);difference=float(np.max(abs(r0-r1)))
            checks=dict(initial_rates_match=difference<2e-12,
                original_has_two_inputs=original['mlp.net.0.weight'].shape[1]==2,
                static_has_one_input=static['mlp.net.0.weight'].shape[1]==1)
            joint=BASE/'formal/coupled'/case/'joint_hard'/f'seed_{seed}'/'initial_state.pt'
            joint_sha=None
            if joint.exists():
                s=torch.load(joint,weights_only=False)['state_dict'];keys={k[8:]:v for k,v in s.items() if k.startswith('carrier.')}
                checks['joint_starts_from_same_untrained_static_weights']=set(keys)==set(static) and all(
                    tuple(keys[k].shape)==tuple(v.shape) and keys[k].dtype==v.dtype and keys[k].contiguous().numpy().tobytes()==v.contiguous().numpy().tobytes()
                    for k,v in static.items())
                joint_sha=sha(joint)
            else:joint_missing.append(dict(case=case,seed=seed))
            row=dict(case=case,seed=seed,original_initial_state=op.relative_to(BASE).as_posix(),original_sha256=sha(op),
                static_initial_state=sp.relative_to(BASE).as_posix(),static_sha256=sha(sp),joint_initial_state_sha256=joint_sha,
                original_static_initial_rate_max_abs_difference=difference,zero_time_phases_both_zero_by_frozen_ansatz=True,
                no_eigenfrequency_labels_or_future_trajectory_loaded=True,checks=checks,pass_check=all(checks.values()))
            rows.append(row)
            if not row['pass_check']:errors.append(row)
    full=len(rows)==15 and not missing and not joint_missing and not errors
    report=dict(updated_utc=time.time(),status='FAIL' if errors else ('ALL_INITIAL_CARRIER_CONTROLS_VERIFIED' if full else 'PASS_AVAILABLE_INITIAL_CARRIER_CONTROLS'),
        prescribed_pairs=15,actual_carrier_pairs_checked=len(rows),actual_joint_initializers_checked=sum(r['joint_initial_state_sha256'] is not None for r in rows),
        missing_carrier_pairs=missing,missing_joint_initializers=joint_missing,errors=errors,checks=rows,
        independently_evaluated_saved_weights_with_NumPy=True,source_code_sha256=sha(__file__),
        all_initial_carrier_controls_verified=full,original_weights_not_written=True,whole_goal_complete=False)
    write(BASE/'analysis_outputs/carrier_initialization_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','errors','missing_carrier_pairs','missing_joint_initializers')},ensure_ascii=False))
    if errors:raise RuntimeError('Initial carrier rate or joint-start control differs')


if __name__=='__main__':
    import psutil
    from threadpoolctl import threadpool_limits
    psutil.Process().cpu_affinity([6]);torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
