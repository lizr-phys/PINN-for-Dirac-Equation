"""Higher-precision diagnostic for spectral benchmark floors; primary reference untouched."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from plan import BASE,CASES,write,sha
from closure.config import configuration
from closure.reference import compare,diagonalize,trajectory
from closure.spectral import metrics

def main():
    out=BASE/'spectral_reference_check';out.mkdir(exist_ok=True);rows=[];updated=[]
    for case in CASES:
        cfg=configuration(case,(-112,144));big=configuration(case,(-128,160));times=np.linspace(0,cfg.t_max,401)
        row,a,b,diag=compare(cfg,big,times)
        row['diagnostic_high_precision_trajectory_pass']=row['trajectory_rel_difference']<1e-10
        rows.append(row);np.savez_compressed(out/f'{case}_wide401.npz',t=times,n=np.array(big.indices),state=b)
        ref=b[::2];primary=BASE/'references'/case/'test201.npz'
        with np.load(primary) as f:coarse=f['state']
        padded=np.zeros_like(ref);offset=-64-big.n_min;padded[:,offset:offset+161]=coarse
        row['primary161_zero_padded_to_wide_relative_difference']=float(np.linalg.norm(padded-ref)/np.linalg.norm(ref))
        row['primary161_reference_unchanged_sha256']=sha(primary)
        print('SPECTRAL_WIDE_REF',case,row['trajectory_rel_difference'],flush=True)
        for record in sorted((BASE/'spectral_benchmark').glob(case+'_*.json')):
            r=json.loads(record.read_text(encoding='utf-8'));predpath=record.with_suffix('.npz')
            if not predpath.exists():continue
            with np.load(predpath) as f:pred=f['prediction'];n=f['n']
            # Common full spectral union, with enough quadrature for1024-FFT-span squared bilinears.
            union=np.arange(min(int(n.min()),big.n_min),max(int(n.max()),big.n_max)+1)
            pp=np.zeros((201,len(union),2),dtype=np.complex128);rr=np.zeros_like(pp)
            pp[:,n-union[0]]=pred;rr[:,np.array(big.indices)-union[0]]=ref
            met=metrics(pp,rr,big,union)
            updated.append(dict(case=case,grid=r['grid'],dt=r['dt'],source_record=str(record.relative_to(BASE)),
                cost_median_seconds=r['cost_median_seconds'],cpu_median_seconds=r['cpu_median_seconds'],
                primary161_spinor_error=r['spinor_rel_l2_gauge_aligned'],reference_wide_truncation='[-128,160]',
                wide_reference_pair_difference=row['trajectory_rel_difference'],**met))
    pd.DataFrame(rows).to_csv(out/'reference_convergence.csv',index=False)
    pd.DataFrame(updated).to_csv(out/'spectral_error_cost_wide_reference.csv',index=False)
    write(out/'audit.json',dict(status='PASS' if all(r['diagnostic_high_precision_trajectory_pass'] and r['pass'] for r in rows) else 'REFERENCE_PRECISION_LIMITATION',
        comparisons=rows,benchmark_records_recomputed=len(updated),primary_success_reference_unchanged=True,
        no_PINN_metric_used_for_bounds=True,fixed_widest_pair_from_initial_candidates=True,
        higher_precision_check_is_independent_post_calibration_diagnostic=True))

if __name__=='__main__':
    import torch,psutil
    torch.set_num_threads(1);psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
