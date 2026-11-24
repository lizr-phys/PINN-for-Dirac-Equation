from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from plan import BASE,write,sha,THRESHOLDS
from .independent_metrics import compute,grid_checks

def main():
    rows=[];missing=[];bad=[]
    for directory in (BASE/'formal/coupled',BASE/'development'):
        if not directory.exists():continue
        for p in sorted(directory.rglob('metrics.json')):
            target=p.parent;raw=target/'prediction.npz';conf=json.loads(p.read_text(encoding='utf-8'));job=conf['job']
            if not raw.exists():missing.append(str(raw.relative_to(BASE)));continue
            with np.load(raw) as f:pred=f['prediction'];n=f['n'];t=f['t'];pred401=f['prediction401'] if 'prediction401' in f else None;t401=f['t401'] if 't401' in f else None
            ref_path=BASE/'references'/job['case']/'test201.npz'
            with np.load(ref_path) as f:ref=f['state'];ref_t=f['t'];ref_n=f['n']
            checkpoint=target/'model_state.pt';provenance_pass=np.array_equal(t,ref_t) and sha(checkpoint)==conf['checkpoint_sha256']
            provenance_pass=provenance_pass and np.array_equal(n,ref_n) and np.isfinite(pred).all() and pred.shape==ref.shape
            provenance_pass=provenance_pass and conf['reference_sha256']==sha(ref_path)
            full_expected=conf['step'] in (20000,26000,job['budget'])
            if full_expected and (pred401 is None or t401 is None):
                provenance_pass=False;missing.append(str(raw.relative_to(BASE))+'#mandatory401')
            result=compute(pred,ref,n,10. if job['case']=='m10' else 5.)
            if full_expected:result.update(grid_checks(pred,ref,n))
            else:result['initial_condition_max_abs_error']=float(np.max(abs(pred[0]-ref[0])))
            difference={k:abs(result[k]-conf[k]) for k in result if k in conf and result[k] is not None and conf[k] is not None}
            difference401={};reference401_sha=None
            if pred401 is not None:
                ref401_path=BASE/'references'/job['case']/'test401.npz'
                with np.load(ref401_path) as f:ref401=f['state'];ref401_t=f['t'];ref401_n=f['n']
                provenance_pass=provenance_pass and np.array_equal(t401,ref401_t) and np.array_equal(n,ref401_n) and np.isfinite(pred401).all() and pred401.shape==ref401.shape
                result401=compute(pred401,ref401,n,10. if job['case']=='m10' else 5.)
                difference401={k:abs(result401[k]-conf[k+'_401']) for k in THRESHOLDS}
                reference401_sha=sha(ref401_path)
            maximum=max(list(difference.values())+list(difference401.values()))
            row=dict(path=str(p.relative_to(BASE)),case=job['case'],group=job['group'],seed=job['seed'],step=conf['step'],
                maximum_metric_abs_difference=maximum,maximum_401_metric_abs_difference=max(difference401.values(),default=None),
                full_endpoint_grid_checks_required=full_expected,mandatory_401_diagnostics_present=pred401 is not None and t401 is not None,
                raw_and_checkpoint_provenance_pass=bool(provenance_pass),pass_check=bool(maximum<2e-11 and provenance_pass),checkpoint_sha256=sha(checkpoint),
                metric_record_sha256=sha(p),prediction_sha256=sha(raw),reference_sha256=sha(ref_path),reference401_sha256=reference401_sha,
                **{k:v for k,v in result.items() if isinstance(v,(int,float))})
            rows.append(row)
            if not row['pass_check']:bad.append(row['path'])
    out=BASE/'analysis_outputs';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'independent_metrics_per_snapshot.csv',index=False)
    record=dict(updated_utc=time.time(),status='PASS_AVAILABLE_SNAPSHOTS' if not missing and not bad else 'FAIL',snapshots_audited=len(rows),
        missing=missing,metric_mismatches=bad,independent_implementation='FFT quadrature and independent NumPy branch rotation',
        source_code_sha256=sha(__file__),independent_metric_code_sha256=sha(Path(__file__).with_name('independent_metrics.py')),
        required_401_and_spatial_grid_diagnostics_checked=True,
        entire_study_complete=False,does_not_mark_missing_training_complete=True)
    write(out/'independent_metric_audit.json',record);print(record)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
