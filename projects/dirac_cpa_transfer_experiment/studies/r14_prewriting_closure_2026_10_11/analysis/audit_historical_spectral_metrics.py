"""Read-only independent metric recalculation for canonical R9-R13 spectra."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from plan import BASE,PROJECT,write
from .independent_metrics import compute

NAMES=('r9_wavepacket_periodic_potential','r10_wavepacket_budget_diagnosis','r11_independent_replication_ablation','r12_cross_condition_robustness','r13_long_time_recovery')

def main():
    rows=[];skipped=[]
    for name in NAMES:
        directory=PROJECT/'results/raw'/name/'formal'
        for p in sorted(directory.rglob('dense_grid.npz')):
            if 'snapshots' in p.parts:continue
            metric=p.parent/'metrics.json';config=p.parent/'config.json'
            if not metric.exists():continue
            m=json.loads(metric.read_text(encoding='utf-8'))
            # Carrier-only archives have no physical trajectory metric.
            if 'spinor_rel_l2_gauge_aligned' not in m:continue
            with np.load(p) as f:
                if not all(k in f for k in ('spectral_predicted_raw','spectral_reference','n')):
                    skipped.append(str(p.relative_to(PROJECT)));continue
                pred,ref,n=f['spectral_predicted_raw'],f['spectral_reference'],f['n']
            c=json.loads(config.read_text(encoding='utf-8'))
            cfg=c.get('configuration',c.get('source_configuration',{}))
            if isinstance(cfg,dict) and 'configuration' in cfg:cfg=cfg['configuration']
            mass=float(cfg.get('mass',10. if m.get('case')=='m10' else 5.))
            a=compute(pred,ref,n,mass)
            differences={k:abs(a[k]-m[k]) for k in a if k in m and a[k] is not None and m[k] is not None}
            maximum=max(differences.values());row=dict(experiment=name,path=str(p.relative_to(PROJECT)),
                max_metric_absolute_difference=maximum,pass_check=maximum<2e-11,**{k:v for k,v in a.items() if isinstance(v,(int,float))})
            rows.append(row)
        print('HISTORICAL_METRIC_AUDIT',name,len([r for r in rows if r['experiment']==name]),flush=True)
    out=BASE/'history';pd.DataFrame(rows).to_csv(out/'independent_R9_R13_metrics.csv',index=False)
    write(out/'independent_R9_R13_audit.json',dict(status='PASS' if rows and all(r['pass_check'] for r in rows) and not skipped else 'LIMITATION_OR_MISMATCH',
        endpoints_audited=len(rows),failures=[r['path'] for r in rows if not r['pass_check']],skipped=skipped,
        raw_history_unchanged=True,no_training_repeated=True,does_not_cover_R1_R8_or_all_secondary_snapshots=True))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
