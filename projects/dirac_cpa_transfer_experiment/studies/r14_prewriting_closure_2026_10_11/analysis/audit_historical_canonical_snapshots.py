"""Independently recalculate legacy canonical table snapshots beyond final-array audits."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
from plan import BASE,PROJECT,write,sha
from .independent_metrics import compute

def configuration(path):
    for parent in path.parents:
        candidate=parent/'config.json'
        if candidate.exists():
            data=json.loads(candidate.read_text(encoding='utf-8'))
            config=data.get('configuration',data.get('source_configuration',data))
            if isinstance(config,dict) and 'configuration' in config:config=config['configuration']
            return config,candidate
        if parent==PROJECT:break
    raise RuntimeError('No archived configuration for '+str(path))

def main():
    start=time.time();trace=pd.read_csv(BASE/'history/canonical_row_traceability.csv')
    final_paths=set()
    for filename in ('independent_R1_R8_metrics.csv','independent_R9_R13_metrics.csv'):
        final_paths.update(pd.read_csv(BASE/'history'/filename).path.str.replace('\\','/',regex=False))
    missing=trace[~trace.raw_array_path.isin(final_paths)]
    # Audit only new canonical sources; historical final-array evidence stays untouched.
    rows=[];errors=[]
    for relative in sorted(set(missing.raw_array_path.dropna())):
        path=PROJECT/relative;metric_path=path.parent/'metrics.json'
        try:
            native=json.loads(metric_path.read_text(encoding='utf-8'));cfg,conf_path=configuration(path)
            with np.load(path) as raw:
                pred,reference,n=raw['spectral_predicted_raw'],raw['spectral_reference'],raw['n']
            if not np.isfinite(pred).all() or not np.isfinite(reference).all():raise FloatingPointError('Nonfinite archived trajectory')
            result=compute(pred,reference,n,float(cfg['mass']))
            differences={k:abs(v-native[k]) for k,v in result.items() if k in native and v is not None and native[k] is not None}
            passed=bool(differences) and max(differences.values())<2e-11
            row=dict(path=relative,pass_check=passed,compared_metric_count=len(differences),
                max_metric_absolute_difference=max(differences.values()),raw_sha256=sha(path),
                native_metrics_sha256=sha(metric_path),configuration_sha256=sha(conf_path),
                **{k:v for k,v in result.items() if isinstance(v,(int,float))})
            rows.append(row)
            if not passed:errors.append(dict(path=relative,differences=differences))
        except Exception as error:errors.append(dict(path=relative,error=repr(error)))
    pd.DataFrame(rows).to_csv(BASE/'history/independent_canonical_snapshot_metrics.csv',index=False)
    report=dict(updated_utc=time.time(),status='PASS_CANONICAL_SNAPSHOTS' if rows and not errors else 'MISMATCH_OR_LIMITATION',
        unique_canonical_snapshots_audited=len(rows),canonical_table_rows_requiring_this_audit=len(missing),
        errors=errors,elapsed_wall_seconds=time.time()-start,
        historical_final_audits_not_overwritten=True,old_scientific_decisions_and_gauges_unchanged=True,
        original_history_read_only=True,no_training_repeated=True,snapshots_not_independent_training_replicates=True)
    write(BASE/'history/independent_canonical_snapshot_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
