"""Reconcile canonical legacy table rows with original metric files and raw audit coverage."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from plan import BASE,PROJECT,write,sha
from .archive_history import SOURCES

METRICS=('spinor_rel_l2','spinor_rel_l2_raw','spinor_rel_l2_gauge_aligned','current_rel_l2',
    'fidelity_mean','max_norm_drift','density_rel_l2','rho_rel_l2','modal_population_rel_l2',
    'mode_occupation_rmse_rel','spectral_occupation_rel_l2','carrier_free_pde_residual_rms',
    'carrier_phase_slope_mean_relative_error','branch_coefficient_rel_l2')

def main():
    table=pd.read_csv(BASE/'history/historical_results_all_native_fields.csv')
    roots={n:[PROJECT/'results/raw'/name] for n,name,_ in SOURCES}
    roots[1].append(PROJECT/'results/raw/structure_weight_sensitivity')
    roots[10].append(roots[9][0])  # Explicitly preserve the R9 ancestor rows in R10.
    cache={};pool={}
    for number,directories in roots.items():
        candidates=[]
        for directory in directories:
            for path in sorted(directory.rglob('metrics.json')):
                key=path.as_posix()
                if key not in cache:
                    data=json.loads(path.read_text(encoding='utf-8'));cache[key]=(path,data)
                candidates.append(cache[key])
        pool[number]=candidates
    coverage={}
    for filename in ('independent_R1_R8_metrics.csv','independent_R9_R13_metrics.csv','independent_canonical_snapshot_metrics.csv'):
        if not (BASE/'history'/filename).exists():continue
        audit=pd.read_csv(BASE/'history'/filename)
        coverage.update({Path(str(r['path'])).as_posix():bool(r['pass_check']) for _,r in audit.iterrows()})
    rows=[]
    for _,record in table.iterrows():
        number=int(record.historical_round);matches=[]
        for path,native in pool[number]:
            if 'seed' in native and native['seed']!=record.seed:continue
            keys=[k for k in METRICS if k in native and k in record and pd.notna(record[k]) and native[k] is not None]
            if len(keys)<2:continue
            if all(np.isclose(float(record[k]),float(native[k]),rtol=2e-11,atol=2e-11) for k in keys):
                matches.append((path,keys))
        # All matching aliases remain visible; prefer a native explicit path then a final artifact.
        direct=None
        if pd.notna(record.get('metrics_path')):
            old=str(record.metrics_path).replace('\\','/');relative=old.split('/results/raw/',1)[-1]
            if relative.startswith('main/'):relative='cpa_transfer/'+relative[5:]
            if relative.startswith('sensitivity/'):relative='structure_weight_sensitivity/'+relative[12:]
            direct=(PROJECT/'results/raw'/relative).as_posix()
        if pd.notna(record.get('run_path')):
            direct=(roots[number][0]/str(record.run_path)/'metrics.json').as_posix()
        matches.sort(key=lambda item:(item[0].as_posix()!=direct,
            'checkpoints' in item[0].parts or 'snapshots' in item[0].parts,-len(item[1]),item[0].as_posix()))
        r=dict(historical_round=number,canonical_source_csv=record.canonical_source_csv,
            canonical_row_index=int(record.canonical_row_index),model=record.get('model'),case=record.get('case'),
            seed=int(record.seed),native_steps=record.get('steps'),native_stage=record.get('stage'),
            status='MATCHED_NATIVE_METRICS' if matches else 'UNRESOLVED_NATIVE_SOURCE',matching_native_sources=len(matches))
        if matches:
            path,keys=matches[0];dense=path.parent/'dense_grid.npz';relative=dense.relative_to(PROJECT).as_posix()
            r.update(native_metrics_path=path.relative_to(PROJECT).as_posix(),native_metrics_sha256=sha(path),
                matching_metric_names=';'.join(keys),all_matching_metrics_paths=';'.join(p.relative_to(PROJECT).as_posix() for p,_ in matches),
                raw_array_available=dense.exists(),raw_is_secondary_snapshot=bool('snapshots' in dense.parts or 'checkpoints' in dense.parts),raw_array_path=relative if dense.exists() else None,
                independent_raw_metric_audit=coverage.get(relative),
                independent_audit_scope=('PASS_CANONICAL_SECONDARY_ARRAY' if 'snapshots' in dense.parts or 'checkpoints' in dense.parts else 'PASS_NATIVE_FINAL_ARRAY') if coverage.get(relative) else 'NOT_COVERED_BY_CURRENT_RAW_ARRAY_AUDIT')
        rows.append(r)
    result=pd.DataFrame(rows)
    result['canonical_rows_sharing_selected_raw_path']=result.groupby('raw_array_path',dropna=False)['canonical_row_index'].transform('count')
    result.to_csv(BASE/'history/canonical_row_traceability.csv',index=False)
    report=dict(updated_utc=time.time(),status='PASS_CANONICAL_ROW_LINKS' if all(r['status']=='MATCHED_NATIVE_METRICS' for r in rows) else 'UNRESOLVED_ROWS_RETAINED',
        canonical_rows=len(rows),matched_rows=sum(r['status']=='MATCHED_NATIVE_METRICS' for r in rows),
        unresolved_rows=[{k:r[k] for k in ('historical_round','canonical_row_index','model','seed')} for r in rows if r['status']!='MATCHED_NATIVE_METRICS'],
        rows_with_independently_recomputed_raw_array=sum(r.get('independent_raw_metric_audit') is True for r in rows),
        raw_arrays_and_aliases_are_not_independent_replicate_count=True,R10_ancestor_rows_are_continuations_not_new_replicates=True,
        archived_table_and_original_sources_unchanged=True,all_matching_alias_paths_preserved=True,
        source_matching_does_not_change_native_decisions=True)
    write(BASE/'history/canonical_row_traceability_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
