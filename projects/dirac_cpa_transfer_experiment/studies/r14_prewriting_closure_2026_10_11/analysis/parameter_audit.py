"""Count saved weight/bias tensors, without constructing a training model."""
from __future__ import annotations
import json,time
import pandas as pd
import torch
from plan import BASE,sha,write
from .expected_jobs import expected_jobs


def main():
    rows=[];missing=[];dependencies=[];errors=[]
    for job in expected_jobs():
        directory=BASE/job['directory'];path=directory/'initial_state.pt';cp=directory/'config.json'
        if not path.exists() or not cp.exists():
            fp=directory/'failure.json'
            if fp.exists() and json.loads(fp.read_text(encoding='utf-8'))['status']=='DEPENDENCY_FAILED':
                dependencies.append(dict(directory=job['directory'],failure_sha256=sha(fp),valid_model_not_created=True))
            else:missing.append(job['directory'])
            continue
        conf=json.loads(cp.read_text(encoding='utf-8'));saved=torch.load(path,weights_only=False);state=saved['state_dict']
        params={k:v for k,v in state.items() if k.endswith(('.weight','.bias'))}
        buffers={k:v for k,v in state.items() if k not in params}
        total=sum(v.numel() for v in params.values());carrier=sum(v.numel() for k,v in params.items() if k.startswith('carrier.'))
        if job['type']=='carrier':carrier=total
        frozen=carrier if job.get('source') else 0;active=total-frozen
        checks=dict(config_job=conf['job']==job,initial_hash_metadata=saved['initial_hash']==conf['initial_hash'],
            parameters_float64=all(v.dtype==torch.float64 for v in params.values()),
            parameters_finite=all(bool(torch.isfinite(v).all()) for v in params.values()),nonempty_weights=total>0)
        cost_paths=list(directory.glob('snapshots/*/cost.json'))
        if (directory/'complete.json').exists():cost_paths.append(directory/'complete.json')
        for p in cost_paths:
            cost=json.loads(p.read_text(encoding='utf-8'))
            checks[p.relative_to(BASE).as_posix()]=all(cost[k]==v for k,v in
                dict(total_parameters=total,active_parameters=active,frozen_parameters=frozen).items())
        row=dict(directory=job['directory'],case=job['case'],group=job.get('group',job.get('kind')),seed=job['seed'],
            stage=job['stage'],type=job['type'],total_parameters=total,active_parameters=active,frozen_parameters=frozen,
            carrier_parameters=carrier,envelope_parameters=total-carrier if job['type']=='coupled' else None,
            saved_parameter_tensor_bytes=sum(v.numel()*v.element_size() for v in params.values()),
            saved_nonparameter_buffer_bytes=sum(v.numel()*v.element_size() for v in buffers.values()),
            fixed_Fourier_feature_coefficients=int(state['networks.0.B'].numel()) if 'networks.0.B' in state else 0,
            buffers_excluded_from_nn_parameter_count=True,initial_state_sha256=sha(path),config_sha256=sha(cp),
            weight_tensor_count=len(params),cost_records_checked=len(cost_paths),pass_check=all(checks.values()),
            failed_checks=[k for k,v in checks.items() if not v],parameter_shapes={k:list(v.shape) for k,v in params.items()},
            buffer_shapes={k:list(v.shape) for k,v in buffers.items()})
        rows.append(row)
        if not row['pass_check']:errors.append(dict(directory=job['directory'],failed_checks=row['failed_checks']))
    attempts=json.loads((BASE/'analysis_outputs/attempt_completion_audit.json').read_text(encoding='utf-8'))
    full=attempts['all_training_attempts_verified'] and len(rows)+len(dependencies)==234 and not errors
    out=BASE/'analysis_outputs';pd.DataFrame([{k:v for k,v in r.items() if k not in ('parameter_shapes','buffer_shapes','failed_checks')} for r in rows]).to_csv(out/'independent_parameter_counts.csv',index=False)
    report=dict(updated_utc=time.time(),status='FAIL' if errors else ('ALL_PRESCRIBED_PARAMETER_RECORDS_VERIFIED' if full else 'PASS_AVAILABLE_PARAMETER_RECORDS'),
        prescribed_attempts=234,actual_initial_states_checked=len(rows),cost_records_checked=sum(r['cost_records_checked'] for r in rows),
        missing=missing,preserved_no_model_dependencies=dependencies,errors=errors,checks=rows,
        all_prescribed_parameter_records_verified=full,source_code_sha256=sha(__file__),
        independent_saved_tensor_shapes_without_model_constructor=True,fixed_buffers_separately_reported=True,
        frozen_architectures_have_linear_weight_bias_parameters_only=True,
        original_states_not_written=True,counts_are_not_process_peak_memory=True,whole_goal_complete=False)
    write(out/'independent_parameter_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','missing','preserved_no_model_dependencies')},ensure_ascii=False))
    if errors:raise RuntimeError('Saved tensor parameter counts differ from cost records')


if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6]);torch.set_num_threads(1);main()
