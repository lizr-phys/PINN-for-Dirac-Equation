"""Prespecified unchanged complete-model transfer. No correction or optimization."""
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
from plan import BASE,SEEDS,EXTENSION,THRESHOLDS,write,sha
from closure.config import configuration
from closure.models import Model
from closure.vendor.r9_evaluate import spectral_grid
from .independent_metrics import compute

def state_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()

def main():
    selected=json.loads((BASE/'frozen/baseline_selection.json').read_text(encoding='utf-8'));family=selected['strongest_family']
    groups=('learned_hard','analytic_hard',family);rows=[];out=BASE/'zero_shot';out.mkdir(exist_ok=True)
    for case in EXTENSION:
        target_cfg=configuration(case)
        with np.load(BASE/'references'/case/'test201.npz') as f:times,reference=f['t'],f['state']
        for group in groups:
            for seed in SEEDS:
                source=BASE/'formal/coupled/m5T5'/group/f'seed_{seed}';checkpoint=source/'model_state.pt'
                result_path=out/f'{case}_{group}_seed{seed}.json';raw_path=out/f'{case}_{group}_seed{seed}.npz'
                if result_path.exists():
                    result=json.loads(result_path.read_text(encoding='utf-8'))
                    if result.get('evaluation_script_sha256')!=sha(__file__):raise RuntimeError('Changed zero-shot measurement source; retain old evidence before any repair')
                    if result.get('status')=='SOURCE_NUMERICAL_OR_DEPENDENCY_FAILURE':
                        if sha(source/'failure.json')!=result['source_failure_sha256']:raise RuntimeError('Changed zero-shot failure provenance')
                    elif sha(checkpoint)!=result['source_sha256'] or sha(raw_path)!=result['prediction_sha256']:
                        raise RuntimeError('Changed zero-shot cached source or prediction')
                    rows.append(result);continue
                failure=source/'failure.json'
                if failure.exists():
                    f=json.loads(failure.read_text(encoding='utf-8'))
                    result=dict(case=case,group=group,seed=seed,status='SOURCE_NUMERICAL_OR_DEPENDENCY_FAILURE',
                        source_failure=str(failure.relative_to(BASE)),source_failure_sha256=sha(failure),
                        evaluation_script_sha256=sha(__file__),
                        zero_shot=True,retrained=False,seed_joint_pass=False,new_training_performed=False,
                        inference_executed=False,accuracy_and_inference_cost_not_filled_with_zero=True)
                    rows.append(result);write(result_path,result);continue
                if not checkpoint.exists() or not (source/'complete.json').exists():raise RuntimeError('Zero-shot source incomplete '+str(checkpoint))
                setup_tick=time.perf_counter();saved=torch.load(checkpoint,weights_only=False);cfg=configuration('m5T5',saved['bounds'])
                expected_budget=26000 if group in ('siren','fourier') else 20000
                if saved['step']!=expected_budget or saved['job']['budget']!=expected_budget:raise RuntimeError('Source is not the frozen terminal checkpoint')
                model=Model(cfg,group,seed,saved['job'].get('options'));model.load_state_dict(saved['state_dict'])
                before=state_digest(model);source_sha=sha(checkpoint)
                setup_seconds=time.perf_counter()-setup_tick
                tick=time.perf_counter();cpu_tick=time.process_time();prediction=spectral_grid(model,cfg,times);seconds=time.perf_counter()-tick;cpu_seconds=time.process_time()-cpu_tick
                if state_digest(model)!=before or sha(checkpoint)!=source_sha:raise RuntimeError('Zero-shot evaluation changed weights or initial data')
                metric_tick=time.perf_counter();result=compute(prediction,reference,np.array(target_cfg.indices),target_cfg.mass)
                metric_seconds=time.perf_counter()-metric_tick
                result.update(case=case,group=group,seed=seed,transfer='UNCHANGED_M5T5_MODEL_NO_WEIGHT_OR_INITIAL_DATA_CORRECTION',
                    status='EVALUATED_UNCHANGED_MODEL',training_seconds_new=0.,evaluation_inference_seconds=seconds,
                    evaluation_inference_cpu_seconds=cpu_seconds,model_loading_and_setup_seconds=setup_seconds,
                    metric_computation_seconds=metric_seconds,inference_timed_calls=1,inference_warmup_calls=0,
                    isolated_source_inference_record=f'isolated_inference/m5T5/{group}/seed_{seed}/step_{saved["step"]}.json',
                    evaluation_script_sha256=sha(__file__),
                    source_checkpoint=str(checkpoint.relative_to(BASE)),source_sha256=source_sha,source_step=saved['step'],
                    initial_condition_error_max=float(abs(prediction[0]-reference[0]).max()),initial_error_definition='maximum spectral spinor coefficient difference at t=0',
                    zero_shot=True,retrained=False,inference_executed=True,weights_and_initial_buffers_unchanged=True,model_state_digest=before)
                result['seed_joint_pass']=all(result[k]>=v if k=='fidelity_mean' else result[k]<=v for k,v in THRESHOLDS.items())
                np.savez_compressed(raw_path,t=times,n=np.array(target_cfg.indices),prediction=prediction)
                result['prediction_sha256']=sha(raw_path);rows.append(result);write(result_path,result)
    pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    executed=sum(r.get('inference_executed',False) for r in rows)
    assert len(rows)==45
    write(out/'complete.json',dict(status='ALL_PRESCRIBED_DIAGNOSTICS_ACCOUNTED',evaluations_executed=executed,
        source_dependency_failures=len(rows)-executed,attempts_accounted=len(rows),expected=45,failed_accuracy_preserved=True,
        initial_condition_mismatch_is_reported=True,does_not_substitute_for_retraining_validation=True,
        independent_raw_recalculation_still_required=True,whole_study_complete=False))
    print('ZERO_SHOT',len(rows))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
