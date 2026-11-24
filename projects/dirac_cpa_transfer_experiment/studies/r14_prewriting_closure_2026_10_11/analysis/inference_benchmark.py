"""Separate-process inference/RSS measurements at every primary and terminal endpoint."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,threading
from pathlib import Path
from plan import BASE,sha,write
from .expected_jobs import expected_jobs

def measure(checkpoint,output):
    import psutil
    process=psutil.Process();process.cpu_affinity([6]);start=time.perf_counter()
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from closure.config import configuration
    from closure.models import Model
    from closure.vendor.r9_evaluate import spectral_grid
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    checkpoint=Path(checkpoint);saved=torch.load(checkpoint,weights_only=False);job=saved['job']
    cfg=configuration(job['case'],saved['bounds']);model=Model(cfg,job['group'],job['seed'],job.get('options'))
    model.load_state_dict(saved['state_dict']);model.eval()
    with np.load(checkpoint.parent/'prediction.npz') as raw:times=raw['t'];expected=raw['prediction']
    setup=time.perf_counter()-start;rss_model=process.memory_info().rss;peak=rss_model;stop=threading.Event()
    def monitor():
        nonlocal peak
        while not stop.wait(.01):peak=max(peak,process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    wall=[];cpu=[]
    try:
        with threadpool_limits(limits=1):
            predicted=spectral_grid(model,cfg,times)
            difference=float(np.max(abs(predicted-expected)))
            if not np.isfinite(predicted).all() or difference>=2e-12:raise RuntimeError('Inference checkpoint/raw prediction mismatch')
            for _ in range(7):
                tick=time.perf_counter();ctick=time.process_time();spectral_grid(model,cfg,times)
                wall.append(time.perf_counter()-tick);cpu.append(time.process_time()-ctick)
    finally:stop.set();thread.join(timeout=1)
    memory=process.memory_info();peak=max(peak,memory.rss,getattr(memory,'peak_wset',0))
    record=dict(status='COMPLETE_VERIFIED_INFERENCE',case=job['case'],group=job['group'],stage=job['stage'],seed=job['seed'],step=saved['step'],
        checkpoint=checkpoint.relative_to(BASE).as_posix(),checkpoint_sha256=sha(checkpoint),script_sha256=sha(__file__),
        model_setup_seconds=setup,inference_seconds_samples=wall,inference_cpu_seconds_samples=cpu,
        inference_seconds_median=float(np.median(wall)),inference_cpu_seconds_median=float(np.median(cpu)),
        process_rss_after_model_setup_bytes=rss_model,process_peak_rss_bytes=int(peak),
        memory_includes_interpreter_model_loading_and_caches=True,peak_method='10ms sampling plus OS peak working set when available',
        points_per_call=len(times)*cfg.mode_count,warmup_calls=1,timed_calls=7,batch_time_points=32,
        analytic_free_prior=job['group']=='analytic_hard',all_carrier_network_calls_included=True,
        no_precomputed_carrier=True,maximum_difference_from_archived_prediction=difference,cpu_affinity=process.cpu_affinity(),
        threads=1,evaluation_reference_not_loaded=True,measurement_observations_do_not_select_checkpoints=True)
    write(output,record)

def main():
    import psutil
    runtime=BASE/'runtime';lock=runtime/'inference_benchmark_lock.json'
    if lock.exists():
        previous=json.loads(lock.read_text(encoding='utf-8'))
        try:
            p=psutil.Process(previous['pid'])
            if abs(p.create_time()-previous['created'])<.01 and p.pid!=os.getpid():raise RuntimeError('An inference benchmark is already active')
        except psutil.NoSuchProcess:pass
    process=psutil.Process();process.cpu_affinity([7]);write(lock,dict(pid=os.getpid(),created=process.create_time()))
    from closure.train import verify_freeze
    verify_freeze();out=BASE/'isolated_inference';out.mkdir(exist_ok=True);expected=0;complete=0;missing=[];failed_no_checkpoint=[]
    for j in expected_jobs():
        if j['type']!='coupled' or j['stage']=='development':continue
        for step in sorted({20000,j['budget']}):
            expected+=1;cp=BASE/j['directory']/'snapshots'/f'{step:05d}'/'model_state.pt'
            target=out/j['case']/j['group']/f"seed_{j['seed']}"/f'step_{step}.json'
            if target.exists():
                record=json.loads(target.read_text(encoding='utf-8'))
                if record['checkpoint_sha256']!=sha(cp) or record['script_sha256']!=sha(__file__):raise RuntimeError('Existing inference evidence source changed: '+str(target))
                complete+=1;continue
            if not (cp.parent/'metrics.json').exists():
                failure=BASE/j['directory']/'failure.json'
                if failure.exists():
                    f=json.loads(failure.read_text(encoding='utf-8'))
                    if f['status']=='DEPENDENCY_FAILED' or f.get('occurred_step',0)<=step:
                        failed_no_checkpoint.append(dict(checkpoint=cp.relative_to(BASE).as_posix(),failure=failure.relative_to(BASE).as_posix(),status=f['status']));continue
                missing.append(cp.relative_to(BASE).as_posix());continue
            target.parent.mkdir(parents=True,exist_ok=True);log=target.with_suffix('.log')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            tick=time.perf_counter()
            with log.open('a',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-X','utf8','-m','analysis.inference_benchmark','--checkpoint',str(cp),'--output',str(target)],
                    cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode:raise RuntimeError('Inference measurement engineering failure; '+str(log))
            record=json.loads(target.read_text(encoding='utf-8'));record['parent_observed_process_wall_seconds']=time.perf_counter()-tick;write(target,record)
            complete+=1;print('INFERENCE_MEASURED',j['case'],j['group'],j['seed'],step,flush=True)
    assert expected==255
    report=dict(updated_utc=time.time(),status='ALL_AVAILABLE_ENDPOINTS_MEASURED_FAILURES_ACCOUNTED' if complete+len(failed_no_checkpoint)==expected else 'WAITING_FOR_FIXED_ENDPOINTS',
        prescribed_checkpoints=expected,complete_measurements=complete,missing=missing,failed_no_valid_checkpoint=failed_no_checkpoint,whole_study_complete=False,
        scientific_failed_checkpoints_are_absent_not_zero_cost=True)
    write(out/'status.json',report);print('INFERENCE_STATUS',complete,'/',expected)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--output');a=parser.parse_args()
    if a.checkpoint:measure(a.checkpoint,a.output)
    else:main()

