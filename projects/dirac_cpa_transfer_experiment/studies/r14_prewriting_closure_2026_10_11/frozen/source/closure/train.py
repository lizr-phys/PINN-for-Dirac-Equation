"""Fixed-budget trainer. Only physical configuration and source weights are read.

Future reference trajectories are available only to a separate evaluation process.
"""
from __future__ import annotations
import argparse,json,os,random,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from plan import BASE,write,sha
from .config import configuration
from .models import Model,StaticCarrier,free_loss,training_loss
from .vendor.r13_models import LearnedRecoveryCarrier
from .vendor.r9_models import tensor_digest,parameter_counts
from .vendor.r8_sampling import TimeSampler

def atomic_save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def verify_freeze():
    path=BASE/'frozen/source_manifest.json'
    if not path.is_file():raise RuntimeError('Source not frozen')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for entry in manifest['files']:
        if sha(BASE/entry['path'])!=entry['sha256']:raise RuntimeError('Frozen file changed: '+entry['path'])
    return sha(path)

def run(job,*,engineering=False,stop_after=None,do_evaluation=True):
    starttick=time.perf_counter();free=job['type']=='carrier';directory=BASE/job['directory'];directory.mkdir(parents=True,exist_ok=True)
    if not engineering:freeze=verify_freeze()
    else:freeze='ENGINEERING_NOT_SCIENCE'
    final=directory/'complete.json'
    if final.exists():
        old=json.loads(final.read_text(encoding='utf-8'))
        if old['job']!=job:raise RuntimeError('Changed completed task')
        return old
    if (directory/'failure.json').exists():return json.loads((directory/'failure.json').read_text(encoding='utf-8'))
    mask=int(os.environ.get('R14_CPU_MASK','0'));process=psutil.Process()
    if mask:
        cores=[i for i in range(os.cpu_count()) if mask & (1<<i)];process.cpu_affinity(cores)
        if process.cpu_affinity()!=cores:raise RuntimeError('Affinity mismatch')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.use_deterministic_algorithms(True)
    cfg=configuration(job['case'],job.get('bounds'))
    source_cost={};source_sha=None
    if free:
        model=StaticCarrier(cfg,job['seed']) if job['kind']=='static_frequency' else LearnedRecoveryCarrier(cfg,job['kind'],job['seed'])
    else:
        model=Model(cfg,job['group'],job['seed'],job.get('options'))
        if job.get('source'):
            source=BASE/job['source']
            if (source/'failure.json').exists():
                result=dict(status='DEPENDENCY_FAILED',job=job,complete=False,source=job['source']);write(directory/'failure.json',result);return result
            if not (source/'complete.json').exists():raise RuntimeError('Carrier source incomplete')
            s=torch.load(source/'model_state.pt',weights_only=False)
            if s['job']['seed']!=job['seed'] or s['job']['kind']!=model.carrier_kind:raise RuntimeError('Source seed/kind mismatch')
            if not engineering and (s['step']!=6000 or s['job']['budget']!=6000):raise RuntimeError('Formal carrier source did not reach6000 steps')
            src_cfg=configuration(s['job']['case'],s['bounds'])
            if (src_cfg.mass,src_cfg.t_max,src_cfg.wave_numbers)!=(cfg.mass,cfg.t_max,cfg.wave_numbers):raise RuntimeError('Source physical incompatibility')
            model.carrier.load_state_dict(s['state_dict']);model.carrier.freeze();source_sha=sha(source/'model_state.pt')
            source_cost=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    params=tuple(p for p in model.parameters() if p.requires_grad)
    initial_hash={'all':tensor_digest(model.parameters()),'envelope':None if free else tensor_digest(model.networks[0].parameters()),
        'carrier':tensor_digest(model.parameters()) if free else tensor_digest(model.carrier.parameters())}
    sampler=TimeSampler(cfg,job['seed'],carrier=free)
    optimizer=torch.optim.Adam(params,lr=job.get('options',{}).get('lr',.001))
    conf=dict(job=job,bounds=[cfg.n_min,cfg.n_max],configuration=cfg.as_dict(),source_sha256=source_sha,
        initial_hash=initial_hash,source_manifest_sha256=freeze,analytic_prior=(not free and model.carrier_kind=='exact'),
        no_future_reference_in_training=True,reference_evaluation_separate_process=True,torch_version=torch.__version__,
        python=sys.version,cpu_mask=mask,cpu_affinity=process.cpu_affinity(),threads=1)
    if (directory/'config.json').exists():
        old=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        for k in ('job','bounds','source_sha256','source_manifest_sha256'):
            if old[k]!=json.loads(json.dumps(conf[k])):raise RuntimeError('Recovery config changed '+k)
    else:
        write(directory/'config.json',conf);atomic_save(directory/'initial_state.pt',dict(state_dict=model.state_dict(),initial_hash=initial_hash))
    recovery=directory/'recovery_state.pt';step0=0;history=[];completed=[]
    seconds=cpu_seconds=eval_seconds=io_seconds=0.;setup_seconds=time.perf_counter()-starttick
    peak=process.memory_info().rss
    if recovery.exists():
        saved=torch.load(recovery,weights_only=False)
        if saved['job']!=job:raise RuntimeError('Changed recovery job')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);sampler.load_state_dict(saved['sampler'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        step0=saved['step'];history=saved['history'];completed=saved['completed']
        seconds,cpu_seconds,eval_seconds,io_seconds=(saved[k] for k in ('training_wall_seconds','training_cpu_seconds','evaluation_seconds','checkpoint_io_seconds'))
        setup_seconds+=saved['setup_seconds'];peak=max(peak,saved['peak_rss_bytes'])
        with (directory/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'event':'EXACT_RESUME','saved_step':step0,'utc':time.time()})+'\n')
    stopmonitor=threading.Event()
    def monitor():
        nonlocal peak
        while not stopmonitor.wait(.05):peak=max(peak,process.memory_info().rss)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    budget=job['budget'];snapshots=job.get('snapshots',[budget])
    def costs(step):
        number=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in params)
        prewall=source_cost.get('training_wall_seconds',0.);pre_cpu=source_cost.get('training_cpu_seconds',0.)
        prepoints=source_cost.get('unit_spinor_residual_points',0)
        return dict(training_wall_seconds=seconds,training_cpu_seconds=cpu_seconds,evaluation_seconds=eval_seconds,
            checkpoint_io_seconds=io_seconds,setup_seconds=setup_seconds,peak_rss_bytes=peak,total_parameters=number,
            active_parameters=active,frozen_parameters=number-active,step=step,source_pretraining_wall_seconds=prewall,
            source_pretraining_cpu_seconds=pre_cpu,attributed_training_wall_seconds=seconds+prewall,
            attributed_training_cpu_seconds=cpu_seconds+pre_cpu,carrier_pretraining_reused=bool(job.get('reuse_source')),
            incurred_pipeline_seconds=seconds+eval_seconds+io_seconds+setup_seconds,
            attributed_pipeline_seconds=seconds+eval_seconds+io_seconds+setup_seconds+source_cost.get('incurred_pipeline_seconds',0.),
            unit_spinor_residual_points=(2 if free else 1)*step*cfg.spectral_collocation_points,
            attributed_unit_spinor_residual_points=(2 if free else 1)*step*cfg.spectral_collocation_points+prepoints,
            complex_scalar_residual_components=2*((2 if free else 1)*step*cfg.spectral_collocation_points+prepoints),
            free_steps=step if free else (6000 if job.get('source') else 0),coupled_steps=0 if free else step,
            peak_rss_includes_process_runtime_and_cache=True,engineering=engineering)
    def save(step):
        nonlocal io_seconds
        tick=time.perf_counter();cost=costs(step)
        atomic_save(recovery,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
            job=job,bounds=[cfg.n_min,cfg.n_max],step=step,history=history,completed=completed,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),**{k:v for k,v in cost.items() if k!='step'}))
        pd.DataFrame(history).to_csv(directory/'loss_history.csv',index=False)
        write(directory/'progress.json',dict(job=job,fixed_endpoint_reached=step==budget,utc=time.time(),**cost))
        io_seconds+=time.perf_counter()-tick
    def snapshot(step):
        nonlocal io_seconds,eval_seconds
        d=directory/'snapshots'/f'{step:05d}';d.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
        atomic_save(d/'model_state.pt',dict(state_dict=model.state_dict(),job=job,bounds=[cfg.n_min,cfg.n_max],step=step,
            initial_hash=initial_hash,time_stream_sha256=sampler.digest(),normalized_time_stream_sha256=sampler.digest(True),source_sha256=source_sha))
        write(d/'cost.json',costs(step));io_seconds+=time.perf_counter()-tick
        if do_evaluation:
            before=(torch.get_rng_state().clone(),sampler.generator.get_state().clone(),random.getstate(),np.random.get_state())
            tick=time.perf_counter();log=d/'evaluation.log'
            with log.open('a',encoding='utf-8') as f:
                result=subprocess.run([sys.executable,'-X','utf8','-m','closure.evaluate',str(directory),str(step)],cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
            eval_seconds+=time.perf_counter()-tick
            if result.returncode:raise RuntimeError('Evaluation subprocess failure '+str(log))
            if not torch.equal(before[0],torch.get_rng_state()) or not torch.equal(before[1],sampler.generator.get_state()) or before[2]!=random.getstate() or not np.array_equal(before[3][1],np.random.get_state()[1]):
                raise RuntimeError('Evaluation altered training random stream')
        completed.append(step);save(step);write(d/'cost.json',costs(step))
    try:
        if step0 in snapshots and step0 not in completed:snapshot(step0)
        for step in range(step0+1,budget+1):
            tick=time.perf_counter();ctick=time.process_time();optimizer.zero_grad(set_to_none=True)
            k,t=sampler.sample();loss=free_loss(model,cfg,k,t) if free else training_loss(model,cfg,k,t)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('Missing/nonfinite gradient')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Nonfinite weights')
            seconds+=time.perf_counter()-tick;cpu_seconds+=time.process_time()-ctick
            if step==1 or step%25==0 or step==budget:history.append(dict(step=step,loss=float(loss.detach()),training_wall_seconds=seconds,training_cpu_seconds=cpu_seconds))
            if step%500==0 or step in snapshots or step==stop_after:
                if shutil.disk_usage(BASE).free<4*2**30:save(step);raise RuntimeError('OPERATIONAL_STORAGE_GATE: free space below4GiB')
                save(step);print('TRAIN',job['case'],job.get('group',job.get('kind')),job['seed'],step,'/',budget,round(seconds,1),flush=True)
            if step in snapshots:snapshot(step)
            if step==stop_after:return dict(status='ENGINEERING_CONTROLLED_INTERRUPT',step=step)
        if budget not in completed:raise RuntimeError('Missing fixed endpoint')
        tick=time.perf_counter();shutil.copyfile(directory/'snapshots'/f'{budget:05d}'/'model_state.pt',directory/'model_state.pt');io_seconds+=time.perf_counter()-tick
        record=dict(job=job,status='COMPLETE',fixed_endpoint_reached=True,source_sha256=source_sha,
            initial_hash=initial_hash,time_stream_sha256=sampler.digest(),normalized_time_stream_sha256=sampler.digest(True),**costs(budget))
        write(final,record);return record
    except Exception as error:
        record=dict(job=job,status='NUMERICAL_FAILURE' if isinstance(error,FloatingPointError) else 'ENGINEERING_INTERRUPTION',complete=False,
            error=repr(error),traceback=traceback.format_exc(),last_saved_step=json.loads((directory/'progress.json').read_text(encoding='utf-8')).get('step',0) if (directory/'progress.json').exists() else 0,
            occurred_step=locals().get('step',step0),observed_training_wall_seconds=seconds,observed_training_cpu_seconds=cpu_seconds,utc=time.time())
        if isinstance(error,FloatingPointError):
            atomic_save(directory/'failed_state.pt',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),job=job));write(directory/'failure.json',record)
        else:
            d=directory/'interruptions';d.mkdir(exist_ok=True);write(d/f'{time.time_ns()}.json',record)
        raise
    finally:stopmonitor.set();monitor_thread.join(timeout=1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job_file');p.add_argument('--engineering',action='store_true');a=p.parse_args()
    job=json.loads(Path(a.job_file).read_text(encoding='utf-8'))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run(job,engineering=a.engineering)
