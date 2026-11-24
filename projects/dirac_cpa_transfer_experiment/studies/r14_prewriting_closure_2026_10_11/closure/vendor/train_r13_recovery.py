from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .r13_config import EXPERIMENT, ARTIFACTS, CURVE_STEPS, FREE_CURVE_STEPS, configuration
from .r13_models import LearnedRecoveryCarrier, R13Model, hashes, parameter_counts, coupled_parameters
from .r13_residual import training_loss, free_carrier_residual
from .r9_models import tensor_digest
from .r8_sampling import TimeSampler
from .freeze_r7_phase_release import write_json, sha256
from .train_r7_phase_release import atomic_torch_save, event
from .train_wavepacket_shared_spectral_ltpb import ProcessRSSMonitor, _set_runtime, _environment_snapshot
from .train_r9_wavepacket import math_norm

def train_job(job, directory, *, smoke=False, stop_after=None):
    """Separate reusable free sources and full coupled runs, with exact Adam/RNG resume."""
    job_started=time.perf_counter()
    cfg, directory = configuration(job['case']), Path(directory)
    free_job = job['type']=='carrier'
    if job['type'] not in ('carrier','coupled'):
        raise ValueError(job)
    budget = 2 if smoke else (6000 if free_job else 20000)
    checkpoints = (budget,) if smoke else (FREE_CURVE_STEPS if free_job else CURVE_STEPS)
    frozen = None
    if not smoke:
        from .freeze_r13_recovery import verify_freeze
        from .r13_plan import validate_job
        frozen = verify_freeze(); validate_job(job)
    final = directory/'metrics.json'
    if final.exists():
        metrics=json.loads(final.read_text(encoding='utf-8'))
        if metrics['executed_steps']!=budget or metrics['job']!=job:
            raise RuntimeError('Changed cached R13 job')
        if not all((directory/name).is_file() for name in ARTIFACTS+('recovery_state.pt','execution_cost.json')):
            raise RuntimeError('Incomplete final artifacts require completion from saved endpoint')
        return metrics
    if (directory/'failure.json').exists():
        raise RuntimeError('Numerical failure retained; no selective fresh rerun: '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    device = _set_runtime(job['seed'],1)
    source=None; source_metrics=None; source_sha=None
    if free_job:
        model = LearnedRecoveryCarrier(cfg,job['carrier_kind'],job['seed'])
        initial = dict(carrier=tensor_digest(model.parameters()))
        parameters=tuple(model.parameters())
    else:
        model=R13Model(cfg,job['carrier_kind'],job['envelope_kind'],job['seed'])
        initial=hashes(model)
        if job['carrier_kind']!='exact':
            source=Path(job['pretrain_source'])
            if not source.is_absolute():
                from .r13_config import ROOT
                source=ROOT/source
            source_metrics=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
            state=torch.load(source/'model_state.pt',weights_only=False)
            if (state['job']['seed']!=job['seed'] or state['job']['case']!=job['case']
                or state['job']['carrier_kind']!=job['carrier_kind'] or state['executed_steps']!=(2 if smoke else 6000)):
                raise RuntimeError('Carrier source mismatch')
            source_sha=sha256(source/'model_state.pt')
            model.carrier.load_state_dict(state['state_dict'])
            model.carrier.freeze()
        parameters=coupled_parameters(model)
    sampler=TimeSampler(cfg,job['seed'],carrier=free_job)
    optimizer=torch.optim.Adam(parameters,lr=cfg.learning_rate)
    config=dict(experiment=EXPERIMENT,job=job,seed=job['seed'],model=job.get('group',job['carrier_kind']),
        smoke=smoke,configuration=cfg.as_dict(),executed_budget=budget,analytic_prior=job['carrier_kind']=='exact',
        freeze_manifest_sha256=frozen,initial_hashes=initial,pretrain_source_sha256=source_sha,
        carrier_initial_pairing='matched phase and t=0 rate; only same-shaped unscaled weights elementwise paired',
        carrier_frozen=not free_job,full_time_derivative=True,future_reference_in_checkpoint=False,
        environment=_environment_snapshot(device),cpu_affinity_mask=os.environ.get('R7_CPU_AFFINITY_MASK'))
    config_path=directory/'config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding='utf-8'))
        for key in ('job','smoke','configuration','freeze_manifest_sha256','pretrain_source_sha256','cpu_affinity_mask'):
            if old[key]!=json.loads(json.dumps(config))[key]:
                raise RuntimeError('Changed recovery configuration: '+key)
    else:
        write_json(config_path,config)
        atomic_torch_save(directory/'initial_state.pt',dict(state_dict=model.state_dict(),job=job,initial_hashes=initial))
    start=1; seconds=0.; eval_seconds=0.; io_seconds=0.; setup_seconds=0.; peak=0
    history=[]; curves=[]; completed=[]
    recovery=directory/'recovery_state.pt'
    if recovery.exists():
        saved=torch.load(recovery,weights_only=False)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        start=saved['step']+1
        seconds,eval_seconds,io_seconds,peak=(saved[k] for k in ('seconds','eval_seconds','io_seconds','peak'))
        setup_seconds=saved.get('setup_seconds',0.)
        history,curves,completed=saved['history'],saved['curves'],saved['completed_snapshots']
        event(directory,'exact_resume',step=start-1)
    else:
        event(directory,'start',budget=budget,job=job)
    setup_seconds+=time.perf_counter()-job_started
    monitor=ProcessRSSMonitor();monitor.start();session_start=time.perf_counter()
    pre_seconds=source_metrics['training_seconds'] if source_metrics else 0.
    pre_eval=source_metrics['cumulative_evaluation_seconds'] if source_metrics else 0.
    pre_setup=source_metrics.get('setup_seconds',0.) if source_metrics else 0.
    pre_io=source_metrics.get('io_seconds',0.) if source_metrics else 0.
    pre_steps=(2 if smoke else 6000) if source_metrics else 0

    def counts():
        if free_job:
            number=sum(p.numel() for p in model.parameters())
            return dict(total_parameters=number,active_parameters=number,frozen_parameters=0,carrier_parameters=number)
        return parameter_counts(model)

    def save(step):
        nonlocal peak,io_seconds
        tick=time.perf_counter();peak=max(peak,int(monitor.snapshot()['rss_sampled_peak_bytes']))
        atomic_torch_save(recovery,dict(step=step,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(),job=job,configuration=cfg.as_dict(),seconds=seconds,eval_seconds=eval_seconds,
            io_seconds=io_seconds,setup_seconds=setup_seconds,peak=peak,history=history,curves=curves,completed_snapshots=completed,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()))
        pd.DataFrame(history).to_csv(directory/'loss_history.csv',index=False)
        pd.DataFrame(curves).to_csv(directory/'cost_curve.csv',index=False)
        io_seconds+=time.perf_counter()-tick

    def cost(step,elapsed):
        attributed_pre=step if free_job else pre_steps
        coupled_step=0 if free_job else step
        record=dict(training_seconds=seconds if free_job else seconds+pre_seconds,
            incurred_this_job_training_seconds=seconds,carrier_training_seconds=seconds if free_job else pre_seconds,
            coupled_training_seconds=0. if free_job else seconds,
            incurred_this_job_pretraining_seconds=seconds if free_job else 0.,
            attributed_carrier_evaluation_seconds=0. if free_job else pre_eval,
            evaluation_seconds=elapsed,cumulative_evaluation_seconds=eval_seconds,io_seconds=io_seconds,
            setup_seconds=setup_seconds,attributed_carrier_setup_seconds=pre_setup,attributed_carrier_io_seconds=pre_io,
            unique_incurred_pipeline_seconds=seconds+eval_seconds+io_seconds+setup_seconds,
            full_attributed_pipeline_seconds=seconds+pre_seconds+eval_seconds+pre_eval+io_seconds+pre_io+setup_seconds+pre_setup,
            executed_steps=step,carrier_steps=attributed_pre,coupled_steps=coupled_step,
            steps=attributed_pre+coupled_step,training_residual_batches=attributed_pre+coupled_step,
            training_carrier_unit_spinor_residual_points=2*attributed_pre*cfg.spectral_collocation_points,
            training_coupled_spinor_residual_points=coupled_step*cfg.spectral_collocation_points,
            training_spinor_residual_points=(2*attributed_pre+coupled_step)*cfg.spectral_collocation_points,
            training_complex_scalar_residual_evaluations=2*(2*attributed_pre+coupled_step)*cfg.spectral_collocation_points,
            training_norm_penalty_forward_spinor_points=coupled_step*cfg.spectral_collocation_points
                if not free_job and job['envelope_kind']=='norm_regularized' else 0,
            peak_rss_mib=peak/2**20,rss_measurement_method=monitor.snapshot()['rss_measurement_method'],
            run_elapsed_seconds_this_session=time.perf_counter()-session_start,
            analytic_prior=job['carrier_kind']=='exact',pretrain_source_sha256=source_sha,
            cpu_affinity_mask=config['cpu_affinity_mask'],**counts())
        return record

    def snapshot(step):
        nonlocal eval_seconds,io_seconds,peak
        target=directory/'snapshots'/f'{"free" if free_job else "coupled"}_{step:05d}'
        target.mkdir(parents=True,exist_ok=True)
        tick=time.perf_counter()
        write_json(target/'config.json',config|dict(snapshot_step=step))
        atomic_torch_save(target/'model_state.pt',dict(state_dict=model.state_dict(),job=job,configuration=cfg.as_dict(),
            executed_steps=step,steps=step+pre_steps,fixed_checkpoint=True,analytic_prior=job['carrier_kind']=='exact',
            initial_hashes=initial,time_stream_sha256=sampler.digest(),normalized_time_stream_sha256=sampler.digest(True),
            pretrain_source_sha256=source_sha))
        pd.DataFrame(history).to_csv(target/'loss_history.csv',index=False)
        atomic_torch_save(target/'recovery_state.pt',torch.load(recovery,weights_only=False))
        io_seconds+=time.perf_counter()-tick
        before=(random.getstate(),np.random.get_state()[1].copy(),torch.get_rng_state().clone(),sampler.generator.get_state().clone())
        tick=time.perf_counter()
        if smoke:
            ks=torch.as_tensor(cfg.wave_numbers,dtype=torch.float64).reshape(-1,1)
            ts=torch.tensor([0.,cfg.t_max/2,cfg.t_max],dtype=torch.float64)
            if free_job:
                state=model(ks.repeat(3,1),ts.repeat_interleave(cfg.mode_count).reshape(-1,1)).detach().numpy().reshape(3,cfg.mode_count,2)
                error=float(abs(state[0]-1).max())
            else:
                from .r9_evaluate import spectral_grid
                from .r9_models import compose
                state=spectral_grid(model,cfg,ts.numpy())
                error=float(abs(state[0]-compose(model.initial_coeff,ks,cfg.mass).detach().numpy()).max())
            metrics=dict(all_finite=bool(np.isfinite(state).all()),initial_condition_max_abs_error=error,
                engineering_only=True,accuracy_not_used=True)
            series=pd.DataFrame(dict(t=ts.numpy()));dense=dict(t=ts.numpy(),prediction=state);tables={}
        elif free_job:
            from .r13_phase_evaluate import evaluate_carrier
            metrics,series,dense,table=evaluate_carrier(model,cfg)
            tables={'phase_error_by_mode.csv':table}
        else:
            from .r13_evaluate import evaluate
            metrics,series,dense,tables=evaluate(model,cfg,full=step==budget)
        elapsed=time.perf_counter()-tick;eval_seconds+=elapsed
        if (before[0]!=random.getstate() or not np.array_equal(before[1],np.random.get_state()[1])
            or not torch.equal(before[2],torch.get_rng_state())
            or not torch.equal(before[3],sampler.generator.get_state())):
            raise RuntimeError('Independent evaluation altered training random stream')
        peak=max(peak,int(monitor.snapshot()['rss_sampled_peak_bytes']))
        metrics.update(experiment=EXPERIMENT,job=job,stage=job['stage'],case=job['case'],group=job.get('group',job['carrier_kind']),
            model=job.get('group',job['carrier_kind']),seed=job['seed'],smoke=smoke,fixed_endpoint=True,
            carrier_kind=job['carrier_kind'],envelope_kind=job.get('envelope_kind'),initial_hashes=initial,
            current_hashes=dict(carrier=tensor_digest(model.parameters())) if free_job else hashes(model),
            time_stream_sha256=sampler.digest(),normalized_time_stream_sha256=sampler.digest(True),
            evaluation_does_not_change_training_rng=True,**cost(step,elapsed))
        tick=time.perf_counter()
        series.to_csv(target/'time_series.csv',index=False);np.savez_compressed(target/'dense_grid.npz',**dense)
        for name,table in tables.items():table.to_csv(target/name,index=False)
        write_json(target/'metrics.json',metrics)
        io_seconds+=time.perf_counter()-tick
        curves.append({k:v for k,v in metrics.items() if isinstance(v,(str,int,float,bool)) or v is None})
        completed.append(step);save(step)
        if step==budget:
            # Write cost before metrics so a crash cannot falsely mark a partially copied run complete.
            write_json(directory/'execution_cost.json',cost(step,elapsed)|dict(
                pretraining_source_reused=bool(source),full_pipeline_cost_attributed=not free_job,
                unique_incurred_cost_uses_only_this_job=True,
                pretraining_active_parameters=source_metrics['active_parameters'] if source_metrics else (counts()['active_parameters'] if free_job else 0)))
            for name in ARTIFACTS:
                if name!='metrics.json':shutil.copyfile(target/name,directory/name)
            for name in tables:shutil.copyfile(target/name,directory/name)
            shutil.copyfile(target/'metrics.json',directory/'metrics.json')
            event(directory,'complete',steps=budget,job=job)
        print('CHECKPOINT',job['case'],job.get('group',job['carrier_kind']),job['seed'],step,'eval_seconds',round(elapsed,1),flush=True)

    try:
        if start-1 in checkpoints and (start-1 not in completed or start-1==budget):
            snapshot(start-1)
        for step in range(start,budget+1):
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            k,t=sampler.sample()
            if free_job:
                residual=free_carrier_residual(model,cfg,k,t)
                loss=residual.abs().square().mean()
                power=residual.detach().abs().square().reshape(32,cfg.mode_count,2,2).mean((0,2,3))
                terms=dict(pde_loss=float(loss.detach()),norm_penalty=0.)
            else:
                loss,power,terms=training_loss(model,cfg,k,t)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError('Missing or nonfinite active gradient')
            grad=math_norm(parameters);optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError('Nonfinite parameter after optimizer step')
            seconds+=time.perf_counter()-tick
            if step==1 or step%cfg.history_every==0 or step==budget:
                history.append(dict(step=step,phase='free' if free_job else 'coupled',objective=float(loss.detach()),
                    incurred_training_seconds=seconds,attributed_training_seconds=seconds+pre_seconds,
                    gradient_norm=grad,**terms,**{f'n_{n}_unweighted_component_mean_power':float(v) for n,v in zip(cfg.indices,power)}))
            if step==1 or step%cfg.log_every==0 or step==budget:
                print(job['case'],job.get('group',job['carrier_kind']),job['seed'],'step',step,'/',budget,
                    'loss',float(loss.detach()),'seconds',round(seconds,1),flush=True)
            if step%cfg.recovery_every==0 or step in checkpoints or step==stop_after:save(step)
            if step in checkpoints:snapshot(step)
            if step==stop_after:
                event(directory,'controlled_smoke_interruption',step=step)
                return dict(interrupted_at=step)
        return json.loads(final.read_text(encoding='utf-8'))
    except Exception as error:
        event(directory,'failure_preserved',error_type=type(error).__name__,message=str(error))
        if isinstance(error,FloatingPointError):
            atomic_torch_save(directory/'failed_state.pt',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                sampler=sampler.state_dict(),history=history,job=job))
            pd.DataFrame(history).to_csv(directory/'loss_history.csv',index=False)
            write_json(directory/'failure.json',dict(status='NUMERICAL_FAILURE',job=job,error=repr(error),
                successful_steps=len(sampler.trace)-1,fixed_endpoint_reached=False,no_selective_rerun=True))
        raise
    finally:
        monitor.stop()
