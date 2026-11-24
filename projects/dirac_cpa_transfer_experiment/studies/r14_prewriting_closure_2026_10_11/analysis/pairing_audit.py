"""Verify actual initializer bytes and private sampling traces, without training."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
import torch
from plan import BASE,CORE,SEEDS,write,sha

def tensor_digest(values):
    h=hashlib.sha256()
    for value in values:
        a=value.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def random_trace(seed,steps,batch,offset):
    g=torch.Generator().manual_seed(offset+seed)
    out=np.empty((steps,batch,1),dtype=np.float64)
    for i in range(steps):
        out[i]=np.sort(torch.rand((batch,1),generator=g,dtype=torch.float64).numpy(),axis=0)
    return out

def main():
    d=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    steps=d['mechanism']['coupled_primary_steps'];batch=d['mechanism']['fresh_sorted_uniform_times']
    offset=d['mechanism']['sampling_streams']['coupled'];groups=tuple(d['mechanism_groups'])
    expected={s:random_trace(s,steps,batch,offset) for s in SEEDS}
    rows=[];missing=[];blocks=[];bad=[];initial_checks=[];trace_checks=[]
    for case in CORE:
        duration=10. if case=='T10' else 5.
        for seed in SEEDS:
            inits={};streams={};failed=[];pending=[]
            uniform=expected[seed];physical=duration*uniform
            ush=hashlib.sha256(uniform.tobytes()).hexdigest();tsh=hashlib.sha256(physical.tobytes()).hexdigest()
            for group in groups:
                folder=BASE/'formal/coupled'/case/group/f'seed_{seed}'
                confpath=folder/'config.json';ip=folder/'initial_state.pt'
                cp=folder/'snapshots/20000/model_state.pt';fp=folder/'failure.json'
                conf=json.loads(confpath.read_text(encoding='utf-8')) if confpath.exists() else None
                failure=json.loads(fp.read_text(encoding='utf-8')) if fp.exists() else None
                if conf is not None and ip.exists():
                    initial=torch.load(ip,weights_only=False)
                    tensors={k:v for k,v in initial['state_dict'].items() if k.startswith('networks.0.') and k.endswith(('.weight','.bias'))}
                    computed=tensor_digest(tensors.values())
                    passed=bool(tensors) and computed==conf['initial_hash']['envelope']==initial['initial_hash']['envelope']
                    info=dict(case=case,seed=seed,group=group,path=ip.relative_to(BASE).as_posix(),
                        initial_file_sha256=sha(ip),computed_envelope_sha256=computed,
                        parameter_count=sum(v.numel() for v in tensors.values()),pass_check=passed)
                    initial_checks.append(info)
                    if not passed:bad.append(info)
                    inits[group]=(computed,{k:(tuple(v.shape),str(v.dtype),v.contiguous().numpy().tobytes()) for k,v in tensors.items()},passed)
                if conf is None or not cp.exists():
                    if failure and (failure['status']=='DEPENDENCY_FAILED' or
                        (failure['status']=='NUMERICAL_FAILURE' and failure.get('occurred_step',0)<=steps)):
                        failed.append(group)
                    else:
                        pending.append(group);missing.append(folder.relative_to(BASE).as_posix())
                    continue
                # Terminal markers follow the final recovery write. Do not open
                # live recovery files while their Windows atomic replacement runs.
                if not (folder/'complete.json').exists() and failure is None:
                    pending.append(group);missing.append(folder.relative_to(BASE).as_posix()+'#terminal_trace_pending');continue
                rp=folder/'recovery_state.pt'
                if not rp.exists():
                    pending.append(group);missing.append(rp.relative_to(BASE).as_posix());continue
                checkpoint=torch.load(cp,weights_only=False);recovery=torch.load(rp,weights_only=False)
                u=recovery['sampler']['normalized_trace'][:steps].contiguous().numpy()
                t=recovery['sampler']['trace'][:steps].contiguous().numpy()
                actual_u=hashlib.sha256(u.tobytes()).hexdigest();actual_t=hashlib.sha256(t.tobytes()).hexdigest()
                passed=u.shape==uniform.shape and t.shape==physical.shape and np.array_equal(u,uniform) and np.array_equal(t,physical)
                passed=passed and actual_u==checkpoint['normalized_time_stream_sha256']==ush and actual_t==checkpoint['time_stream_sha256']==tsh
                passed=passed and checkpoint['step']==steps and checkpoint['job']==recovery['job']==conf['job']
                info=dict(case=case,seed=seed,group=group,checkpoint=cp.relative_to(BASE).as_posix(),
                    checkpoint_sha256=sha(cp),terminal_recovery=rp.relative_to(BASE).as_posix(),
                    recovery_sha256=sha(rp),recovery_saved_step=recovery['step'],prefix_steps_checked=steps,
                    actual_normalized_prefix_sha256=actual_u,declared_seed_replay_sha256=ush,
                    actual_physical_prefix_sha256=actual_t,declared_physical_replay_sha256=tsh,
                    actual_saved_trace_matches_independent_replay=bool(passed),pass_check=bool(passed))
                trace_checks.append(info);streams[group]=info
                if not passed:bad.append(info)
            block=dict(case=case,seed=seed,actual_initializer_groups=len(inits),actual_stable_20000_trace_groups=len(streams),
                scientific_failure_groups=failed,missing_or_unexplained_groups=pending,
                status='INCOMPLETE' if pending else 'SCIENTIFIC_FAILURE_PREVENTS_FULL_BITWISE_BLOCK')
            if len(inits)==len(streams)==7:
                first=next(iter(inits.values()))
                matched=all(v==first and v[2] for v in inits.values()) and all(v['pass_check'] for v in streams.values())
                block['status']='PASS_FULL_BITWISE_BLOCK' if matched else 'MISMATCH'
                if not matched:bad.append(block)
                else:rows.append(dict(case=case,seed=seed,all_seven_envelope_initializations_bitwise_paired=True,
                    actual_saved_weight_bytes_compared=True,all_seven_20000_time_streams_identical=True,
                    actual_saved_traces_replayed_from_declared_private_seed=True,envelope_sha256=first[0],time_sha256=ush))
            blocks.append(block)
    accounted=sum(b['status'] in ('PASS_FULL_BITWISE_BLOCK','SCIENTIFIC_FAILURE_PREVENTS_FULL_BITWISE_BLOCK') for b in blocks)
    report=dict(updated_utc=time.time(),status='FAIL' if bad else ('ALL_PRESCRIBED_BLOCKS_ACCOUNTED' if accounted==15 else 'PASS_AVAILABLE_ACTUAL_INITIALIZERS_AND_TERMINAL_TRACES'),
        seed_blocks=len(rows),blocks_accounted=accounted,checks=rows,prescribed_block_status=blocks,missing=missing,mismatches=bad,
        actual_initializers_checked=len(initial_checks),actual_terminal_trace_prefixes_checked=len(trace_checks),
        initializer_checks=initial_checks,stream_replay_checks=trace_checks,source_code_sha256=sha(__file__),
        current_workers_recovery_files_not_opened=True,training_and_optimizer_states_not_written=True,
        entire_study_pairing_complete=len(rows)==15 and not bad,all_prescribed_blocks_accounted=accounted==15 and not bad,
        scientific_failure_does_not_imply_bitwise_pairing_verified=True)
    write(BASE/'analysis_outputs/pairing_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','prescribed_block_status','missing','initializer_checks','stream_replay_checks','mismatches')},ensure_ascii=False))
    if bad:
        print(json.dumps(bad[:3],ensure_ascii=False));raise RuntimeError('Actual mechanism pairing differs from frozen design')

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6]);torch.set_num_threads(1);main()
