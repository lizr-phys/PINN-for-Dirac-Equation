from __future__ import annotations

"""Operational restart of the frozen R13 tail after a system reboot.

No models, losses, checkpoints, selection rules, or optimizer budgets change.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

from .r13_config import ROOT, RAW, LOGS, VALIDATION_SEEDS, CURVE_STEPS
from .r13_plan import jobs_for, directory_for
from .freeze_r7_phase_release import write_json, sha256
from .freeze_r13_recovery import verify_freeze


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def utc():
    return datetime.now(timezone.utc).isoformat()


def active_r13_processes():
    # A PID can be reused after reboot. Verify the executable and module too.
    query = """@(Get-CimInstance Win32_Process | Where-Object {
      $_.Name -match '^python(?:w)?\\.exe$' -and
      $_.CommandLine -match '-m\\s+src\\.(run_r13_recovery|launch_r13_detached|resume_r13_remaining)(\\s|$)'
    } | Select-Object ProcessId, CreationDate, CommandLine) | ConvertTo-Json -Compress"""
    output = subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-Command', query], text=True, encoding='utf-8')
    rows = json.loads(output or '[]')
    if isinstance(rows, dict):
        rows = [rows]
    return [r for r in rows if int(r['ProcessId']) != os.getpid()]


def compact_completed_copies():
    frozen = verify_freeze()
    if active_r13_processes():
        raise RuntimeError('R13 processes still active; compaction refused')
    manifest = json.loads((RAW/'storage_duplicate_candidates.json').read_text(encoding='utf-8'))
    before = shutil.disk_usage(ROOT).free
    records = []
    for row in manifest['candidates']:
        source, destination = ((ROOT/row[k]).resolve() for k in ('source', 'destination'))
        for path in (source, destination):
            path.relative_to((RAW/'formal').resolve())
        if source.parents[2] != destination.parent or source.parent.parent.name != 'snapshots':
            raise RuntimeError('Unexpected duplicate path')
        if not (destination.parent/'execution_cost.json').is_file():
            raise RuntimeError('Only immutable completed jobs can be compacted')
        if os.path.samefile(source, destination):
            records.append(row | dict(status='ALREADY_SHARED'))
            continue
        digest = sha256(source)
        if sha256(destination) != digest:
            records.append(row | dict(status='DIFFERENT_BYTES_PRESERVED'))
            continue
        temporary = destination.with_name(destination.name+'.r13_link_pending')
        if temporary.exists():
            raise RuntimeError('Unexpected temporary link; inspect before proceeding')
        os.link(source, temporary)
        os.replace(temporary, destination)
        if not os.path.samefile(source, destination) or sha256(destination) != digest:
            raise RuntimeError('Hard-link integrity verification failed')
        records.append(row | dict(status='IDENTICAL_BYTES_SHARED', sha256=digest))
    after = shutil.disk_usage(ROOT).free
    record = dict(created_utc=utc(), scientific_sha256=frozen,
                  method='NTFS hard links between byte-identical completed root and final snapshot files',
                  every_original_path_preserved=True, all_contents_unchanged=True,
                  historical_rounds_untouched=True, free_bytes_before=before, free_bytes_after=after,
                  recovered_bytes=after-before, files=records)
    target = RAW/'storage_history'/f'{stamp()}.json'
    target.parent.mkdir(exist_ok=True)
    write_json(target, record)
    write_json(RAW/'storage_compaction.json', record)
    print(json.dumps({k:v for k,v in record.items() if k != 'files'}, ensure_ascii=False), flush=True)


def tail_storage_gate(pending):
    import torch
    sample_sizes = [sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
                    for path in (RAW/'formal/coupled').glob('*/m10/*/seed_*/snapshots/coupled_20000')]
    if not sample_sizes:
        raise RuntimeError('No completed m10 storage sample')
    per_snapshot = max(sample_sizes)*1.25
    future_snapshots = 0
    states = []
    for job in pending:
        directory = directory_for(job)
        path = directory/'recovery_state.pt'
        step = int(torch.load(path, weights_only=False)['step']) if path.exists() else 0
        future_snapshots += sum(s > step for s in CURVE_STEPS)
        states.append(dict(job=job, restored_step=step,
                           recovery_sha256=sha256(path) if path.exists() else None))
    # Include duplicated final root exports, a 2x uncertainty margin and 4 GiB
    # for plots, source CSVs, manifests, recovery files, and postprocessing.
    estimated_exports = (future_snapshots+len(pending))*per_snapshot
    required = 2*estimated_exports+4*2**30
    free = shutil.disk_usage(ROOT).free
    record = dict(created_utc=utc(), original_full_campaign_reserve_gib=25,
                  remaining_jobs=len(pending), remaining_fixed_snapshots=future_snapshots,
                  maximum_observed_m10_snapshot_bytes=max(sample_sizes),
                  forecast_snapshot_size_margin=1.25, export_uncertainty_multiplier=2,
                  postprocessing_and_recovery_reserve_gib=4,
                  required_tail_free_bytes=required, observed_free_bytes=free,
                  pass_check=free >= required, recovery_states=states,
                  operational_resource_amendment_only=True, no_accuracy_based_budget_change=True,
                  no_scientific_code_change=True, no_history_deletion=True)
    write_json(RAW/'remaining_resource_gate.json', record)
    if not record['pass_check']:
        raise RuntimeError('Insufficient storage even for the measured remaining tail')
    return record


def launch_tail():
    from .r7_runtime import physical_core_masks, set_cpu_affinity
    from .run_r9_wavepacket import validate_masks
    frozen = verify_freeze()
    if active_r13_processes():
        raise RuntimeError('Verified existing R13 module process; overlapping launch refused')
    jobs = [j for seed in VALIDATION_SEEDS for j in jobs_for('independent_validation', seed)]
    pending = [j for j in jobs if not (directory_for(j)/'metrics.json').exists()
               and not (directory_for(j)/'failure.json').exists()]
    if any(j['type'] != 'coupled' or j['case'] != 'm10' for j in pending):
        raise RuntimeError('Unexpected unfinished job outside the verified m10 tail')
    gate = tail_storage_gate(pending)
    plan_path = RAW/'runtime_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    cores = physical_core_masks()
    masks, control = plan['masks'], plan['control_mask']
    if any(mask not in cores for mask in masks+[control]):
        raise RuntimeError('Physical core layout changed')
    validate_masks(masks+[control])
    set_cpu_affinity(control)
    LOGS.mkdir(parents=True, exist_ok=True)
    sys.stdout = (LOGS/'controller_stdout.log').open('a', encoding='utf-8', buffering=1)
    sys.stderr = (LOGS/'controller_stderr.log').open('a', encoding='utf-8', buffering=1)
    history = RAW/'runtime_history'
    history.mkdir(exist_ok=True)
    shutil.copyfile(plan_path, history/f'{stamp()}_before_tail.json')
    for batch in plan['batches']:
        if batch['status'] == 'RUNNING':
            batch['status'] = 'INTERRUPTED_AFTER_SYSTEM_RESTART'
        for slot in batch['slots']:
            slot.setdefault('previous_pid', slot.get('pid'))
            slot['pid'] = 0
    plan.update(status='RUNNING', driver_pid=os.getpid(), resumed_utc=utc(),
                operational_launcher='src.resume_r13_remaining',
                remaining_storage_gate=gate, freeze_manifest_sha256=frozen,
                exact_recovery=True, no_selective_new_initialization=True)
    batch = dict(stage='independent_validation', status='RUNNING', started_utc=utc(),
                 restart_after_reboot=True, slots=[])
    plan['batches'].append(batch)
    processes = []
    for slot, (seed, mask) in enumerate(zip(VALIDATION_SEEDS, masks)):
        log = LOGS/f'independent_validation_slot_{slot}_seed{seed}.log'
        stream = log.open('a', encoding='utf-8')
        command = [sys.executable, '-X', 'utf8', '-u', '-m', 'src.run_r13_recovery',
                   'worker', '--stage', 'independent_validation', '--seed', str(seed), '--affinity', str(mask)]
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdout=stream,
                                 stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        batch['slots'].append(dict(stage='independent_validation', slot=slot, seed=seed, mask=mask,
                                   pid=child.pid, log=log.relative_to(ROOT).as_posix(),
                                   jobs=jobs_for('independent_validation', seed)))
        processes.append((child, stream, seed))
    write_json(plan_path, plan)
    write_json(RAW/'tail_resume_launch.json', dict(created_utc=utc(), driver_pid=os.getpid(),
               scientific_sha256=frozen, actual_process_identity_checked=True,
               disjoint_masks=masks+[control], remaining_jobs=len(pending),
               resource_gate='remaining_resource_gate.json', no_budget_change=True))
    print('R13 remaining tail resumed', len(pending), 'fixed-budget jobs', flush=True)
    while any(p.poll() is None for p, _, _ in processes):
        time.sleep(.25)
    errors = []
    for child, stream, seed in processes:
        stream.close()
        if child.returncode:
            errors.append(dict(seed=seed, exit_code=child.returncode))
    batch.update(status='FAIL' if errors else 'COMPLETE', errors=errors, ended_utc=utc())
    plan.update(status='INTERRUPTED_OR_IMPLEMENTATION_FAILURE' if errors else 'TRAINING_COMPLETE')
    write_json(plan_path, plan)
    if errors:
        raise RuntimeError('Worker error preserved: '+str(errors))
    from .finalize_r13_recovery import finalize
    try:
        result = finalize()
    except Exception as error:
        plan.update(status='POSTPROCESSING_NEEDS_ATTENTION', error=repr(error), ended_utc=utc())
        write_json(plan_path, plan)
        raise
    plan.update(status='COMPLETE' if result == 'PASS' else 'TRAINING_COMPLETE_VISUAL_REVIEW_PENDING',
                ended_utc=utc())
    write_json(plan_path, plan)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('compact', 'launch'):
        raise SystemExit('Use compact or launch')
    compact_completed_copies() if sys.argv[1] == 'compact' else launch_tail()
