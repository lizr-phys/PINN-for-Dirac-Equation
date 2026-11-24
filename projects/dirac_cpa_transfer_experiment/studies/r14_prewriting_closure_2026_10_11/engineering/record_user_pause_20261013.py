"""Record an explicitly requested pause; never start training or verification jobs."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
import torch

BASE = Path(__file__).resolve().parents[1]
EVENT = BASE / 'runtime/user_pause_events/1791887266842631000'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def create_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.user_pause_write.tmp')
    create_json(temporary, value)
    temporary.replace(path)


def task_processes():
    found = []
    paths = {str(BASE / n).lower() for n in ('driver.py', 'postprocess_when_ready.py', 'keep_awake.py')}
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            info = process.info
            command = info['cmdline'] or []
            if 'python' not in (info['name'] or '').lower():
                continue
            if any(arg.lower() in paths for arg in command) or (
                'closure.train' in command and any(str(BASE).lower() in arg.lower() for arg in command)
            ):
                found.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def main():
    psutil.Process().cpu_affinity([6])
    torch.set_num_threads(1)
    assert not task_processes(), 'Stop actual task processes before recording a pause'
    stopped = read(EVENT / 'actual_stop_verification.json')
    assert stopped['remaining'] == [] and stopped['whole_goal_complete'] is False
    assert not (EVENT / 'pause_summary.json').exists(), 'Preserve the first pause receipt'

    frozen = read(BASE / 'frozen/source_manifest.json')
    references = read(BASE / 'frozen/reference_manifest.json')
    assert len(frozen['files']) == 204 and len(references) == 39
    for entry in frozen['files'] + references:
        assert sha(BASE / entry['path']) == entry['sha256'], entry['path']
    assert sha(BASE / 'frozen/design.json') == frozen['design_sha256']

    execution = read(BASE / 'runtime/selected_execution_plan.json')
    formal = [job for stage in ('core', 'extension')
              for seeds in execution[stage].values() for jobs in seeds.values() for job in jobs]
    assert len(formal) == len({job['directory'] for job in formal}) == 210
    attempts = []
    for job in formal:
        folder = BASE / job['directory']
        complete, failure = folder / 'complete.json', folder / 'failure.json'
        assert not (complete.exists() and failure.exists()), job['directory']
        state = 'INCOMPLETE'
        if complete.exists():
            record = read(complete)
            assert record['job'] == job and record['status'] == 'COMPLETE'
            assert record['step'] == job['budget'] and record['fixed_endpoint_reached']
            state = 'PRESCRIBED_TRAINING_ENDPOINT_REACHED'
        elif failure.exists():
            record = read(failure)
            assert record['job'] == job
            state = 'PRESERVED_NUMERICAL_OR_DEPENDENCY_FAILURE'
        attempts.append({'job': job, 'training_state': state,
                         'complete_sha256': sha(complete) if complete.exists() else None,
                         'failure_sha256': sha(failure) if failure.exists() else None})
    completed = sum(row['training_state'] == 'PRESCRIBED_TRAINING_ENDPOINT_REACHED' for row in attempts)
    failures = sum(row['training_state'] == 'PRESERVED_NUMERICAL_OR_DEPENDENCY_FAILURE' for row in attempts)
    assert completed == 168 and failures == 0
    development = list((BASE / 'development').rglob('complete.json'))
    assert len(development) == 24
    for path in development:
        value = read(path)
        assert value['status'] == 'COMPLETE' and value['fixed_endpoint_reached']
        assert value['step'] == value['job']['budget'] == 4000

    saved_workers = []
    recovery_rows = []
    pre_stop = read(EVENT / 'pre_stop_processes.json')
    for process in pre_stop['processes']:
        if process['role'] != 'worker':
            continue
        job = read(process['cmdline'][-1])
        folder = BASE / job['directory']
        recovery = folder / 'recovery_state.pt'
        value = torch.load(recovery, weights_only=False, map_location='cpu')
        required = ('state_dict', 'optimizer', 'sampler', 'python_rng', 'numpy_rng', 'torch_rng',
                    'step', 'history', 'completed', 'training_wall_seconds', 'training_cpu_seconds')
        assert all(key in value for key in required)
        assert value['job'] == job and 0 < value['step'] < job['budget']
        assert value['step'] % 500 == 0
        assert value['state_dict'] and value['optimizer']['state'] and value['optimizer']['param_groups']
        assert torch.is_tensor(value['torch_rng'])
        configuration = read(folder / 'config.json')
        assert configuration['job'] == job
        assert configuration['source_manifest_sha256'] == sha(BASE / 'frozen/source_manifest.json')
        destination = EVENT / 'preserved_checkpoints' / job['group'] / f"seed_{job['seed']}"
        destination.mkdir(parents=True, exist_ok=False)
        copied = {}
        for name in ('recovery_state.pt', 'config.json', 'progress.json'):
            source = folder / name
            target = destination / name
            shutil.copyfile(source, target)
            assert sha(target) == sha(source)
            copied[name] = {'original_path': source.relative_to(BASE).as_posix(),
                            'preserved_path': target.relative_to(BASE).as_posix(),
                            'sha256': sha(target), 'bytes': target.stat().st_size}
        saved_workers.append({'pid_at_pause': process['pid'], 'job': job, 'saved_step': value['step'],
                              'preserved_files': copied, 'Adam_weights_sampler_and_all_RNG_fields_present': True,
                              'checkpoint_load_verified': True, 'training_endpoint_complete': False,
                              'resume_execution_not_performed': True})
        recovery_rows.append({'case': job['case'], 'group': job['group'], 'seed': job['seed'],
                              'directory': job['directory'], 'saved_step': value['step'],
                              'uncheckpointed_extra_steps_lower_bound': 0,
                              'uncheckpointed_extra_steps_upper_bound': 500,
                              'unobserved_extra_training_wall_seconds': '',
                              'unobserved_extra_training_CPU_seconds': '',
                              'unknown_not_zero': True,
                              'reason': 'Explicit user pause; last periodic exact checkpoint retained',
                              'attribution_scope': 'Additional to all earlier preserved recovery cost supplements'})
        del value
    assert len(saved_workers) == 5
    supplement = EVENT / 'unknown_replay_costs_user_pause.csv'
    with supplement.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recovery_rows[0]))
        writer.writeheader()
        writer.writerows(recovery_rows)

    automation_path = Path(r'D:\Codex\home\automations\r14\automation.toml')
    automation = tomllib.loads(automation_path.read_text(encoding='utf-8'))
    assert automation['id'] == 'r14' and automation['kind'] == 'heartbeat'
    assert automation['status'] == 'PAUSED'
    shutil.copyfile(automation_path, EVENT / 'automation_paused.toml')
    with (BASE / 'analysis_outputs/independent_metrics_per_snapshot.csv').open(encoding='utf-8', newline='') as stream:
        independent = list(csv.DictReader(stream))
    audited = {Path(row['path']).as_posix() for row in independent}
    raw = {path.relative_to(BASE).as_posix() for root in ('formal/coupled', 'development')
           for path in (BASE / root).rglob('metrics.json')}
    pending = sorted(raw - audited)
    metric = read(BASE / 'analysis_outputs/independent_metric_audit.json')
    residual = read(BASE / 'analysis_outputs/residual_audit.json')
    inference = read(BASE / 'isolated_inference/independent_audit.json')
    now = time.time()
    local = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    summary = {'paused_utc': now, 'paused_local_time': local, 'status': 'PAUSED_BY_USER',
               'explicit_user_request': '停下目前的工作，向我汇报完成的任务',
               'formal_training_endpoints': completed, 'formal_planned': 210,
               'development_training_endpoints': 24, 'development_planned': 24,
               'complete_core_cases': ['T10', 'm10', 'm5T5'],
               'core_formal_endpoints': 165, 'independently_verified_core_seed_endpoints': 195,
               'extension_training_endpoints': completed - 165, 'extension_planned': 45,
               'preserved_numerical_or_dependency_failures': failures,
               'accuracy_failures_retained_in_original_tables': True,
               'independent_neural_snapshots': metric['snapshots_audited'],
               'independent_residual_endpoints': residual['trained_endpoints_checked'],
               'isolated_inference_measurements': inference['actual_measurements_checked'],
               'pending_new_neural_metric_sources': pending,
               'saved_workers': saved_workers, 'actual_remaining_task_processes': task_processes(),
               'automation_status_verified_from_file': 'PAUSED',
               'current_goal_tool_lookup': 'get_goal returned goal=null in this user-request turn; no completion or duplicate goal created',
               'source_files_actual_sha_verified': 204, 'reference_files_actual_sha_verified': 39,
               'source_manifest_sha256': sha(BASE / 'frozen/source_manifest.json'),
               'reference_manifest_sha256': sha(BASE / 'frozen/reference_manifest.json'),
               'design_sha256': sha(BASE / 'frozen/design.json'),
               'pause_reader_sha256': sha(Path(__file__)),
               'unknown_replay_supplement': supplement.relative_to(BASE).as_posix(),
               'unknown_replay_supplement_sha256': sha(supplement),
               'no_new_training_or_scientific_validation_executed_for_this_report': True,
               'whole_goal_complete': False, 'resume_requires_explicit_user_instruction': True}
    assert not summary['actual_remaining_task_processes']
    create_json(EVENT / 'formal_training_endpoint_inventory.json', attempts)
    create_json(EVENT / 'pause_summary.json', summary)
    create_json(BASE / 'runtime/user_pause_request.json', {
        'status': 'PAUSED_BY_USER', 'paused_utc': now, 'user_request': summary['explicit_user_request'],
        'pause_event': EVENT.relative_to(BASE).as_posix(), 'resume_requires_explicit_user_instruction': True,
        'frozen_plan_and_all_evidence_retained': True, 'whole_goal_complete': False})

    progress = read(BASE / 'progress.json')
    progress.update(status='PAUSED_BY_USER', active_children=[], updated_utc=now, paused_utc=now,
                    formal_complete=completed, development_complete=24,
                    paused_saved_workers=[{'job': row['job'], 'saved_step': row['saved_step']} for row in saved_workers],
                    next_step='等待用户明确恢复；当前停止训练和验证，保留原Adam、权重、RNG及采样断点',
                    user_pause_event=EVENT.relative_to(BASE).as_posix(), whole_goal_complete=False)
    replace_json(BASE / 'progress.json', progress)
    (BASE / 'progress.md').write_bytes((
        f'# R14 当前进度\n\n北京时间：{local}\n\n状态：PAUSED_BY_USER（用户要求停止）。\n\n'
        f'开发训练终点24/24；正式训练终点{completed}/210，其中三核心165/165、扩展3/45。\n\n'
        '三核心195个规定逐种子观测已独立闭环；新增扩展终点不等于全部独立验证完成。'
        f'当前独立神经快照{metric["snapshots_audited"]}、残差观测{residual["trained_endpoints_checked"]}、'
        f'隔离进程推理{inference["actual_measurements_checked"]}/255。\n\n'
        '任务驱动器、五个worker、后处理助手及临时防休眠进程已全部停止；r14自动续办已暂停。'
        '五个恢复状态已实际加载核验并封存，保留未落盘额外工作的未知成本，不填零。\n\n'
        '仅用户明确恢复后继续；整轮证据包和18项验收尚未完成。\n\n'
        '[暂停清点](runtime/user_pause_events/1791887266842631000/pause_summary.json) · '
        '[冻结方案](preregistration.md) · [完整协议](frozen/design.json)\n'
    ).encode('utf-8'))
    observation = read(BASE / 'runtime/continuation_observation.json')
    observation.update(status='PAUSED_BY_USER', observed_utc=now, local_time=local,
                       formal_complete=completed, workers=[], actual_task_processes=[], helpers=[],
                       automatic_continuation_paused=True, explicit_user_pause=True,
                       pause_event=EVENT.relative_to(BASE).as_posix(), goal_complete=False,
                       continued_under_latest_user_heartbeat=False, resume_requires_explicit_user_instruction=True,
                       pending_new_neural_metrics=pending)
    replace_json(BASE / 'runtime/continuation_observation.json', observation)
    for relative in ('runtime/postprocessing_progress.json', 'runtime/awake_guard_status.json'):
        value = read(BASE / relative)
        value.update(updated_utc=now, status='PAUSED_BY_USER', active=False,
                     system_idle_sleep_request=False, active_training_pids=[], explicit_user_pause=True,
                     next_step='等待用户明确恢复；当前没有活跃训练或后处理进程')
        replace_json(BASE / relative, value)
    continuation = (BASE / 'CONTINUE.md').read_text(encoding='utf-8-sig').replace('\r', '')
    heading, rest = continuation.split('\n', 1)
    notice = (f'\n**用户暂停优先（{local}）**：用户明确要求“停下目前的工作，向我汇报完成的任务”。'
              '驱动器、五个worker、后处理和临时防休眠进程已停止；r14自动续办已暂停。'
              '禁止按旧心跳或下方历史续办指令重新启动。仅用户明确恢复后，按封存原Adam、权重、RNG和采样断点续办。'
              '科学计划及历史证据全部保留；整轮未完成、未验收。'
              '暂停实物见runtime/user_pause_events/1791887266842631000/pause_summary.json。\n')
    continuation = heading + '\n' + notice + rest
    (BASE / 'CONTINUE.md').write_bytes(continuation.encode('utf-8'))
    observation['CONTINUE_sha256'] = sha(BASE / 'CONTINUE.md')
    replace_json(BASE / 'runtime/continuation_observation.json', observation)
    print(json.dumps({'status': summary['status'], 'formal': completed, 'development': 24,
                      'saved_steps': [{'group': row['job']['group'], 'seed': row['job']['seed'],
                                       'step': row['saved_step']} for row in saved_workers],
                      'pending_new_metrics': len(pending), 'automatic_continuation': 'PAUSED',
                      'remaining_task_processes': 0, 'whole_goal_complete': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
