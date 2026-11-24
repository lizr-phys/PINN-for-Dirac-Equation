from __future__ import annotations

"""Write an explicitly interim status report without changing experiment data."""
import json
import shutil
from datetime import datetime, timezone, timedelta
import pandas as pd
import torch
from .r13_config import ROOT, RAW, DOCS, THRESHOLDS
from .aggregate_r13_recovery import collect, summaries
from .r13_plan import directory_for
from .freeze_r7_phase_release import write_json
from .resume_r13_remaining import active_r13_processes


def update():
    instant = datetime.now(timezone.utc)
    stamp = instant.strftime('%Y%m%dT%H%M%S%fZ')
    directory = RAW/'progress_history'/stamp
    directory.mkdir(parents=True)
    coupled, carriers, _, _, attempts = collect()
    unique = coupled[~coupled.alias].copy()
    unique['joint_pass'] = unique.apply(lambda row: all(
        row[key] >= value if key == 'fidelity_mean' else row[key] <= value
        for key, value in THRESHOLDS.items()), axis=1)
    unique.to_csv(directory/'completed_coupled_per_seed.csv', index=False)
    carriers.to_csv(directory/'completed_carriers_per_seed.csv', index=False)
    summary = summaries(unique, tuple(THRESHOLDS))
    summary.to_csv(directory/'completed_group_summary.csv', index=False)
    independent = unique[unique.stage == 'independent_validation']
    independent.to_csv(directory/'independent_completed_per_seed.csv', index=False)
    remaining = []
    for job in attempts:
        if job['status'] != 'INCOMPLETE':
            continue
        path = ROOT/job['directory']/'recovery_state.pt'
        state = torch.load(path, weights_only=False) if path.exists() else None
        remaining.append(dict(stage=job['stage'], case=job['case'], group=job.get('group'),
                              seed=job['seed'], restored_or_saved_step=int(state['step']) if state else 0,
                              budget=6000 if job['type'] == 'carrier' else 20000))
    processes = active_r13_processes()
    plan = json.loads((RAW/'runtime_plan.json').read_text(encoding='utf-8'))
    record = dict(created_utc=instant.isoformat(), completed_carriers=len(carriers),
                  completed_unique_coupled=len(unique), total_carriers=41, total_coupled=66,
                  remaining_jobs=remaining, actual_module_processes=processes,
                  runtime_plan_status=plan['status'], not_final_report=True)
    write_json(directory/'status.json', record)
    write_json(RAW/'progress_snapshot.json', record)
    report = ROOT/'R13_LONG_TIME_RECOVERY_REPORT.md'
    archives = DOCS/'report_history'
    archives.mkdir(parents=True, exist_ok=True)
    if report.exists():
        shutil.copyfile(report, archives/f'{stamp}_before_progress_update.md')
    validation = summary[(summary.stage == 'independent_validation') & (summary.case == 'T10')]
    table = validation[['group', 'n', 'spinor_rel_l2_gauge_aligned_mean', 'current_rel_l2_mean',
                        'fidelity_mean_mean', 'modal_population_rel_l2_mean',
                        'max_norm_drift_mean', 'joint_pass_count', 'absolute_success']].copy()
    table.columns = ['组别', '种子数', '旋量误差均值', '概率流误差均值', '保真度均值',
                     '模态占据误差均值', '最大范数漂移均值', '联合通过数', '绝对成功']
    partial = independent[independent.case == 'm10'][['group', 'seed']+list(THRESHOLDS)+['joint_pass']]
    gate_path = RAW/'remaining_resource_gate.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else None
    compaction_path = RAW/'storage_compaction.json'
    compaction = json.loads(compaction_path.read_text(encoding='utf-8')) if compaction_path.exists() else None
    resource = ('完成文件逐字节SHA-256相同的根目录与最终快照改用NTFS硬链接，保留全部路径和内容，'
                f'释放约{compaction["recovered_bytes"]/2**30:.2f} GiB；' if compaction else '')
    if gate:
        resource += (f'剩余工作存储门槛按已完成m10快照最大体积、1.25倍快照余量、2倍导出余量及4 GiB后处理余量重算，'
                     f'要求{gate["required_tail_free_bytes"]/2**30:.2f} GiB，核查可用{gate["observed_free_bytes"]/2**30:.2f} GiB。'
                     '这是整轮25 GiB初始资源估算的运行层修订，科学代码、种子、损失、选择规则和预算不变。')
    body = f'''# R13 长时间 Dirac 演化：运行进度

更新时间（北京时间）：{instant.astimezone(timezone(timedelta(hours=8))).isoformat()}。

**尚未全部完成，当前不是最终报告。** 载波预训练{len(carriers)}/41、耦合训练{len(unique)}/66已完成；共{len(carriers)+len(unique)}/107个唯一训练任务已有固定终点。桥接、自由载波、精确E包络对照、T10四组消融已全部完成。独立验证已完成{len(independent)}/30；剩余{len(remaining)}个均为m=10耦合run。

## T=10独立验证完整结果

{table.to_markdown(index=False, floatfmt='.7g')}

选定组合为自由残差学习的static_frequency载波＋hard_total_norm包络。五项均值全部满足原门槛且5/5种子联合通过，独立复现了T=10修复效果。原A未通过，原E的主要限制仍是范数漂移。硬范数不替代动力学检验；组合的旋量、概率流、保真度和模态指标也全部通过。

这尚不能建立学习载波相对精确载波的实际价值：T10探索中，精确E＋同样硬范数包络的平均旋量误差约1.67%，组合约1.70%，E省去6000步载波预训练。该同包络比较仅有探索种子600–602；独立验证的E使用原PDE-only包络，解释范围不同。

## m=10独立验证已完成部分

{partial.to_markdown(index=False, floatfmt='.7g')}

各组仅完成3/5，不能替代五种子最终判定。当前组合已完成的3个种子均通过，原E已完成的3个均通过，原A已完成的3个均未通过。

## 恢复与剩余工作

电脑于2026-10-10 13:49北京时间重启。训练进程随后已消失；今天凌晨00:48–00:49的Adam、模型、采样和RNG断点完整保留。最初恢复被整轮25 GiB磁盘预检阻止，未发生选择性重跑或预算追加。{resource}

当前实际核查到{len(processes)}个R13控制/训练进程，运行记录状态为{plan['status']}。

{pd.DataFrame(remaining).to_markdown(index=False)}

剩余固定终点完成后生成正式统计、全部种子图、成本曲线、完整pytest、历史保护审计和链接检查。目前最终图表及审计尚未完成。

[本次进度源数据]({directory.relative_to(ROOT).as_posix()}) · [冻结协议](docs/experiments/r13_long_time_recovery/preregistration.md) · [科学代码SHA-256](results/raw/r13_long_time_recovery/frozen_protocol_and_code.json) · [运行记录](results/raw/r13_long_time_recovery/runtime_plan.json) · [资源修订](results/raw/r13_long_time_recovery/remaining_resource_gate.json)。

历史结果及全部失败终点保留。本轮只说明既定有限Fourier高斯初谱的两个条件下的修复证据，不宣称一般Dirac求解器、未知连续谱发现或数学不可表示。
'''
    report.write_text(body, encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k != 'actual_module_processes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    update()
