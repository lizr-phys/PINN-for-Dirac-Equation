"""Archive the first verified m5T5 core case, without changing scientific files."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'analysis_outputs/core_case_receipts/m5T5_first_complete'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def rows(rel):
    with (ROOT / rel).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    assert not DEST.exists(), 'Preserve an existing first archive; do not overwrite it'
    attempts = load('analysis_outputs/attempt_completion_audit.json')
    jobs = [r for r in attempts['rows'] if r['case'] == 'm5T5' and r['stage'] == 'core']
    assert len(jobs) == 55 and len({r['directory'] for r in jobs}) == 55
    assert all(r['state'] == 'COMPLETE_VERIFIED' and not r['issues'] for r in jobs)
    assert sum(r['type'] == 'carrier' for r in jobs) == 10
    metric = load('analysis_outputs/independent_metric_audit.json')
    cost = load('analysis_outputs/independent_cost_audit.json')
    stats = load('analysis_outputs/independent_statistical_audit.json')
    residual = load('analysis_outputs/residual_audit.json')
    residual_source = load('analysis_outputs/residual_provenance_audit.json')
    inference_audit = load('isolated_inference/independent_audit.json')
    parameter = load('analysis_outputs/independent_parameter_audit.json')
    assert metric['status'] == 'PASS_AVAILABLE_SNAPSHOTS' and not metric['metric_mismatches'] and not metric['missing']
    assert cost['status'] == 'PASS_AVAILABLE_COST_RECORDS' and not cost['mismatches']
    assert stats['status'] == 'PASS_AVAILABLE_STATISTICAL_TABLES' and not stats['errors']
    assert residual['status'] == 'PASS_AVAILABLE_FIXED_ENDPOINTS' and not residual['missing']
    assert residual_source['status'].startswith('PASS')
    assert inference_audit['status'] == 'PASS_AVAILABLE_INFERENCE_RECORDS'
    assert parameter['status'] == 'PASS_AVAILABLE_PARAMETER_RECORDS' and not parameter['errors']
    for name, expected in stats['source_table_sha256'].items():
        assert digest(ROOT / 'analysis_outputs' / name) == expected, name
    summary = [r for r in rows('analysis_outputs/summary_metrics.csv')
               if r['case'] == 'm5T5' and r['endpoint'] in ('primary', 'terminal')]
    assert len(summary) == 13 and all(r['all_five_present'] == 'True' and r['independent_metric_audit_complete'] == 'True' for r in summary)
    assert len({(r['group'], int(r['step'])) for r in summary}) == 13
    endpoints = {(r['group'], int(r['step']), seed) for r in summary for seed in range(810, 815)}
    independent = [r for r in rows('analysis_outputs/independent_metrics_per_snapshot.csv')
                   if r['case'] == 'm5T5' and (r['group'], int(r['step']), int(r['seed'])) in endpoints]
    residual_rows = [r for r in rows('analysis_outputs/independent_residual_checks.csv')
                     if r['case'] == 'm5T5' and (r['group'], int(r['step']), int(r['seed'])) in endpoints]
    costs = [r for r in rows('analysis_outputs/algorithm_costs_with_search_overhead.csv')
             if r['case'] == 'm5T5' and (r['group'], int(r['step']), int(r['seed'])) in endpoints]
    assert len(independent) == len(residual_rows) == len(costs) == 65
    assert all(r['pass_check'] == 'True' for r in independent + residual_rows)
    assert all(r['isolated_inference_seconds_median'] and r['isolated_inference_peak_rss_bytes'] for r in costs)
    measurements = []
    for r in independent:
        for field, rel in [('metric_record_sha256', r['path']),
                           ('prediction_sha256', str(Path(r['path']).parent / 'prediction.npz')),
                           ('checkpoint_sha256', str(Path(r['path']).parent / 'model_state.pt'))]:
            assert digest(ROOT / rel) == r[field], rel
        rel = f"isolated_inference/m5T5/{r['group']}/seed_{r['seed']}/step_{r['step']}.json"
        m = load(rel)
        assert m['status'] == 'COMPLETE_VERIFIED_INFERENCE' and m['checkpoint_sha256'] == r['checkpoint_sha256']
        assert len(m['inference_seconds_samples']) == len(m['inference_cpu_seconds_samples']) == 7
        measurements.append({'group': r['group'], 'seed': r['seed'], 'step': r['step'],
                             'measurement_path': rel, 'measurement_sha256': digest(ROOT / rel),
                             'inference_wall_seconds_median': m['inference_seconds_median'],
                             'inference_cpu_seconds_median': m['inference_cpu_seconds_median'],
                             'process_peak_rss_bytes': m['process_peak_rss_bytes'],
                             'parent_process_wall_seconds': m['parent_observed_process_wall_seconds']})
    source_reports = ['analysis_outputs/attempt_completion_audit.json',
                      'analysis_outputs/independent_metric_audit.json',
                      'analysis_outputs/independent_statistical_audit.json',
                      'analysis_outputs/independent_cost_audit.json',
                      'analysis_outputs/independent_parameter_audit.json',
                      'analysis_outputs/residual_audit.json',
                      'analysis_outputs/residual_provenance_audit.json',
                      'analysis_outputs/pairing_audit.json',
                      'analysis_outputs/carrier_initialization_audit.json',
                      'analysis_outputs/baseline_selection_audit.json',
                      'analysis_outputs/learning_advantage_assessment.json',
                      'isolated_inference/independent_audit.json',
                      'runtime/recovery_events/1791852771281141600/unknown_replay_costs_checkpoint_bound_corrected.csv']
    sources = {rel: digest(ROOT / rel) for rel in source_reports}
    provenance = []
    for j in jobs:
        rel = j['directory']
        complete = load(rel + '/complete.json')
        assert complete['status'] == 'COMPLETE' and complete['fixed_endpoint_reached'] and complete['step'] == j['budget']
        provenance.append({'directory': rel, 'type': j['type'], 'budget': j['budget'],
                           'complete_sha256': digest(ROOT / rel / 'complete.json'),
                           'config_sha256': digest(ROOT / rel / 'config.json'),
                           'complete': complete})
    DEST.mkdir(parents=True)
    files = {}
    for rel in source_reports:
        target = DEST / Path(rel).name
        shutil.copyfile(ROOT / rel, target)
        assert digest(target) == sources[rel] == digest(ROOT / rel)
        files[target.name] = digest(target)
    for name, selected in [('m5T5_fixed_endpoint_summary.csv', summary),
                           ('m5T5_independent_seed_metrics.csv', independent),
                           ('m5T5_independent_residual_checks.csv', residual_rows),
                           ('m5T5_costs_with_search_and_auxiliary.csv', costs),
                           ('m5T5_inference_measurements.csv', measurements)]:
        with (DEST / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
        files[name] = digest(DEST / name)
    (DEST / 'm5T5_all_formal_attempts.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    files['m5T5_all_formal_attempts.json'] = digest(DEST / 'm5T5_all_formal_attempts.json')
    receipt = {'archived_utc': time.time(), 'status': 'm5T5_FIXED_TRAINING_AND_ENDPOINT_AUDITS_COMPLETE',
               'case': 'm5T5', 'formal_attempts': 55, 'free_carrier_attempts': 10, 'coupled_attempts': 45,
               'prescribed_group_endpoints': 13, 'independently_verified_seed_endpoints': 65,
               'inference_measurements': 65, 'formal_seeds': list(range(810, 815)),
               'design_sha256': digest(ROOT / 'frozen/design.json'),
               'source_manifest_sha256': digest(ROOT / 'frozen/source_manifest.json'),
               'archive_code_sha256': digest(Path(__file__)), 'original_report_sha256': sources,
               'preserved_files_sha256': files, 'group_verdicts': summary,
               'scientific_failures_retained': True, 'norm_is_structural_not_dynamics_evidence': True,
               'unknown_extra_replay_costs_supplement_preserved': True,
               'unknown_extra_wall_and_CPU_not_filled_with_zero': True,
               'cost_scope': 'Recorded snapshot curves and completion ledger; supplementary unobserved replay costs remain unknown',
               'adapted_baseline_results_not_all_published_solver_results': True,
               'whole_goal_complete': False, 'final_figure_visual_checks_complete': False,
               'remaining_cases_and_zero_shot_and_final_documents_pending': True}
    (DEST / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if not isinstance(v, (list, dict))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
