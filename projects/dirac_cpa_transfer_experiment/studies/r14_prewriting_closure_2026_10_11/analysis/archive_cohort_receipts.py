"""Preserve the first independently verified full five-seed endpoint assessment."""
from __future__ import annotations
import csv,json,math,shutil,time
from pathlib import Path
from plan import BASE,sha,write

def read_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))

def main():
    output=BASE/'analysis_outputs';design=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    metric_audit=output/'independent_metric_audit.json';stats_audit=output/'independent_statistical_audit.json'
    metric=json.loads(metric_audit.read_text(encoding='utf-8'));stats=json.loads(stats_audit.read_text(encoding='utf-8'))
    assert metric['status']=='PASS_AVAILABLE_SNAPSHOTS' and not metric['metric_mismatches']
    assert stats['status'] in ('PASS_AVAILABLE_STATISTICAL_TABLES','ALL_PRESCRIBED_STATISTICS_VERIFIED') and not stats['errors']
    for name in ('summary_metrics.csv','independent_metrics_per_snapshot.csv'):
        assert stats['source_table_sha256'][name]==sha(output/name),'Reaggregate and independently audit the current tables first'
    independent={Path(r['path']).as_posix():r for r in read_csv(output/'independent_metrics_per_snapshot.csv')}
    created=[];existing=[]
    for row in read_csv(output/'summary_metrics.csv'):
        if row['endpoint'] not in ('primary','terminal') or row['all_five_present']!='True' or row['independent_metric_audit_complete']!='True':continue
        assert row['status'] in ('PASS','FAIL') and int(row['n'])==5 and not row['missing_seeds'] and not row['failed_seeds']
        case,group,step=row['case'],row['group'],int(row['step'])
        directory=output/'cohort_receipts'/f'{case}_{group}_{step}_first'
        receipt_path=directory/'receipt.json'
        # The separately preserved analytic first receipt predates this helper.
        if case=='T10' and group=='analytic_hard' and step==20000 and (output/'T10_analytic_hard_primary_first_full_cohort.json').exists():
            existing.append('T10_analytic_hard_primary_first_full_cohort.json');continue
        per_seed=[];independent_rows=[];passes=[]
        for seed in design['formal_seeds']:
            path=BASE/f'formal/coupled/{case}/{group}/seed_{seed}/snapshots/{step:05d}/metrics.json'
            relative=path.relative_to(BASE).as_posix();raw=json.loads(path.read_text(encoding='utf-8'));check=independent[relative]
            assert check['pass_check']=='True' and check['metric_record_sha256']==sha(path)
            assert check['checkpoint_sha256']==raw['checkpoint_sha256']==sha(path.parent/'model_state.pt')
            assert check['prediction_sha256']==sha(path.parent/'prediction.npz')
            assert raw['job']['case']==case and raw['job']['group']==group and raw['job']['seed']==seed and raw['step']==step
            values={name:float(check[name]) for name in design['thresholds']}
            passed=all(math.isfinite(value) and (value>=design['thresholds'][name] if name=='fidelity_mean' else value<=design['thresholds'][name]) for name,value in values.items())
            assert passed==raw['seed_joint_pass'];passes.append(passed);independent_rows.append(check)
            per_seed.append(dict(seed=seed,metric_path=relative,metric_sha256=sha(path),checkpoint_sha256=raw['checkpoint_sha256'],
                prediction_sha256=check['prediction_sha256'],seed_joint_pass=passed,independent_values=values))
        means={name:sum(float(r[name]) for r in independent_rows)/5 for name in design['thresholds']}
        absolute=all(math.isfinite(value) and (value>=design['thresholds'][name] if name=='fidelity_mean' else value<=design['thresholds'][name]) for name,value in means.items()) and sum(passes)>=design['required_joint_successes']
        assert row['status']==('PASS' if absolute else 'FAIL') and int(row['joint_pass_count'])==sum(passes)
        if receipt_path.exists():
            prior=json.loads(receipt_path.read_text(encoding='utf-8'));assert prior['per_seed_sources']==per_seed,'Preserved endpoint evidence changed'
            existing.append(receipt_path.relative_to(BASE).as_posix());continue
        assert not directory.exists(),'An interrupted first archive must be reconciled without overwriting evidence'
        directory.mkdir(parents=True)
        copies={}
        for source in (metric_audit,stats_audit):
            target=directory/source.name;shutil.copyfile(source,target);copies[target.name]=sha(target)
        seed_table=directory/'independent_five_seed_rows.csv'
        with seed_table.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(independent_rows[0]));writer.writeheader();writer.writerows(independent_rows)
        copies[seed_table.name]=sha(seed_table)
        receipt=dict(assessed_utc=time.time(),status='PASS_FIXED_CASE_GROUP_ENDPOINT' if absolute else 'FAIL_FIXED_CASE_GROUP_ENDPOINT',
            case=case,group=group,coupled_steps=step,endpoint=row['endpoint'],all_five_seeds=design['formal_seeds'],summary=row,
            per_seed_sources=per_seed,independent_means=means,preserved_audit_files=copies,summary_table_sha256_at_assessment=sha(output/'summary_metrics.csv'),
            design_sha256=sha(BASE/'frozen/design.json'),archive_code_sha256=sha(__file__),
            no_extra_training_or_accuracy_driven_budget=True,norm_conservation_is_structural_not_dynamics_evidence=True,
            pretraining_necessity_or_universal_method_superiority_not_inferred=True,whole_study_complete=False)
        write(receipt_path,receipt);created.append(dict(path=receipt_path.relative_to(BASE).as_posix(),status=receipt['status'],joint_successes=sum(passes)))
    print(json.dumps(dict(created=created,previously_preserved=len(existing),whole_study_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
