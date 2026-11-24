from __future__ import annotations
import importlib.metadata,json,shutil,sys,time
from plan import BASE,CORE,EXTENSION,SEEDS,write,sha,freeze_design

def main():
    freeze_design()
    for path in ('history/audit_complete.json','engineering/tests_and_recovery.json','engineering/TSFP_tests.json','engineering/full_pipeline_check.json','references/calibration.json'):
        r=json.loads((BASE/path).read_text(encoding='utf-8'))
        if r['status']!='PASS':raise RuntimeError('Pre-freeze gate failed: '+path)
    used=json.loads((BASE/'history/used_seed_audit.json').read_text(encoding='utf-8'))
    if not used['unused_formal'] or not used['unused_development']:raise RuntimeError('Seed overlap')
    files=[BASE/'plan.py',BASE/'preregistration.md']+sorted((BASE/'closure').rglob('*.py'))
    record=dict(utc=time.time(),design_sha256=sha(BASE/'frozen/design.json'),files=[dict(path=str(p.relative_to(BASE)).replace('\\','/'),sha256=sha(p)) for p in files])
    target=BASE/'frozen/source_manifest.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['files']!=record['files']:raise RuntimeError('Already frozen science cannot be replaced')
    else:
        snapshot=BASE/'frozen/source';snapshot.mkdir(parents=True,exist_ok=True)
        for p in files:
            q=snapshot/p.relative_to(BASE);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        write(target,record)
    ref_files=sorted((BASE/'references').rglob('*'))
    ref_manifest=[dict(path=str(p.relative_to(BASE)).replace('\\','/'),sha256=sha(p)) for p in ref_files if p.is_file()]
    ref_target=BASE/'frozen/reference_manifest.json'
    if ref_target.exists():
        if json.loads(ref_target.read_text(encoding='utf-8'))!=ref_manifest:raise RuntimeError('Reference changed after freeze')
    else:write(ref_target,ref_manifest)
    versions={}
    for name in ('torch','numpy','scipy','pandas','matplotlib','psutil','threadpoolctl'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    write(BASE/'frozen/environment.json',dict(python=sys.version,executable=sys.executable,packages=versions))
    jobs=dict(carrier_jobs=30,development_jobs=24,core_coupled_jobs=135,extension_coupled_jobs=45,
        unique_train_jobs=234,zero_shot_evaluations=45,TSFP_configurations=6*3*6,TSFP_cost_repetitions=5,
        optimizer_steps=6000*30+4000*24+3*5*(5*20000+4*26000)+3*5*(2*20000+26000))
    estimate=dict(**jobs,estimated_bound_core_hours=355.,estimated_wall_hours_five_workers=71.,
        plausible_wall_hours_range=[60,120],evaluation_io_interruptions_extra=True,
        basis='R13 measured hard static~5511sec, analytic~4073sec, original~6896-7199sec per20000; autonomous evaluation coalescing is exact engineering',
        disk_free_bytes=shutil.disk_usage(BASE).free,estimated_new_storage_bytes=int(4.5*2**30),reserve_free_bytes=4*2**30,
        estimated_peak_RAM_bytes=6*2**30,all_costs_measured_in_final=True,no_step_count_speed_claim=True)
    write(BASE/'resource_estimate.json',estimate)
    print('SCIENCE_FROZEN',sha(target),json.dumps(jobs))

if __name__=='__main__':main()
