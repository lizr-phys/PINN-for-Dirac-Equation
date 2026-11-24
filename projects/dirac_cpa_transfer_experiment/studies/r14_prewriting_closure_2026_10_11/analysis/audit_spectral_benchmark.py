"""Audit all fixed TSFP configurations with full spectral convolutions.

Current/density L2 norms use Parseval on bilinear Fourier convolutions, avoiding
the producer's spatial quadrature routine. This does not run extra solvers or
alter the reference, algorithm, steps, repeats, or neural training budgets.
"""
from __future__ import annotations
import csv,json,math,statistics,time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from plan import BASE,sha,write


def metrics(pred,ref,n,bounds):
    def relative(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
    phase=float(np.angle(np.sum(ref.conj()*pred)))
    aligned=pred*np.exp(-1j*phase)
    def bilinears(s):
        a,b=s[...,0],s[...,1]
        autoa=fftconvolve(a.conj()[:,::-1],a,axes=1)
        autob=fftconvolve(b.conj()[:,::-1],b,axes=1)
        cross=fftconvolve(a.conj()[:,::-1],b,axes=1)
        opposite=fftconvolve(b.conj()[:,::-1],a,axes=1)
        return autoa+autob,cross+opposite
    rho_p,j_p=bilinears(pred);rho_r,j_r=bilinears(ref)
    power_p=np.sum(np.abs(aligned)**2,axis=2);power_r=np.sum(np.abs(ref)**2,axis=2)
    normp=power_p.sum(axis=1);normr=power_r.sum(axis=1)
    fidelity=np.abs(np.sum(ref.conj()*pred,axis=(1,2)))**2/(normp*normr)
    outside=(n<bounds[0])|(n>bounds[1])
    return dict(spinor_rel_l2_raw=relative(pred,ref),spinor_rel_l2_gauge_aligned=relative(aligned,ref),
        current_rel_l2=relative(j_p,j_r),density_rel_l2=relative(rho_p,rho_r),
        fidelity_mean=float(np.mean(fidelity)),fidelity_min=float(np.min(fidelity)),
        modal_population_rel_l2=relative(power_p,power_r),max_norm_drift=float(np.max(np.abs(normp-1))),
        global_spacetime_constant_phase=phase,
        predicted_occupation_outside_reference_truncation_max=float(np.max(np.sum(np.abs(pred[:,outside])**2,axis=(1,2)))))


def main():
    started=time.perf_counter();d=json.loads((BASE/'frozen/design.json').read_text(encoding='utf-8'))
    directory=BASE/'spectral_benchmark';wide_dir=BASE/'spectral_reference_check';rows=[];missing=[];errors=[]
    expected=set();primary_cache={};wide_cache={};thresholds=d['thresholds']
    wide_csv=wide_dir/'spectral_error_cost_wide_reference.csv'
    with wide_csv.open(encoding='utf-8-sig',newline='') as stream:
        wide_rows=list(csv.DictReader(stream))
    wide_table={(r['case'],int(r['grid']),float(r['dt'])):r for r in wide_rows}
    with (directory/'error_cost.csv').open(encoding='utf-8-sig',newline='') as stream:
        primary_rows=list(csv.DictReader(stream))
    primary_table={(r['case'],int(r['grid']),float(r['dt'])):r for r in primary_rows}
    for case in d['core_cases']+d['extension_cases']:
        primary=BASE/'references'/case/'test201.npz';wide=wide_dir/f'{case}_wide401.npz'
        with np.load(primary) as z:primary_cache[case]=(z['t'],z['n'],z['state'])
        with np.load(wide) as z:wide_cache[case]=(z['t'][::2],z['n'],z['state'][::2])
        for nx in d['spectral_integrator']['grid_points']:
            for dt in d['spectral_integrator']['time_steps']:
                tag=f'{case}_N{nx}_dt{dt:.8f}';record_path=directory/(tag+'.json');raw_path=directory/(tag+'.npz')
                expected.add(record_path.name)
                if not record_path.exists() or not raw_path.exists():missing.append(tag);continue
                record=json.loads(record_path.read_text(encoding='utf-8'))
                with np.load(raw_path) as z:t,n,pred=z['t'],z['n'],z['prediction']
                issues=[]
                def check(field,actual,expected_value):
                    if isinstance(expected_value,bool):valid=actual==expected_value
                    elif isinstance(expected_value,str):valid=actual==expected_value
                    else:valid=math.isfinite(float(actual)) and math.isclose(float(actual),expected_value,rel_tol=2e-10,abs_tol=2e-11)
                    if not valid:issues.append(dict(field=field,actual=actual,expected=expected_value))
                for field,value in dict(case=case,grid=nx,dt=dt,method='TSFP2',cost_repetitions=5,
                    integration_steps=round((10. if case=='T10' else 5.)/dt),
                    FFT_calls=4*round((10. if case=='T10' else 5.)/dt),training_seconds=0.,auxiliary_pretraining_seconds=0.,
                    unit_spinor_residual_points=0,setup_and_all_free_phase_computation_included=True,
                    known_free_operator_prior=True,reference_diagonalization_excluded=True,
                    full_FFT_spectrum_in_error=True,physical_quadrature_points=2048).items():check(field,record.get(field),value)
                wall=record['cost_seconds_samples'];cpu=record['cpu_seconds_samples']
                if len(wall)!=len(cpu) or len(wall)!=5 or not all(math.isfinite(v) and v>0 for v in wall) or not all(math.isfinite(v) and v>=0 for v in cpu):
                    issues.append(dict(field='five_valid_cost_observations'))
                check('cost_median_seconds',record['cost_median_seconds'],statistics.median(wall))
                check('cpu_median_seconds',record['cpu_median_seconds'],statistics.median(cpu))
                if not record['peak_rss_bytes']>=record['process_rss_bytes']>0:issues.append(dict(field='memory_range'))
                if pred.shape!=(201,nx,2) or not np.isfinite(pred).all() or not np.array_equal(n,np.rint(np.fft.fftfreq(nx)*nx).astype(int)):
                    issues.append(dict(field='full_FFT_grid_or_finite_prediction'))
                key=(case,nx,dt);computed={};differences={}
                for scope,cache in (('primary161',primary_cache[case]),('wide289',wide_cache[case])):
                    rt,rn,reference=cache
                    if not np.array_equal(t,rt):issues.append(dict(field=scope+'_evaluation_time_grid'))
                    union=np.arange(min(n.min(),rn.min()),max(n.max(),rn.max())+1)
                    pp=np.zeros((201,len(union),2),dtype=np.complex128);rr=np.zeros_like(pp)
                    pp[:,n-union[0]]=pred;rr[:,rn-union[0]]=reference
                    met=metrics(pp,rr,union,(int(rn.min()),int(rn.max())))
                    produced=record if scope=='primary161' else wide_table.get(key)
                    if produced is None:issues.append(dict(field=scope+'_summary_row_missing'));continue
                    for field,value in met.items():
                        check(scope+'_'+field,float(produced[field]),value)
                    computed.update({scope+'_'+field:value for field,value in met.items()})
                    differences[scope]=max(abs(float(produced[k])-v) for k,v in met.items())
                    passed=all(met[k]>=v if k=='fidelity_mean' else met[k]<=v for k,v in thresholds.items())
                    archived_pass=produced['seed_joint_pass'] if scope=='primary161' else str(produced['seed_joint_pass']).lower()=='true'
                    check(scope+'_joint_pass',archived_pass,passed)
                tab=primary_table.get(key)
                if tab is None:issues.append(dict(field='primary_cost_table_row_missing'))
                else:
                    for field in ('cost_median_seconds','cpu_median_seconds','spinor_rel_l2_gauge_aligned','current_rel_l2','fidelity_mean','modal_population_rel_l2','max_norm_drift'):
                        check('primary_table_'+field,float(tab[field]),record[field])
                wide_row=wide_table.get(key)
                if wide_row is not None:
                    for field in ('cost_median_seconds','cpu_median_seconds'):check('wide_table_'+field,float(wide_row[field]),record[field])
                row=dict(case=case,grid=nx,dt=dt,path=record_path.relative_to(BASE).as_posix(),
                    metric_record_sha256=sha(record_path),prediction_sha256=sha(raw_path),
                    primary_reference_sha256=sha(primary),wide_reference_sha256=sha(wide),
                    cost_median_seconds=statistics.median(wall),cpu_median_seconds=statistics.median(cpu),
                    cost_wall_min_seconds=min(wall),cost_wall_max_seconds=max(wall),
                    peak_rss_bytes=record['peak_rss_bytes'],memory_is_cumulative_benchmark_process_peak=True,
                    full_spectral_bilinear_convolution=True,pass_check=not issues,
                    maximum_primary_metric_abs_difference=differences.get('primary161'),
                    maximum_wide_metric_abs_difference=differences.get('wide289'),**computed)
                rows.append(row)
                if issues:errors.append(dict(tag=tag,issues=issues))
    actual={p.name for p in directory.glob('*.json') if p.name!='complete.json' and '_N' in p.name}
    unexpected=sorted(actual-expected)
    if len(primary_rows)!=108 or len(primary_table)!=108 or len(wide_rows)!=108 or len(wide_table)!=108 or unexpected:
        errors.append(dict(field='full_108_configuration_inventory',primary=len(primary_rows),wide=len(wide_rows),unexpected=unexpected))
    all_checked=len(rows)==108 and not missing and not errors
    output=BASE/'analysis_outputs';pd.DataFrame(rows).to_csv(output/'independent_TSFP_metrics_costs.csv',index=False)
    audit=dict(updated_utc=time.time(),status='PASS' if all_checked else 'FAIL_OR_INCOMPLETE',
        fixed_configurations_checked=len(rows),prescribed_configurations=108,cost_repetitions_per_configuration=5,
        five_cost_samples_recomputed_without_rerunning_solver=True,all_spectral_benchmark_records_verified=all_checked,
        primary_and_wide_reference_metrics_independently_recomputed=True,
        bilinear_L2_from_full_convolution_not_spatial_quadrature=True,missing=missing,errors=errors,
        source_code_sha256=sha(__file__),source_table_sha256={str(p.relative_to(BASE)):sha(p) for p in (directory/'error_cost.csv',wide_csv)},
        original_records_references_and_training_unchanged=True,
        benchmark_OS_peak_RSS_is_cumulative_not_isolated_per_configuration=True,
        elapsed_wall_seconds=time.perf_counter()-started,whole_goal_complete=False)
    write(output/'independent_TSFP_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('errors','source_table_sha256')},ensure_ascii=False))
    if errors:
        print(json.dumps(errors[:3],ensure_ascii=False));raise RuntimeError('Independent TSFP benchmark audit mismatch')


if __name__=='__main__':
    import psutil
    from threadpoolctl import threadpool_limits
    psutil.Process().cpu_affinity([6])
    with threadpool_limits(limits=1):main()
