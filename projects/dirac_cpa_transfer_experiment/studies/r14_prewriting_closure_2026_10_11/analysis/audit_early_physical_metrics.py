"""Read-only R1-R8 adapters, preserving each round's native gauge and modal metric."""
from __future__ import annotations
import json,time
import numpy as np
import pandas as pd
from plan import BASE,PROJECT,write,sha
from .early_format_inventory import NAMES

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

def configuration_for(path,root):
    for parent in (path.parent,*path.parents):
        if parent==root.parent:break
        p=parent/'config.json'
        if p.exists():
            c=json.loads(p.read_text(encoding='utf-8'))
            return c.get('experiment_config',c.get('configuration',c)),p
    raise RuntimeError('No legacy config for '+str(path))

def physical_metrics(pred,ref,length):
    alpha=float(np.angle(np.vdot(ref.ravel(),pred.ravel())));aligned=pred*np.exp(-1j*alpha)
    rho=np.sum(abs(pred)**2,-1);rho_ref=np.sum(abs(ref)**2,-1)
    current=2*(pred[...,0].conj()*pred[...,1]).real;current_ref=2*(ref[...,0].conj()*ref[...,1]).real
    norms=length*rho.mean(1);norm_ref=length*rho_ref.mean(1)
    overlap=length*np.sum(ref.conj()*pred,-1).mean(1)
    fidelity=np.clip(abs(overlap)**2/np.maximum(norms*norm_ref,1e-30),0.,1.)
    return dict(spinor_rel_l2=rel(pred,ref),spinor_rel_l2_raw=rel(pred,ref),spinor_rel_l2_gauge_aligned=rel(aligned,ref),
        density_rel_l2=rel(rho,rho_ref),rho_rel_l2=rel(rho,rho_ref),current_rel_l2=rel(current,current_ref),
        max_norm_drift=float(abs(norms-1).max()),fidelity_mean=float(fidelity.mean()),fidelity_min=float(fidelity.min()),
        fidelity_final=float(fidelity[-1]),global_gauge_phase_rad=alpha)

def modal_metrics(pred,ref,x,k,length,mass,raw_spectra=None):
    # Independently project the raw physical field; do not reuse archived branch coefficients.
    kernel=np.exp(-1j*np.asarray(k)[:,None]*x[None,:])*np.sqrt(length)/len(x)
    if raw_spectra is None:
        ps=np.einsum('kx,txc->tkc',kernel,pred,optimize=True);rs=np.einsum('kx,txc->tkc',kernel,ref,optimize=True)
    else:ps,rs=raw_spectra
    theta=.5*np.arctan2(k,mass);up=np.stack((np.cos(theta),np.sin(theta)),-1);um=np.stack((-np.sin(theta),np.cos(theta)),-1)
    def branches(s):return np.stack((np.sum(s*up[None],-1),np.sum(s*um[None],-1)),-1)
    bp,br=branches(ps),branches(rs)
    result=dict(modal_population_rel_l2=rel(np.sum(abs(ps)**2,-1),np.sum(abs(rs)**2,-1)),
        spectral_occupation_rel_l2=rel(abs(bp)**2,abs(br)**2),mode_occupation_rmse_rel=rel(abs(bp)**2,abs(br)**2))
    if raw_spectra is not None:
        numerator=denominator=0.
        for p,r in zip(bp,br):
            p,r=p.ravel(),r.ravel();ids=np.flatnonzero(abs(r)**2>=1e-6);p,r=p[ids],r[ids]
            a=p[:,None]*p.conj()[None];b=r[:,None]*r.conj()[None];mask=~np.eye(len(ids),dtype=bool)
            numerator+=np.sum(abs(a[mask]-b[mask])**2);denominator+=np.sum(abs(b[mask])**2)
        result['active_coherence_rel_l2']=float(np.sqrt(numerator/denominator)) if denominator else None
    return result

def main():
    rows=[];errors=[];skipped=[];start=time.time()
    roots=tuple(enumerate(NAMES,1))+((1,'structure_weight_sensitivity'),)
    for round_no,name in roots:
        root=PROJECT/'results/raw'/name
        for p in sorted(root.rglob('dense_grid.npz')):
            if 'checkpoints' in p.parts or 'snapshots' in p.parts:continue
            metric_path=p.parent/'metrics.json'
            if not metric_path.exists():skipped.append(dict(path=p.relative_to(PROJECT).as_posix(),reason='No native metrics'));continue
            try:
                conf,config_path=configuration_for(p,root);native=json.loads(metric_path.read_text(encoding='utf-8'))
                with np.load(p) as f:
                    key='psi_predicted_raw' if 'psi_predicted_raw' in f else 'psi_predicted'
                    pred,ref,x=f[key],f['psi_exact'],f['x']
                    k=f['wave_numbers'] if 'wave_numbers' in f else f['training_wave_numbers'] if 'training_wave_numbers' in f else np.asarray(conf.get('wave_numbers',[]))
                    raw_spectra=(f['spectral_state_predicted_raw'],f['spectral_state_exact']) if 'spectral_state_predicted_raw' in f else None
                if not np.isfinite(pred).all() or not np.isfinite(ref).all():raise FloatingPointError('Nonfinite legacy raw fields')
                if 'length' in conf:length=float(conf['length']);length_source='native configuration'
                elif 'x_max' in conf:length=float(conf['x_max']-conf['x_min']);length_source='native bounds'
                else:length=float((x[1]-x[0])*len(x));length_source='uniform periodic archived grid'
                a=physical_metrics(pred,ref,length);alias=False;bilinear_grid_safe=None
                if len(k):
                    indices=np.rint(k*length/(2*np.pi)).astype(int)
                    alias=len(np.unique(indices%len(x)))!=len(indices)
                    bilinear_grid_safe=len(x)>2*(indices.max()-indices.min())
                    mass=float(conf['mass']) if 'mass' in conf else float(np.sqrt(conf['reference_energies'][0]**2-k[0]**2))
                    # Old smoke grids can alias distinct retained modes. Native modal metrics then
                    # refer to the archived raw spectrum, not an invertible physical projection.
                    a.update(modal_metrics(pred,ref,x,k,length,mass,raw_spectra if alias else None))
                    if raw_spectra is not None and not alias:
                        a['active_coherence_rel_l2']=modal_metrics(pred,ref,x,k,length,mass,raw_spectra).get('active_coherence_rel_l2')
                compare={key:abs(a[key]-native[key]) for key in a if key in native and native[key] is not None}
                valid=all(v<2e-11+2e-12*abs(native[key]) for key,v in compare.items())
                row=dict(historical_round=round_no,experiment=name,path=p.relative_to(PROJECT).as_posix(),
                    comparison_metric_count=len(compare),max_metric_absolute_difference=max(compare.values()),pass_check=valid,
                    native_gauge='raw' if round_no<=3 else 'one global constant spacetime phase',
                    native_modality='two plane waves / branch occupations' if round_no==4 else 'native retained modes',
                    length_source=length_source,physical_modal_projection_aliasing=bool(alias),bilinear_error_quadrature_grid_safe=bool(bilinear_grid_safe) if bilinear_grid_safe is not None else None,
                    modal_recalculation_source='archived raw spectrum because physical grid aliases' if alias else 'independent physical Fourier projection',
                    raw_sha256=sha(p),native_metrics_sha256=sha(metric_path),config_sha256=sha(config_path),
                    **{key:value for key,value in a.items() if key not in ('rho_rel_l2','spinor_rel_l2')})
                rows.append(row)
                if not valid:errors.append(dict(path=row['path'],differences=compare))
            except Exception as error:errors.append(dict(path=p.relative_to(PROJECT).as_posix(),error=repr(error)))
        print('EARLY_NATIVE_AUDIT',round_no,name,len([r for r in rows if r['experiment']==name]),flush=True)
    out=BASE/'history';pd.DataFrame(rows).to_csv(out/'independent_R1_R8_metrics.csv',index=False)
    report=dict(status='PASS_AVAILABLE_LEGACY_ENDPOINTS' if rows and not errors and not skipped else 'MISMATCH_OR_LIMITATION',
        endpoints_audited=len(rows),errors=errors,skipped=skipped,elapsed_wall_seconds=time.time()-start,
        scope='Available raw physical final arrays R1-R8, including R1 separate sensitivity root and historical smoke; secondary checkpoints excluded',
        endpoints_by_experiment={name:len([r for r in rows if r['experiment']==name]) for _,name in roots},
        raw_history_unchanged=True,no_training_repeated=True,old_decisions_and_gauges_preserved=True,
        no_cross_round_accuracy_or_cost_pooling=True)
    write(out/'independent_R1_R8_audit.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    import psutil
    psutil.Process().cpu_affinity([6])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
