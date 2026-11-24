from __future__ import annotations

import ast
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.r13_config import ROOT,RAW,THRESHOLDS,ARTIFACTS,EXPLORATION_SEEDS,VALIDATION_SEEDS,CARRIER_KINDS,ENVELOPE_KINDS,configuration,protocol
from src.r13_models import LearnedRecoveryCarrier,R13Model,hashes,parameter_counts,total_norm_parameterization
from src.r13_residual import spinor_value_and_time_derivative,free_carrier_residual,coupled_residual,training_loss
from src.r9_models import initial_coefficients,original_gaussian
from src.r12_hamiltonian import build_hamiltonian,hamiltonian_apply
from src.r8_models import LearnedCarrier
from src.r8_sampling import TimeSampler

torch.set_num_threads(1)

def grid(cfg, nt=2):
    time=torch.linspace(.12*cfg.t_max,.83*cfg.t_max,nt,dtype=torch.float64).requires_grad_()
    k=torch.tensor(cfg.wave_numbers,dtype=torch.float64).reshape(-1,1).repeat(nt,1)
    return k,time,time.repeat_interleave(cfg.mode_count).reshape(-1,1)

@pytest.mark.parametrize('case',('bridge','T10','m10'))
def test_physics_bridge_161_modes_shift8_gaussian_unchanged_and_hermiticity(case):
    cfg=configuration(case)
    assert cfg.mode_count==161 and cfg.ladder_half_width==10 and cfg.potential_harmonics==((1,.2),)
    h=build_hamiltonian(cfg)
    assert np.array_equal(h,h.conj().T) and h[0,16]==.1 and h[0,2]==0
    c=initial_coefficients(cfg)
    assert torch.equal(c[64:97,0],original_gaussian().to(torch.complex128)/2**.5)
    assert c[:64].count_nonzero()==0 and c[97:].count_nonzero()==0
    assert abs(float(c.abs().square().sum())-1)<4e-16
    bigger=initial_coefficients(replace(cfg,n_min=-80,n_max=112))
    assert torch.equal(c,bigger[16:-16])
    if case=='m10':
        from src.r9_reference import initial_numpy
        assert not np.allclose(initial_numpy(cfg),initial_numpy(replace(cfg,mass=5)))

@pytest.mark.parametrize('case',('T10','m10'))
def test_original_migration_scaled_phase_physical_initialization_and_static_initial_slope(case):
    cfg=configuration(case);k,time,t=grid(cfg)
    old=LearnedCarrier(cfg,600)
    original=LearnedRecoveryCarrier(cfg,'original',600)
    scaled=LearnedRecoveryCarrier(cfg,'scaled_phase',600)
    static=LearnedRecoveryCarrier(cfg,'static_frequency',600)
    assert all(torch.equal(a,b) for a,b in zip(old.parameters(),original.parameters()))
    torch.testing.assert_close(original(k,t),old(k,t),atol=0,rtol=0)
    a,at=original.value_and_time_derivative(k,t);b,bt=scaled.value_and_time_derivative(k,t)
    torch.testing.assert_close(a,b,atol=3e-16,rtol=3e-16)
    torch.testing.assert_close(at,bt,atol=3e-16,rtol=3e-16)
    zero=torch.zeros_like(t)
    a,at=original.value_and_time_derivative(k,zero);b,bt=static.value_and_time_derivative(k,zero)
    assert torch.equal(a,b)
    torch.testing.assert_close(at,bt,atol=3e-16,rtol=3e-16)
    assert sum(p.numel() for p in original.parameters())==12802
    assert sum(p.numel() for p in static.parameters())==12738

@pytest.mark.parametrize('kind',CARRIER_KINDS+('exact',))
@pytest.mark.parametrize('envelope',ENVELOPE_KINDS)
def test_hard_initial_time_derivative_and_all_active_parameter_gradients(kind,envelope):
    cfg=configuration('T10');model=R13Model(cfg,kind,envelope,601)
    model.carrier.freeze()
    k,time,t=grid(cfg)
    original=model(k,t)
    value,tangent=spinor_value_and_time_derivative(model,cfg,k,t)
    torch.testing.assert_close(value,original,atol=4e-16,rtol=4e-16)
    # Differentiate each block's SINGLE common physical time. A rowwise partial
    # derivative would incorrectly omit other modal contributions to the norm.
    ad_value,ad_tangent=torch.autograd.functional.jvp(
        lambda times:model(k,times.repeat_interleave(cfg.mode_count).reshape(-1,1)),
        time,torch.ones_like(time),create_graph=True)
    torch.testing.assert_close(value,ad_value,atol=5e-15,rtol=5e-15)
    torch.testing.assert_close(tangent,ad_tangent,atol=5e-14,rtol=5e-14)
    residual=1j*ad_tangent.reshape(2,cfg.mode_count,2)-hamiltonian_apply(ad_value.reshape(2,cfg.mode_count,2),cfg)
    torch.testing.assert_close(coupled_residual(model,cfg,k,t),residual,atol=5e-14,rtol=5e-14)
    reference_loss=residual.abs().square().mean(-1).sum(-1).mean()/cfg.length
    if envelope=='norm_regularized':
        reference_loss+=cfg.norm_regularization_weight*(ad_value.reshape(2,cfg.mode_count,2).abs().square().sum((1,2))-1).square().mean()
    loss,_,terms=training_loss(model,cfg,k,t)
    parameters=tuple(model.networks.parameters())
    fast=torch.autograd.grad(loss,parameters,retain_graph=True)
    slow=torch.autograd.grad(reference_loss,parameters)
    for a,b in zip(fast,slow):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
    assert all(p.grad is None and not p.requires_grad for p in model.carrier.parameters())
    k0=k[:cfg.mode_count];t0=torch.zeros_like(k0)
    expected=R13Model(cfg,'exact','pde_only',601)(k0,t0)
    torch.testing.assert_close(model(k0,t0),expected,atol=1e-14,rtol=0)
    if envelope=='hard_total_norm':
        norms=value.reshape(2,cfg.mode_count,2).abs().square().sum((1,2))
        torch.testing.assert_close(norms,torch.ones_like(norms),atol=8e-16,rtol=0)
        assert float((value.conj()*tangent).real.reshape(2,cfg.mode_count,2).sum((1,2)).abs().max())<1e-13
    else:
        assert terms['norm_penalty']>=0

@pytest.mark.parametrize('kind',CARRIER_KINDS)
def test_free_full_spinor_derivative_and_parameter_gradient_no_diagonal_training(kind):
    cfg=configuration('m10');carrier=LearnedRecoveryCarrier(cfg,kind,602)
    k,time,t=grid(cfg)
    p,pt=carrier.value_and_time_derivative(k,t)
    adp,adpt=torch.autograd.functional.jvp(lambda times:carrier(k,times.repeat_interleave(cfg.mode_count).reshape(-1,1)),
        time,torch.ones_like(time),create_graph=True)
    torch.testing.assert_close(p,adp,atol=2e-15,rtol=2e-15)
    torch.testing.assert_close(pt,adpt,atol=2e-14,rtol=2e-14)
    loss=free_carrier_residual(carrier,cfg,k,t).abs().square().mean()
    gradients=torch.autograd.grad(loss,tuple(carrier.parameters()))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    carrier.freeze()
    t.requires_grad_();phase=carrier(k,t)
    assert torch.autograd.grad(phase.real.sum()+phase.imag.sum(),t)[0].abs().max()>0

def test_total_norm_allows_modal_branch_exchange_and_has_no_silent_clamp():
    raw=torch.zeros((2,3,2),dtype=torch.complex128)
    raw[0,0,0]=1;raw[1,2,1]=2
    normalized=total_norm_parameterization(raw.reshape(-1,2),mode_count=3).reshape_as(raw)
    assert abs(normalized[0,0,0])==1 and abs(normalized[1,2,1])==1
    assert normalized[0,2,1]==0 and normalized[1,0,0]==0
    with pytest.raises(FloatingPointError):total_norm_parameterization(torch.zeros((3,2),dtype=torch.complex128),mode_count=3)

def test_original_A_complete_residual_matches_R12_implementation():
    from src.r12_models import R12Model
    from src.r12_residual import coupled_residual as old_residual
    cfg=configuration('T10');old=R12Model(cfg,'A_pretrained_frozen',600)
    new=R13Model(cfg,'original','pde_only',600)
    old.carrier.freeze();new.carrier.freeze()
    k,time,t=grid(cfg)
    assert set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(old.state_dict()[name],new.state_dict()[name]) for name in old.state_dict())
    torch.testing.assert_close(old(k,t),new(k,t),atol=0,rtol=0)
    torch.testing.assert_close(old_residual(old,cfg,k,t),coupled_residual(new,cfg,k,t),atol=0,rtol=0)

def test_parameters_and_bitwise_envelope_pairing_and_random_streams():
    cfg=configuration('T10')
    models=[R13Model(cfg,k,e,600) for k in CARRIER_KINDS+('exact',) for e in ENVELOPE_KINDS]
    assert len({hashes(m)['envelopes'][0] for m in models})==1
    for model in models:
        model.carrier.freeze();count=parameter_counts(model)
        assert count['active_parameters']==12932
        expected=12932 if model.carrier_kind=='exact' else (25670 if model.carrier_kind=='static_frequency' else 25734)
        assert count['total_parameters']==expected
    sample=TimeSampler(cfg,600);same=TimeSampler(cfg,600);free=TimeSampler(cfg,600,carrier=True)
    for _ in range(3):free.sample()
    for _ in range(3):
        assert torch.equal(sample.sample()[1],same.sample()[1])
    assert not torch.equal(free.normalized_trace[0],sample.normalized_trace[0])
    saved=sample.state_dict();restored=TimeSampler(cfg,600);restored.load_state_dict(saved)
    assert torch.equal(sample.sample()[1],restored.sample()[1])
    cross=TimeSampler(configuration('m10'),600)
    cross.sample();assert torch.equal(cross.normalized_trace[0],same.normalized_trace[0])

def test_protocol_disjoint_seeds_fixed_selection_budget_and_absolute_success():
    p=protocol()
    assert set(EXPLORATION_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert p['planned_unique_optimizer_steps_max']==1566000
    assert p['carrier_pretraining_steps']==6000 and p['coupled_steps']==20000
    assert p['norm_regularization_weight']==.01 and p['required_joint_pass_count']==4
    from src.r13_selection import absolute_success
    frame=pd.DataFrame(dict(seed=VALIDATION_SEEDS,spinor_rel_l2_gauge_aligned=[.02]*5,current_rel_l2=[.03]*5,
        fidelity_mean=[.99]*5,modal_population_rel_l2=[.02]*5,max_norm_drift=[0.]*5))
    assert absolute_success(frame)['pass_check']
    frame['spinor_rel_l2_gauge_aligned']=.7
    assert not absolute_success(frame)['pass_check']
    frame['spinor_rel_l2_gauge_aligned']=.02
    frame.loc[0,'max_norm_drift']=.03
    assert absolute_success(frame)['pass_check']
    frame.loc[1,'max_norm_drift']=.03
    assert not absolute_success(frame)['pass_check']

@pytest.mark.parametrize('combination_passes',(True,False))
def test_selection_uses_only_exploration_fixed_endpoints_and_at_most_one_combination(tmp_path,monkeypatch,combination_passes):
    import src.r13_config as config_module
    import src.r13_selection as selection
    import src.r13_plan as plan
    from src.freeze_r7_phase_release import write_json
    base=tmp_path/'experiment';raw=base/'raw'
    for module in (config_module,selection,plan):
        monkeypatch.setattr(module,'RAW',raw)
        monkeypatch.setattr(module,'ROOT',base)
    for case in ('T10','m10'):
        for kind,phase,residual in (('original',1.,1.),('scaled_phase',.999,.9),('static_frequency',.2,.2)):
            for seed in EXPLORATION_SEEDS:
                write_json(config_module.carrier_directory('free_carriers',case,kind,seed)/'metrics.json',
                    dict(case=case,seed=seed,carrier_kind=kind,initial_weighted_unwrapped_phase_rms=phase,
                        initial_weighted_unit_spinor_residual_rms=residual,carrier_free_pde_residual_rms=residual))
        for envelope in ENVELOPE_KINDS:
            for seed in EXPLORATION_SEEDS:
                write_json(config_module.run_directory('exact_envelopes',case,envelope,seed)/'metrics.json',
                    dict(case=case,seed=seed,envelope_kind=envelope,spinor_rel_l2_gauge_aligned=.02 if envelope=='hard_total_norm' else .03,
                        current_rel_l2=.04,fidelity_mean=.99,modal_population_rel_l2=.03,
                        max_norm_drift=.025 if envelope=='pde_only' else (.0 if envelope=='hard_total_norm' else .003)))
    c=selection.select_carrier();e=selection.select_envelope()
    assert c['eligible'] and c['kind']=='static_frequency'
    assert e['eligible'] and e['kind']=='hard_total_norm'
    for seed in EXPLORATION_SEEDS:
        write_json(config_module.run_directory('T10_factorial','T10','combined',seed)/'metrics.json',
            dict(seed=seed,spinor_rel_l2_gauge_aligned=.02 if combination_passes else .4,
                current_rel_l2=.04,fidelity_mean=.99,modal_population_rel_l2=.03,max_norm_drift=0.))
    decision=selection.select_combination()
    assert decision['selected']==combination_passes and decision['independent_combination_count']==int(combination_passes)
    jobs=[j for seed in VALIDATION_SEEDS for j in plan.jobs_for('independent_validation',seed)]
    assert sum(j['type']=='coupled' for j in jobs)==(30 if combination_passes else 20)
    assert all(j['seed']>=700 for j in jobs)
    assert all(plan.validate_job(j) for j in jobs)
    if combination_passes:
        alljobs=[j for stage in plan.STAGES for seed in (VALIDATION_SEEDS if stage=='independent_validation' else EXPLORATION_SEEDS) for j in plan.jobs_for(stage,seed)]
        assert sum(j['type']=='carrier' for j in alljobs)==41
        assert sum(j['type']=='coupled' for j in alljobs)==66
    assert not any('700' in name for name in decision['evidence_sha256'])
def test_learned_science_paths_no_reference_future_or_exact_frequency_labels():
    for name in ('r13_models.py','r13_residual.py','train_r13_recovery.py'):
        tree=ast.parse((ROOT/'src'/name).read_text(encoding='utf-8'))
        assert not any(isinstance(n,ast.ImportFrom) and 'reference' in (n.module or '') for n in ast.walk(tree))
    source=(ROOT/'src/r13_models.py').read_text(encoding='utf-8')
    assert 'sqrt(k' not in source and 'eigenvalues' not in source
    residual=(ROOT/'src/r13_residual.py').read_text(encoding='utf-8')
    assert 'hamiltonian_apply' in residual and 'target' not in residual
    for kind in CARRIER_KINDS:
        model=R13Model(configuration('m10'),kind,'hard_total_norm',600)
        assert not any(any(w in key for w in ('energy','eigenvalue','reference','target')) for key in model.state_dict())
        assert not model.carrier.analytic_prior

@pytest.mark.parametrize('kind',('original','static_frequency'))
def test_exact_adam_and_sampling_recovery_and_six_artifacts(tmp_path,kind):
    from src.train_r13_recovery import train_job
    job=dict(type='carrier',stage='smoke',case='T10',carrier_kind=kind,seed=602)
    direct,resumed=tmp_path/'direct',tmp_path/'resumed'
    train_job(job,direct,smoke=True)
    train_job(job,resumed,smoke=True,stop_after=1)
    train_job(job,resumed,smoke=True)
    a,b=(torch.load(p/'recovery_state.pt',weights_only=False) for p in (direct,resumed))
    assert all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
    assert all(torch.equal(a['sampler'][k],b['sampler'][k]) for k in a['sampler'])
    for key in a['optimizer']['state']:
        assert all(torch.equal(a['optimizer']['state'][key][v],b['optimizer']['state'][key][v]) for v in a['optimizer']['state'][key])
    coupled_job=dict(type='coupled',stage='smoke',case='T10',carrier_kind=kind,envelope_kind='hard_total_norm',seed=602,
        group='test_hard',pretrain_source=str(direct))
    direct2,resumed2=tmp_path/'direct2',tmp_path/'resumed2'
    train_job(coupled_job,direct2,smoke=True)
    train_job(coupled_job,resumed2,smoke=True,stop_after=1);train_job(coupled_job,resumed2,smoke=True)
    a,b=(torch.load(p/'recovery_state.pt',weights_only=False) for p in (direct2,resumed2))
    assert all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
    for p in (direct,resumed,direct2,resumed2):
        assert all((p/n).exists() for n in ARTIFACTS+('recovery_state.pt','execution_cost.json'))
        metrics=json.loads((p/'metrics.json').read_text(encoding='utf-8'))
        assert metrics['full_attributed_pipeline_seconds']>=metrics['training_seconds']
        assert metrics['unique_incurred_pipeline_seconds']>=metrics['incurred_this_job_training_seconds']

def test_saved_reference_convergence_and_fourier_no_aliasing():
    record=json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
    assert record['status']=='PASS' and record['selected_common_truncation']==[-64,96]
    for case in ('bridge','T10','m10'):
        row=next(r for r in record['comparisons'] if r['case']==case)
        assert row['pass_check'] and row['full_complex_zero_padded_relative_difference']<1e-6
        assert max(row['lower_sixteen_layers_max_total_occupation'],row['upper_sixteen_layers_max_total_occupation'])<1e-8
        cfg=configuration(case)
        with np.load(RAW/'references'/case/'reference_test_grid.npz') as r:
            assert r['state'].shape==(201,161,2)
            assert abs(np.sum(abs(r['state'])**2,(1,2))-1).max()<1e-12
        assert cfg.test_x_points>2*(cfg.n_max-cfg.n_min)

def test_phase_diagnostics_initial_weighting_no_alignment_and_full_evaluation_structure():
    from src.r13_phase_evaluate import evaluate_carrier
    from src.r13_evaluate import evaluate
    cfg=configuration('bridge');model=R13Model(cfg,'exact','hard_total_norm',600)
    phase,series,dense,table=evaluate_carrier(model.carrier,cfg)
    assert phase['initial_weighted_unwrapped_phase_rms']==0
    assert phase['carrier_free_pde_residual_rms']<1e-14 and phase['no_phase_alignment']
    assert abs(dense['initial_branch_occupation_weight'].sum()-1)<4e-16
    metrics,series,dense,tables=evaluate(model,cfg,full=True)
    assert dense['residual_complex_unweighted'].shape==(4001,161,2)
    assert np.array_equal(dense['exterior_residual_complex'],-dense['exterior_positive_coupling_source_complex'])
    assert np.array_equal(dense['phase_reference_active_mask'],dense['branch_occupations_reference']>=1e-6)
    assert metrics['max_norm_drift']<1e-14
    assert not metrics['residual_numerical_estimate_is_continuous_certificate']
    assert metrics['fourier_integral_no_aliasing'] and 'phase_error_series.csv' in tables
    assert metrics['training_forward_equals_evaluation_forward']
    assert not metrics['posthoc_normalization']
