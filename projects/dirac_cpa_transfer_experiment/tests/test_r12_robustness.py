from __future__ import annotations
import ast
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from src.r12_config import CASES,GROUPS,SEEDS,ROOT,RAW,configuration,budget,protocol,group_order
from src.r12_models import R12Model,hashes,coupled_parameters,parameter_counts
from src.r12_residual import training_loss,coupled_residual,spinor_value_and_time_derivative,free_carrier_residual,carrier_value_and_time_derivative
from src.r12_hamiltonian import build_hamiltonian,hamiltonian_apply,potential_apply,exterior_residual
from src.r9_reference import initial_numpy
from src.r9_models import original_gaussian,initial_coefficients
from src.r8_sampling import TimeSampler
from src.r8_loss import derivative
from src.train_wavepacket_shared_spectral_ltpb import _set_runtime

_set_runtime(1,1)

def points(cfg):
    k=torch.tensor(cfg.wave_numbers).reshape(-1,1).repeat(2,1)
    t=torch.tensor([.13*cfg.t_max,.83*cfg.t_max]).reshape(-1,1).repeat_interleave(cfg.mode_count,0).requires_grad_()
    return k,t

@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('group',GROUPS)
def test_hard_initial_full_time_derivative_and_all_parameter_gradients(case,group):
    cfg=configuration(case);model=R12Model(cfg,group,502)
    if group==GROUPS[0]:model.carrier.freeze()
    k0=torch.tensor(cfg.wave_numbers)
    np.testing.assert_allclose(model(k0,torch.zeros_like(k0)).detach().numpy(),initial_numpy(cfg),atol=1e-15,rtol=0)
    k,t=points(cfg);state,dt=spinor_value_and_time_derivative(model,cfg,k,t)
    auto=derivative(model(k,t),t,True)
    assert state.dtype==dt.dtype==torch.complex128
    assert torch.allclose(state,model(k,t),atol=1e-14,rtol=1e-14)
    assert torch.allclose(dt,auto,atol=3e-13,rtol=3e-13)
    residual=coupled_residual(model,cfg,k,t)
    ad_residual=1j*auto.reshape(2,cfg.mode_count,2)-hamiltonian_apply(model(k,t).reshape(2,cfg.mode_count,2),cfg)
    loss,power=training_loss(model,cfg,k,t)
    assert torch.allclose(residual,ad_residual,atol=4e-13,rtol=4e-13)
    assert power.shape==(cfg.mode_count,)
    params=coupled_parameters(model)
    a=torch.autograd.grad(loss,params)
    b=torch.autograd.grad(ad_residual.abs().square().mean(-1).sum(-1).mean()/cfg.length,params)
    assert all(torch.isfinite(x).all() and torch.allclose(x,y,atol=3e-11,rtol=3e-11) for x,y in zip(a,b))
    if group==GROUPS[1]:assert sum(x.abs().sum() for x in a[:len(tuple(model.carrier.parameters()))])>0
    if group==GROUPS[0]:
        p,pt=carrier_value_and_time_derivative(model.carrier,cfg,k,t)
        assert pt.abs().max()>1e-5 and free_carrier_residual(model.carrier,cfg,k,t).abs().max()>1
        assert all(not v.requires_grad and v.grad is None for v in model.carrier.parameters())
    if group==GROUPS[-1]:assert free_carrier_residual(model.carrier,cfg,k,t).abs().max()<3e-14

@pytest.mark.parametrize('case',CASES)
def test_coupling_hermiticity_independent_matrix_exterior_overlap_and_initial_mass(case):
    cfg=configuration(case);h=build_hamiltonian(cfg);rng=np.random.default_rng(321)
    raw=rng.normal(size=(3,cfg.mode_count,2))+1j*rng.normal(size=(3,cfg.mode_count,2))
    assert np.array_equal(h,h.conj().T)
    np.testing.assert_allclose(hamiltonian_apply(raw,cfg),(raw.reshape(3,-1)@h.T).reshape(raw.shape),atol=3e-14,rtol=3e-14)
    np.testing.assert_allclose(hamiltonian_apply(torch.tensor(raw),cfg).numpy(),hamiltonian_apply(raw,cfg),atol=3e-14,rtol=3e-14)
    assert h[0,2]==0 and h[0,16]==dict(cfg.coupling_terms)[8]
    assert h[0,32]==(dict(cfg.coupling_terms).get(16,0))
    exterior,source,ns=exterior_residual(raw,cfg);w=cfg.max_coupling_shift
    big_cfg=replace(cfg,n_min=cfg.n_min-w,n_max=cfg.n_max+w)
    padded=np.zeros((3,big_cfg.mode_count,2),dtype=np.complex128);padded[:,w:w+cfg.mode_count]=raw
    outside=np.r_[np.arange(w),np.arange(w+cfg.mode_count,big_cfg.mode_count)]
    np.testing.assert_allclose(exterior,-hamiltonian_apply(padded,big_cfg)[:,outside],atol=2e-15,rtol=2e-15)
    assert np.array_equal(exterior,-source) and len(ns)==2*w
    c=initial_coefficients(cfg);g=original_gaussian()
    assert torch.equal(c[-cfg.n_min:33-cfg.n_min,0],g.to(torch.complex128)/2**.5)
    assert abs(c.abs().square().sum()-1)<3e-16
    assert torch.equal(c,initial_coefficients(replace(cfg,n_min=cfg.n_min-16,n_max=cfg.n_max+16))[16:16+cfg.mode_count])
    if case=='m10':assert not np.allclose(initial_numpy(cfg),initial_numpy(replace(cfg,mass=5)))

def test_rms_and_only_authorized_physics_changes():
    assert configuration('T10').t_max==10 and configuration('m10').mass==10
    assert configuration('V04').potential_harmonics==((1,.4),)
    assert abs(configuration('two_harmonics').as_dict()['potential_rms']-.2/2**.5)<1e-16
    assert configuration('two_harmonics').t_max==5 and configuration('two_harmonics').mass==5

def test_pairing_parameters_and_independent_sampling_streams_across_conditions():
    models=[R12Model(configuration(c),g,500) for c in CASES for g in GROUPS]
    assert len({hashes(m)['envelopes'][0] for m in models})==1
    assert len({hashes(m)['carrier'] for m in models if m.model_name in GROUPS[:2]})==1
    models[0].carrier.freeze()
    assert [parameter_counts(m)['total_parameters'] for m in models[:4]]==[25734,25734,25864,12932]
    assert [parameter_counts(m)['active_parameters'] for m in models[:4]]==[12932,25734,25864,12932]
    samplers=[TimeSampler(configuration(c),500) for c in CASES]
    free=TimeSampler(configuration('T10'),500,carrier=True)
    for _ in range(4):free.sample()
    for _ in range(3):
        for sampler in samplers:sampler.sample()
        assert all(torch.equal(s.normalized_trace[-1],samplers[0].normalized_trace[-1]) for s in samplers)
    assert not torch.equal(free.normalized_trace[0],samplers[0].normalized_trace[0])
    fresh=R12Model(configuration('T10'),GROUPS[0],500)
    assert hashes(fresh)==hashes(models[0])
    saved=samplers[0].state_dict();restored=TimeSampler(configuration('T10'),500);restored.load_state_dict(saved)
    assert torch.equal(restored.sample()[1],samplers[0].sample()[1])

def test_protocol_complete_independent_seeds_fixed_budget_and_no_cross_case_average():
    p=protocol();assert p['formal_run_count']==80 and SEEDS==(500,501,502,503,504)
    assert [budget(g) for g in GROUPS]==[(6000,20000),(0,26000),(0,26000),(0,26000)]
    assert p['dynamics_seed']==500 and p['required_joint_pass_count']==4
    assert all(sorted(group_order(c,s))==sorted(GROUPS) for c in CASES for s in SEEDS)
    from src.r12_decision import absolute_success
    frame=pd.DataFrame(dict(seed=SEEDS,spinor_rel_l2_gauge_aligned=[.02]*5,current_rel_l2=[.03]*5,
        fidelity_mean=[.99]*5,modal_population_rel_l2=[.02]*5,max_norm_drift=[.005]*5))
    assert absolute_success(frame)['pass_check']
    frame.loc[0,'max_norm_drift']=.03;assert absolute_success(frame)['pass_check']
    frame.loc[1,'max_norm_drift']=.03;assert not absolute_success(frame)['pass_check']

def test_saved_independent_reference_before_training_and_no_aliasing():
    r=json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
    assert r['status']=='PASS' and r['no_pinn_based_selection']
    bounds=r['selected_common_truncation']
    for case in CASES:
        cfg=configuration(case);row=next(row for row in r['comparisons'] if row['case']==case and row['smaller']==bounds)
        assert row['pass_check'] and row['full_complex_zero_padded_relative_difference']<1e-6
        assert row['lower_sixteen_layers_max_total_occupation']<1e-8 and row['upper_sixteen_layers_max_total_occupation']<1e-8
        with np.load(RAW/'references'/case/'reference_test_grid.npz') as ref:
            assert ref['state'].shape==(201,cfg.mode_count,2)
            assert abs(np.sum(abs(ref['state'])**2,(1,2))-1).max()<1e-12
        assert cfg.test_x_points>2*(cfg.n_max-cfg.n_min)

def test_learned_paths_no_future_or_exact_phase_leakage():
    for name in ('r12_models.py','r12_residual.py','r12_hamiltonian.py','train_r12_robustness.py'):
        tree=ast.parse((ROOT/'src'/name).read_text(encoding='utf-8'))
        assert not any(isinstance(n,ast.ImportFrom) and 'reference' in (n.module or '') for n in ast.walk(tree))
    for group in GROUPS:
        model=R12Model(configuration('m10'),group,500)
        assert not any(any(w in k for w in ('energy','frequency','eigenvalue','reference','target')) for k in model.state_dict())
        assert model.carrier is None or bool(getattr(model.carrier,'analytic_prior',False))==(group==GROUPS[-1])

@pytest.mark.parametrize('group',GROUPS[:2])
def test_exact_optimizer_sampling_resume_matches_direct(tmp_path,group):
    from src.train_r12_robustness import train_run
    direct,resumed=tmp_path/'direct',tmp_path/'resumed'
    train_run('two_harmonics',group,504,direct,smoke=True)
    train_run('two_harmonics',group,504,resumed,smoke=True,stop_after=1)
    train_run('two_harmonics',group,504,resumed,smoke=True)
    a,b=(torch.load(p/'recovery_state.pt',weights_only=False) for p in (direct,resumed))
    assert all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
    for key in ('free_sampler','coupled_sampler'):
        assert all(torch.equal(a[key][k],b[key][k]) for k in a[key])
    for key in a['optimizer']['state']:
        assert all(torch.equal(a['optimizer']['state'][key][v],b['optimizer']['state'][key][v]) for v in a['optimizer']['state'][key])

@pytest.mark.parametrize('case',('T10','two_harmonics'))
def test_evaluation_full_derivatives_masks_exterior_sign_and_disk_schema(case):
    from src.r12_evaluate import evaluate
    cfg=configuration(case);model=R12Model(cfg,GROUPS[-1],503)
    metrics,series,dense,tables=evaluate(model,cfg,full=True)
    assert dense['spectral_predicted_raw'].shape==(201,cfg.mode_count,2)
    assert dense['residual_complex_unweighted'].shape==(cfg.residual_time_points,cfg.mode_count,2)
    assert dense['full_physical_residual_complex'].shape==(cfg.residual_time_points,cfg.mode_count+2*cfg.max_coupling_shift,2)
    assert np.array_equal(dense['phase_reference_active_mask'],dense['branch_occupations_reference']>=1e-6)
    assert np.array_equal(dense['exterior_residual_complex'],-dense['exterior_positive_coupling_source_complex'])
    assert all(np.isfinite(a).all() for a in dense.values())
    assert metrics['fourier_integral_no_aliasing'] and not metrics['residual_numerical_estimate_is_continuous_certificate']
    assert metrics['hermitian_coupling_global_norm_projection_max_abs']<1e-12
    assert len(series)==201 and len(tables['norm_balance.csv'])==cfg.residual_time_points
