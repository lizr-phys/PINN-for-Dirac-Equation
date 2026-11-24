from __future__ import annotations
import hashlib,json,math,os
from dataclasses import dataclass,replace
from pathlib import Path

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
SEEDS=(810,811,812,813,814)
DEV_SEEDS=(910,911)
CORE=('T10','m10','m5T5')
EXTENSION=('V04','two_harmonics','chirped')
CASES=CORE+EXTENSION
GROUPS=('original_pde','static_pde','original_hard','learned_hard','analytic_hard','joint_hard','no_carrier_hard')
THRESHOLDS={'spinor_rel_l2_gauge_aligned':.10,'current_rel_l2':.15,'fidelity_mean':.98,'modal_population_rel_l2':.10,'max_norm_drift':.02}
TUNING={
 'siren':({'omega':30.,'lr':.001},{'omega':30.,'lr':.0001},{'omega':60.,'lr':.0001}),
 'fourier':({'scale':1.,'lr':.001},{'scale':4.,'lr':.001},{'scale':8.,'lr':.0003})}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)

def freeze_design():
    p=BASE/'frozen/design.json'
    obj=protocol()
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8'))!=json.loads(json.dumps(obj)):raise RuntimeError('Scientific design changed')
    else:
        write(p,obj);write(BASE/'frozen/design_sha256.json',{'sha256':sha(p),'frozen_before_reference_or_training':True})
    return sha(p)

def protocol():
    return dict(study='r14_prewriting_closure_2026_10_11',formal_seeds=SEEDS,development_seeds=DEV_SEEDS,
      core_cases=CORE,extension_cases=EXTENSION,mechanism_groups=GROUPS,thresholds=THRESHOLDS,
      required_joint_successes=4,all_five_seeds_required=True,success_scope='separate per case, group, and prescribed endpoint',
      reference_policy=dict(candidate_bounds=((-64,96),(-80,112),(-96,128),(-112,144),(-128,160)),
        comparison='choose first smaller passing ALL six cases vs next larger',trajectory_relative_error_max_exclusive=1e-6,
        boundary_sixteen_layer_occupation_max_exclusive=1e-8,hermiticity_max=0.,validation_error_max_exclusive=1e-12,
        dt=.00125,test_t_points=201,time_grid_validation_points=401,test_x_points=1024,double_x_points=2048,
        no_learning_metric_used=True,all_fail_stop=True,fixed_feature_scale=10.),
      initial_data=dict(original='unchanged normalized Gaussian on n=0..32, k0=2, sigma=.5, equal +/- branches; zero padding',
        chirped='same amplitudes; both branches multiplied by exp(0.5 i (k-2)^2), minus branch additionally exp(i*pi/3); no renormalization'),
      physical=dict(length='16*pi',hbar=1.,c=1.,k='n/8',mass='10 for m10, else 5',T='10 for T10, else 5',
        potential='0.4*cos(x) for V04; 0.2*(cos(x)+cos(2*x))/sqrt(2) for two_harmonics; else 0.2*cos(x)'),
      mechanism=dict(network='4 hidden layers, width64, tanh, float64/complex128',Adam_lr=.001,
        carrier_steps=6000,coupled_primary_steps=20000,fresh_sorted_uniform_times=32,all_modes=True,
        hard_norm='full complex branch spectrum r/sqrt(sum |r|^2); exact derivative; no clamp; forward used in loss and evaluation',
        original_carrier='exp(i tau q(k,tau))',static_carrier='exp(-i t omega(k)); one-input net initialized to original t=0 rate',
        analytic_carrier='exp(-i t (+/- sqrt(k^2+m^2))); explicit known-free-operator prior',
        learned_carrier_training='complete H0 spinor residual, both nonzero unit initial branches; no energy labels or reference',
        primary_equal_coupled_budget=True,initialization_streams={'envelope':800000,'carrier':900000},
        sampling_streams={'coupled':920000,'free':910000},
        joint='static carrier and same envelope from scratch; no free loss; joint full PDE; primary 20000, prescribed 26000 cost-budget endpoint',
        no_carrier='same single tanh complex envelope, same full hard norm; primary20000, prescribed26000 endpoint',
        factors='original/static pretrained carrier x PDE-only/hard norm; analytic hard; joint static hard; no carrier hard',
        pairing='bitwise same-shaped envelope initialization, same time RNG, phase/rate-matched carrier initial condition',
        inference_precomputed_carrier='none; all network and carrier evaluations included'),
      baselines=dict(families=['siren','fourier'],architecture='spectral PINN adaptations with same hard IC, full hard norm and PDE, not literal published solver reproduction',
        tuning_configs=TUNING,tuning_cases=['T10','m10'],tuning_seeds=DEV_SEEDS,tuning_steps=4000,
        final_steps=26000,primary_20000_snapshot=True,selected_without_formal_metrics=True,
        candidate_score='mean dev spinor/.10 + current/.15 + modal/.10 + (1-fidelity)/.02; all failed last, fixed index tie-break',
        per_core_case_selection=True,global_selection_for_m5T5_and_extensions=True,
        strongest_family='best global development score, fixed family order tie-break',
        no_second_search=True,no_reference_labels_in_gradient_training=True),
      extensions=dict(groups=['learned_hard','analytic_hard','strongest_neural'],seeds=SEEDS,
        learned_carrier_reuse='m5T5 static same seed; same m,k,domain and T, only V or IC differs; envelope retrained from matched new Adam',
        separate_retrain_and_reuse_labels=True,zero_shot='evaluate unchanged m5T5 complete solution checkpoint on V04/two_harmonics/chirped; no correction of IC or weights; all 3 methods and 5 seeds; diagnostic only',
        zero_shot_no_accuracy_selection=True),
      spectral_integrator=dict(method='Strang time-splitting Fourier pseudospectral (TSFP2)',
        grid_points=[256,512,1024],time_steps=[.025,.0125,.00625,.003125,.0015625,.00078125],
        per_case_repeats=5,known_H0_prior=True,include_setup_fft_free_exp_propagation=True,
        no_training=True,reference_diagonalization='independent calibration, not algorithm timing'),
      snapshots=dict(coupled=[5000,10000,15000,20000,26000],carrier=[1000,3000,6000],fixed_endpoint_only=True,
        save_dense_test_spectrum_all=True,save_full_fine_arrays=False,
        save_full_residual_summary_and_201point_complex_residual=True,reconstruct_fine_from_checkpoint=True),
      statistics=dict(ddof=0,paired_all_seeds=True,primary_pair='learned_hard vs analytic_hard at20000 in T10 and m10',
        exact_sign_flip_two_sided=True,confidence_interval='paired difference Student-t df4 95%; descriptive assumptions; all seeds shown',
        learning_advantage_gate='in EACH T10 and m10, paired spinor mean difference<0, upper95CI<0, >=4/5 lower error, absolute pass; lower total attributed training time for efficiency claim',
        small_n_limitation=True,no_cross_case_pooling=True,no_outlier_deletion=True),
      costs=dict(record=['training_wall_seconds','process_cpu_seconds','setup_seconds','evaluation_seconds','checkpoint_io_seconds',
        'inference_median_seconds_201_times_all_modes','inference_samples','peak_RSS','active_frozen_total_parameters',
        'unit_spinor_residual_points','complex_scalar_residual_components','auxiliary_pretraining','development_tuning'],
        incurred='each unique carrier and tuning run once',attributed='standalone method includes full6000 auxiliary cost; tuning cost separately and as all-benchmark-development overhead',
        reuse='actual m5 carrier reuse disclosed; amortized examples1,5,10,100; no break-even claims without measured benefit',
        concurrent_wall_time_is_not_isolated_speedup=True),
      resources=dict(workers=5,physical_core_masks=[1,2,4,8,16],control_mask=128,threads_per_worker=1,
        reserve_free_E_gib=4.,estimated_new_storage_gib=4.5,estimated_peak_RAM_gib=6.,
        preflight_time_estimate='historical coupled4000-7200 sec, carrier~1400 sec; 300-400 bound-core hours + evaluation; 3-5 days at five workers',
        exact_recovery_every_steps=500,batch_sequence=['references','development','core:T10','core:m10','core:m5T5','extensions','zero_shot','independent_audit','report']),
      stop_conditions=['reference/calibration fails: no formal training','numeric failure retained, no fresh rerun; dependencies marked failed',
        'engineering interruption: exact optimizer/weights/RNG resume, same plan and budget','free E space below4GiB: stop checkpoint boundary, operational storage correction only',
        'all preset attempts, verification, visual QA, literature and links executed: finish, regardless of scientific failures'],
      forbidden=['historical evidence overwrite','accuracy-driven budget extension','seed selection','checkpoint selection','reference training leakage',
        'claim structural norm conservation demonstrates dynamics','claim pretraining necessary without comparison','claim all failures uniquely explained'],
      deliverables=['total_report.md','claims_evidence.csv','historical_results.csv','per_seed_metrics.csv','paired_statistics.csv',
        'reference_convergence.csv','ablation_and_cost_curves','literature_prior_art_map.csv','references.bib','code_data_manifest.json',
        'independent_metric_audit.json','tests_and_recovery.json','visual_QA.md','link_check.json','reproduce.md','submission_positioning.md'])

if __name__=='__main__':print(freeze_design())
