from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .r12_config import R12Config
from .r9_config import THRESHOLDS, ARTIFACTS

EXPERIMENT = 'r13_long_time_recovery'
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/raw'/EXPERIMENT
FIGURES = ROOT/'results/figures'/EXPERIMENT
LOGS = ROOT/'results/logs'/EXPERIMENT
DOCS = ROOT/'docs/experiments'/EXPERIMENT
CASES = ('bridge', 'T10', 'm10')
EXPLORATION_SEEDS = (600, 601, 602)
VALIDATION_SEEDS = (700, 701, 702, 703, 704)
CARRIER_KINDS = ('original', 'scaled_phase', 'static_frequency')
ENVELOPE_KINDS = ('pde_only', 'norm_regularized', 'hard_total_norm')
NORM_WEIGHT = .01
CURVE_STEPS = (5000, 10000, 15000, 20000)
FREE_CURVE_STEPS = (1000, 3000, 6000)

@dataclass(frozen=True)
class R13Config(R12Config):
    n_min: int = -64
    n_max: int = 96
    case: str = 'T10'
    norm_regularization_weight: float = NORM_WEIGHT

def configuration(case, bounds=(-64, 96)):
    if case not in CASES:
        raise ValueError(case)
    return R13Config(case=case, n_min=bounds[0], n_max=bounds[1],
        mass=10. if case == 'm10' else 5., t_max=10. if case == 'T10' else 5.,
        residual_time_points=8001 if case == 'T10' else 4001)

def carrier_directory(stage, case, kind, seed, *, smoke=False):
    return RAW/('smoke' if smoke else 'formal')/'carriers'/stage/case/kind/f'seed_{seed}'

def run_directory(stage, case, group, seed, *, smoke=False):
    return RAW/('smoke' if smoke else 'formal')/'coupled'/stage/case/group/f'seed_{seed}'

def reference_policy():
    return dict(fixed_smaller=[-64,96], fixed_larger=[-80,112], boundary_layers=16,
        full_complex_zero_padded_relative_difference_max_exclusive=1e-6,
        each_boundary_total_occupation_max_exclusive=1e-8, validation_tolerance=1e-12,
        hermiticity_max_abs=0., time_spacing=.00125, no_pinn_based_selection=True,
        all_failed_stop_formal_training=True, no_adaptive_truncation=True)

def protocol():
    return dict(experiment=EXPERIMENT, cases={c:configuration(c).as_dict() for c in CASES},
        exploration_seeds=EXPLORATION_SEEDS, validation_seeds=VALIDATION_SEEDS,
        carrier_candidates=CARRIER_KINDS, envelope_candidates=ENVELOPE_KINDS,
        carrier_pretraining_steps=6000, coupled_steps=20000, optimizer='Adam', learning_rate=.001,
        fresh_times_per_step=32, all_modes=True, precision='float64/complex128',
        coupled_loss='mean_t sum_n mean_spin_component abs(full_spinor_residual)^2 / L',
        carrier_loss='mean of full free spinor PDE residuals for two unit initial eigenbranches',
        norm_regularization_weight=NORM_WEIGHT,
        norm_regularization='weight * mean_t (sum_all_modes_and_branches abs(c)^2 - 1)^2',
        hard_norm='r / sqrt(sum_all_retained_modes_and_branches abs(r)^2), then unit carrier',
        hard_norm_zero_denominator='preserve numerical failure; no clamp or threshold',
        scaled_phase='exp(i tau (m*T) qbar(k,tau)); output layer initialized as original/(m*T)',
        static_frequency='exp(-i t omega_theta(k)); autonomous one-input network; no analytic labels',
        static_initialization='original time network at tau=0, first k column, final layer multiplied by -1/T',
        initialization_streams=dict(carrier=900000,envelope=800000),
        sampling_streams=dict(free=910000,coupled=920000),
        pairing='same-shaped envelope weights bitwise; carriers matched physical initial phase/rate; normalized time prefix',
        complete_carrier_weight_pairing=False, initialization_and_sampling_independent=True,
        freeze_carrier_weights_preserve_time_derivative=True, no_future_reference_in_training=True,
        exact_frequency_only='explicit E and independent evaluation',
        thresholds=THRESHOLDS, required_joint_pass_count=4, population_sd_ddof=0,
        absolute_success_separately_per_condition=True, exploratory_joint_pass_count=2,
        curve_steps=CURVE_STEPS, free_curve_steps=FREE_CURVE_STEPS, final_checkpoint_only=True,
        dynamics_exploration_seed=600, dynamics_validation_seed=700,
        stages=['bridge', 'free_carriers', 'exact_envelopes', 'T10_factorial', 'independent_validation'],
        carrier_selection=dict(
            eligible='initial-Gaussian weighted unwrapped phase RMS mean lower than original in T10 and m10; at least 2/3 paired T10 improvements; full unit residual RMS mean no worse in both cases',
            rank=['T10 mean initial-weighted unwrapped phase RMS','T10 mean weighted unit residual RMS','fixed candidate order'],
            diagnostic_fallback='same rank, explicitly ineligible; no independent combination claim'),
        envelope_selection=dict(
            eligible='all five mean criteria and at least 2/3 joint passes in EACH T10 and m10; mean norm drift lower than E PDE-only in T10',
            rank=['T10 mean dynamic score','T10 mean norm drift','fixed candidate order'],
            dynamic_score='spinor/0.10 + current/0.15 + modal/0.10 + (1-fidelity)/0.02',
            diagnostic_fallback='same rank, explicitly ineligible'),
        combination_selection='select at most one; both component gates plus all five T10 means and at least 2/3 T10 joint passes; no backup combination',
        independent_validation_if_no_combination='A and E only; preserve failed exploration, no rescue',
        bridge=['A_original','E_pde_only'],
        factorial=['A_original','carrier_only','envelope_only','combined','E_pde_only_alias','E_repaired_envelope_alias'],
        matched_E_envelope_cost_comparison='reuse prescribed E envelope probe; no new training; separate from original E validation',
        independent=['A_original','selected_combination_if_eligible','E_pde_only'],
        actual_max_unique_carrier_jobs=41, actual_max_unique_coupled_jobs=66,
        planned_unique_optimizer_steps_max=1566000,
        pretraining_reuse='exploration carrier checkpoint shared within same case/seed; validation fresh sources',
        costs='both unique incurred experiment cost and full standalone attributed algorithm pretraining cost',
        masks_exploration_workers=3, masks_validation_workers=5, threads_per_worker=1,
        balanced_group_rotation=True, recovery_every_steps=500,
        preserve_all_failures=True, selective_reruns=False, accuracy_budget_extension=False,
        predictive_normalization=False, time_windows=False, representation_switch=False,
        continuous_residual_certificate=False, originality_claim=False)
