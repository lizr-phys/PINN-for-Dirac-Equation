from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
from .r9_config import R9Config, THRESHOLDS, ARTIFACTS

EXPERIMENT = 'r12_cross_condition_robustness'
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/raw'/EXPERIMENT
FIGURES = ROOT/'results/figures'/EXPERIMENT
LOGS = ROOT/'results/logs'/EXPERIMENT
DOCS = ROOT/'docs/experiments'/EXPERIMENT
SEEDS = (500, 501, 502, 503, 504)
CASES = ('T10', 'm10', 'V04', 'two_harmonics')
GROUPS = ('A_pretrained_frozen', 'C_joint', 'D_double_complex', 'E_exact_carrier_diagnostic')
LABELS = dict(zip(GROUPS, ('A: pretrained frozen', 'C: joint', 'D: double Complex', 'E: exact diagnostic')))
CASE_LABELS = dict(zip(CASES, ('T = 10', 'm = 10', 'V = 0.4 cos(x)', 'Equal RMS: two harmonics')))
CURVE_STEPS = (5000, 10000, 15000, 20000, 26000)
TRUNCATIONS = ((-48, 80), (-64, 96), (-80, 112), (-96, 128), (-112, 144), (-128, 160))

@dataclass(frozen=True)
class R12Config(R9Config):
    case: str = 'T10'
    potential_harmonics: tuple = ((1, .2),)

    @property
    def coupling_terms(self):
        return tuple((8*q, amplitude/2) for q, amplitude in self.potential_harmonics)

    @property
    def max_coupling_shift(self):
        return max(shift for shift, _ in self.coupling_terms)

    def as_dict(self):
        return super().as_dict() | dict(potential_harmonics=self.potential_harmonics,
            coupling_terms=self.coupling_terms, potential_rms=math.sqrt(sum(a*a/2 for _, a in self.potential_harmonics)),
            initial_support=[0,32], zero_padding_only=True)

def configuration(case, bounds=None):
    if case not in CASES:
        raise ValueError(case)
    if bounds is None:
        record = json.loads((RAW/'reference_validation.json').read_text(encoding='utf-8'))
        if record['status'] != 'PASS':
            raise RuntimeError('Reference calibration failed; formal training forbidden')
        bounds = record['selected_common_truncation']
    return R12Config(case=case, n_min=bounds[0], n_max=bounds[1],
        mass=10 if case == 'm10' else 5, t_max=10 if case == 'T10' else 5,
        potential_strength=.4 if case == 'V04' else .2,
        potential_harmonics=((1,.2/math.sqrt(2)),(2,.2/math.sqrt(2))) if case == 'two_harmonics' else
            ((1,.4 if case == 'V04' else .2),), residual_time_points=8001 if case == 'T10' else 4001)

def budget(group):
    if group not in GROUPS:
        raise ValueError(group)
    return (6000,20000) if group == GROUPS[0] else (0,26000)

def run_directory(case, group, seed, *, smoke=False):
    return RAW/('smoke' if smoke else 'formal')/case/group/f'seed_{seed}'

def snapshot_directory(case, group, seed, step):
    return run_directory(case,group,seed)/'snapshots'/f'coupled_{step:05d}'

def group_order(case, seed):
    rotation = (CASES.index(case)+SEEDS.index(seed)) % len(GROUPS)
    return GROUPS[rotation:]+GROUPS[:rotation]

def reference_policy():
    return dict(candidate_truncations=TRUNCATIONS, compare_adjacent_in_order=True,
        common_truncation='first smaller candidate passing every case',
        zero_padded_complex_trajectory_relative_difference_max_exclusive=1e-6,
        each_outer_sixteen_layer_total_occupation_max_exclusive=1e-8,
        boundary_layers=16, fixed_time_spacing=.00125,
        hermiticity_max=0., validation_tolerance=1e-12,
        no_pinn_based_selection=True, all_failed_stop_formal_training=True)

def protocol():
    return dict(experiment=EXPERIMENT, cases={c:configuration(c).as_dict() for c in CASES},
        seeds=SEEDS, formal_run_count=80, groups={g:dict(carrier_steps=budget(g)[0],
        coupled_steps=budget(g)[1], total_steps=sum(budget(g)), analytic_prior=g==GROUPS[-1]) for g in GROUPS},
        reference_policy=reference_policy(), primary_groups=GROUPS[:3], thresholds=THRESHOLDS,
        required_joint_pass_count=4, population_sd_ddof=0, absolute_success_per_case=True,
        equal_coupled_snapshot=20000, curve_steps=CURVE_STEPS, full_residual_steps=(20000,26000),
        dynamics_seed=500, pairing='same shapes bitwise initialization; common normalized time uniforms',
        initialization_seeds=dict(carrier=900000,envelope=800000,second_complex=830000),
        time_stream_seeds=dict(carrier=910000,coupled=920000),
        coupled_common_prefix_steps=20000, normalized_time_pairing_across_cases=True,
        coupled_loss='R11 unweighted complete spinor PDE: mean_t sum_n mean_component |R|^2 / L',
        carrier_loss='R11 mean of two unit-branch free spinor residuals',
        fresh_times_per_step=32, all_modes=True, double_precision=True, optimizer='Adam', learning_rate=.001,
        no_future_reference_in_training=True, analytic_phase_only_group=GROUPS[-1],
        fixed_endpoints=True, accuracy_tuning=False, selective_reruns=False, rescue_structures=False,
        batch_order=CASES, orders={c:{str(s):group_order(c,s) for s in SEEDS} for c in CASES},
        numerical_residual_integrals_are_continuous_certificates=False,
        scope='retraining on four specified conditions with known finite Fourier Gaussian initial data; no zero-shot claim')
