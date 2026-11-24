from __future__ import annotations
from pathlib import Path
from .r9_config import R9Config, THRESHOLDS, ARTIFACTS

EXPERIMENT = "r11_independent_replication_ablation"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/"results/raw"/EXPERIMENT
FIGURES = ROOT/"results/figures"/EXPERIMENT
DOCS = ROOT/"docs/experiments"/EXPERIMENT
LOGS = ROOT/"results/logs"/EXPERIMENT
SEEDS = (400, 401, 402, 403, 404)
GROUPS = ("A_pretrained_frozen", "B_random_frozen", "C_joint",
          "D_double_complex", "E_exact_carrier_diagnostic")
LABELS = dict(zip(GROUPS, ("A: pretrained frozen", "B: random frozen",
    "C: joint", "D: double Complex", "E: exact diagnostic")))
CURVE_STEPS = (5000, 10000, 15000, 20000, 26000)

def configuration():
    return R9Config(n_min=-48, n_max=80)

def budget(group):
    if group not in GROUPS:
        raise ValueError(group)
    return (6000, 20000) if group == GROUPS[0] else (
        (0, 20000) if group == GROUPS[4] else (0, 26000))

def run_directory(group, seed, *, smoke=False):
    return RAW/("smoke" if smoke else "formal")/group/f"seed_{seed}"

def snapshot_directory(group, seed, coupled_step):
    return run_directory(group, seed)/"snapshots"/f"coupled_{coupled_step:05d}"

def protocol():
    return dict(experiment=EXPERIMENT, seeds=list(SEEDS), configuration=configuration().as_dict(),
        groups={g:dict(carrier_steps=budget(g)[0], coupled_steps=budget(g)[1],
                       total_steps=sum(budget(g)), analytic_prior=(g == GROUPS[4])) for g in GROUPS},
        primary_groups=list(GROUPS[:4]), primary_total_budget=26000,
        equal_coupled_snapshot=20000, curve_steps=list(CURVE_STEPS),
        full_evaluation_coupled_steps=[20000, 26000], fixed_checkpoints_only=True,
        thresholds=THRESHOLDS, required_joint_pass_count=4, population_sd_ddof=0,
        learned_groups_no_explicit_frequency=True, exact_carrier_exception_group=GROUPS[4],
        time_stream_seeds=dict(carrier=910000, coupled=920000),
        initialization_seeds=dict(carrier=900000, envelope=800000, second_complex=830000),
        random_streams_independent=True, coupled_common_prefix_steps=20000,
        fresh_uniform_sorted_times_per_step=32, all_retained_modes=True,
        coupled_loss="unweighted_complete_spinor_PDE", carrier_loss="R9_unit_spinor_free_PDE_mean",
        time_derivative="exact_tanh_chain_rule_with_parameter_graph",
        checkpoint_selection=False, accuracy_tuning=False, selective_reruns=False,
        rescue_structures=False, dynamics_seed=400,
        continuous_error_certification=False,
        scope="known finite Fourier Gaussian initial spectrum, fixed weak periodic potential")
