from __future__ import annotations
from .r9_config import ROOT, RAW as R9_RAW, MODELS as R9_MODELS, SEEDS, ARTIFACTS

EXPERIMENT = "r10_wavepacket_budget_diagnosis"
MODEL = "A_frozen_carrier_continuation"
BASE_MODEL = R9_MODELS[0]
RAW = ROOT/"results/raw"/EXPERIMENT
FIGURES = ROOT/"results/figures"/EXPERIMENT
LOGS = ROOT/"results/logs"/EXPERIMENT
DOCS = ROOT/"docs/experiments"/EXPERIMENT
BASE_GLOBAL_STEP = 20000
ADDITIONAL_STEPS = 6000
FINAL_GLOBAL_STEP = BASE_GLOBAL_STEP+ADDITIONAL_STEPS


def run_directory(seed, *, smoke=False):
    return RAW/("smoke" if smoke else "formal")/f"seed_{seed}"


def source_directory(seed):
    return R9_RAW/"formal"/BASE_MODEL/f"seed_{seed}"
