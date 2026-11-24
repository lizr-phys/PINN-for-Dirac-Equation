from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from .train_periodic_potential_coupled_ltpb import ROOT, RAW, load_formal_config, write_json

MANIFEST = RAW/"frozen_protocol_and_code.json"
CORE_FILES = (
    "PERIODIC_POTENTIAL_COUPLED_LTPB_PREREGISTRATION.md",
    "PERIODIC_POTENTIAL_COUPLED_LTPB_AMENDMENT_01.md",
    "PERIODIC_POTENTIAL_COUPLED_LTPB_FIGURE_CONTRACT.md",
    "src/periodic_potential_config.py", "src/periodic_potential_physics.py",
    "src/periodic_potential_models.py", "src/periodic_potential_evaluate.py",
    "src/periodic_potential_coupled_ltpb_decision.py", "src/train_periodic_potential_coupled_ltpb.py",
    "src/run_periodic_potential_coupled_ltpb.py", "src/launch_periodic_potential_coupled_ltpb.py",
    "src/aggregate_periodic_potential_coupled_ltpb.py", "src/calibrate_periodic_potential.py",
    "src/verify_periodic_potential_reference.py", "src/select_periodic_potential_formal_physics.py",
    "src/freeze_periodic_potential_coupled_ltpb.py",
    "src/capture_periodic_potential_coupled_ltpb_protected_baseline.py",
    "src/physics.py", "src/wavepacket_models.py", "src/wavepacket_physics.py", "src/wavepacket_config.py",
    "src/train_wavepacket_shared_spectral_ltpb.py",
    "tests/test_periodic_potential_models.py", "tests/test_periodic_potential_reference.py",
    "tests/test_periodic_potential_training_evaluation.py",
    "results/raw/periodic_potential_coupled_ltpb/formal_selection.json",
    "results/raw/periodic_potential_coupled_ltpb/calibration/calibration.json",
    "results/raw/periodic_potential_coupled_ltpb/reference_verification.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    load_formal_config()
    if MANIFEST.exists():
        raise FileExistsError("Refusing to replace frozen protocol/code")
    if (RAW/"learned").exists():
        raise RuntimeError("Freeze must precede all formal learned runs")
    from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS
    smoke_root = ROOT/"results/raw/periodic_potential_coupled_ltpb_smoke"
    for model in PERIODIC_POTENTIAL_MODELS:
        smoke = smoke_root/model/"seed_0"
        if not (smoke/"model_state.pt").is_file() or not (smoke/"metrics.json").is_file():
            raise RuntimeError("Smoke artifact check has not completed")
    result = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Calibration, approved physics, model, training, evaluation, statistical decision, runner, scientific tests and runtime dependencies",
              "reporting_scope": "Plot rendering and narrative report scripts are output-only; they cannot change frozen metrics or decisions and receive separate provenance hashes",
              "files": {name: sha(ROOT/name) for name in CORE_FILES}}
    write_json(MANIFEST, result)
    return result


def verify_freeze() -> str:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = [name for name, expected in data["files"].items() if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
    if changed:
        raise RuntimeError(f"Frozen scientific protocol/code changed: {changed}")
    return sha(MANIFEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        freeze()
    print(verify_freeze())
