from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .r7_config import EXPERIMENT

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw" / EXPERIMENT
DOCS = ROOT / "docs/experiments" / EXPERIMENT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def capture_protected_history():
    destination = RAW / "protected_history.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    files = []
    for directory in ("src", "tests", "docs", "results"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if EXPERIMENT in path.parts or path.name.startswith(("r7_", "test_r7", "train_r7", "run_r7", "freeze_r7", "aggregate_r7", "plot_r7", "audit_r7", "build_r7")):
                continue
            files.append(path)
    files += [p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith("R7_")]
    record = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(files))}}
    write_json(destination, record)
    return record


def verify_history():
    record = json.loads((RAW / "protected_history.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in record["files"].items()
               if not (ROOT/name).is_file() or sha256(ROOT/name) != digest]
    return {"pass": not changed, "checked_files": len(record["files"]), "changed_or_missing": changed}


def science_files():
    names = ["r7_config.py", "r7_models.py", "r7_sampling.py", "r7_loss.py", "r7_reference.py", "r7_theory.py", "r7_runtime.py",
             "r7_evaluate.py", "r7_decision.py", "train_r7_phase_release.py", "run_r7_phase_release.py",
             "freeze_r7_phase_release.py", "periodic_potential_config.py", "periodic_potential_models.py",
             "periodic_potential_physics.py", "periodic_potential_evaluate.py", "periodic_potential_coupled_ltpb_decision.py",
             "wavepacket_models.py", "wavepacket_physics.py", "wavepacket_config.py", "physics.py",
             "train_wavepacket_shared_spectral_ltpb.py", "config.py", "__init__.py"]
    paths = [ROOT/"src"/name for name in names]
    paths += sorted((ROOT/"tests").glob("test_r7*.py"))
    paths += [ROOT/"R7_PHASE_RELEASE_EXPERIMENT_PROMPT.md", DOCS/"preregistration.md",
              DOCS/"figure_contract.md", ROOT/"requirements.txt", RAW/"reference_validation.json"]
    return paths


def freeze():
    destination = RAW / "frozen_protocol_and_code.json"
    if destination.exists():
        verify_freeze()
        return sha256(destination)
    if list((RAW/"main").glob("*/*/seed_*/metrics.json")):
        raise RuntimeError("Cannot create first freeze after main results")
    for name in ("theory_validation.json", "smoke_validation.json", "reference_validation.json"):
        evidence = json.loads((RAW/name).read_text(encoding="utf-8"))
        if evidence.get("status") != "PASS":
            raise RuntimeError(f"Pre-freeze check failed: {name}")
    import torch, numpy, scipy, pandas, matplotlib
    payload = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
               "versions": {"python": sys.version, "platform": platform.platform(),
                            "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
                            "pandas": pandas.__version__, "matplotlib": matplotlib.__version__},
               "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in science_files()}}
    write_json(destination, payload)
    return sha256(destination)


def verify_freeze():
    path = RAW/"frozen_protocol_and_code.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in payload["files"].items():
        if not (ROOT/name).is_file() or sha256(ROOT/name) != digest:
            raise RuntimeError(f"Frozen scientific file changed: {name}")
    return sha256(path)
