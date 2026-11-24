from __future__ import annotations

import json
from datetime import datetime, timezone
from .r8_config import ROOT, RAW, DOCS, EXPERIMENT
from .freeze_r7_phase_release import sha256, write_json


def capture_history():
    path = RAW/"protected_history.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    files = []
    for folder in ("src", "tests", "docs", "results"):
        for p in (ROOT/folder).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or EXPERIMENT in p.parts or "r8" in p.name.lower():
                continue
            files.append(p)
    files += [p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith("R8_")]
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(files))})
    write_json(path, record)
    return record


def verify_history():
    record = json.loads((RAW/"protected_history.json").read_text(encoding="utf-8"))
    changed = [p for p, digest in record["files"].items() if not (ROOT/p).is_file() or sha256(ROOT/p) != digest]
    return dict(pass_check=not changed, checked_files=len(record["files"]), changed_or_missing=changed)


def science_files():
    names = ["r8_config.py", "r8_models.py", "r8_sampling.py", "r8_loss.py", "r8_reference.py", "r8_theory.py", "r8_evaluate.py",
             "r8_decision.py", "train_r8_coupling_recovery.py", "run_r8_coupling_recovery.py", "freeze_r8_coupling_recovery.py",
             "periodic_potential_physics.py", "periodic_potential_config.py", "periodic_potential_evaluate.py",
             "periodic_potential_models.py", "wavepacket_models.py", "wavepacket_physics.py", "wavepacket_config.py",
             "physics.py", "train_wavepacket_shared_spectral_ltpb.py", "train_r7_phase_release.py", "freeze_r7_phase_release.py",
             "r7_models.py", "r7_config.py", "r7_loss.py", "r7_sampling.py", "r7_runtime.py", "config.py", "__init__.py"]
    return [ROOT/"src"/n for n in names]+[ROOT/"tests/test_r8_coupling_recovery.py", DOCS/"preregistration.md",
        DOCS/"theory_check.md", DOCS/"figure_contract.md", ROOT/"requirements.txt", RAW/"reference_validation.json"]


def freeze():
    path = RAW/"frozen_protocol_and_code.json"
    if path.exists():
        return verify_freeze()
    if list((RAW/"exploration").rglob("metrics.json")):
        raise RuntimeError("First freeze must precede exploration")
    for file in ("reference_validation.json", "smoke_validation.json", "theory_validation.json"):
        if json.loads((RAW/file).read_text(encoding="utf-8"))["status"] != "PASS":
            raise RuntimeError(f"Pre-freeze check failed: {file}")
    record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in science_files()})
    write_json(path, record)
    return sha256(path)


def verify_freeze():
    path = RAW/"frozen_protocol_and_code.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for p, digest in record["files"].items():
        if not (ROOT/p).is_file() or sha256(ROOT/p) != digest:
            raise RuntimeError(f"Frozen scientific code changed: {p}")
    return sha256(path)
