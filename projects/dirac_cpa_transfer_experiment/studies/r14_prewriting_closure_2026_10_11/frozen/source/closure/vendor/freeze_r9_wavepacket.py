from __future__ import annotations
import json
from datetime import datetime, timezone
from .r9_config import ROOT, RAW, DOCS, R9Config
from .freeze_r7_phase_release import sha256, write_json


def selected_config():
    record = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    if record["status"] != "PASS":
        raise RuntimeError("Reference calibration failed; formal training is prohibited")
    low, high = record["selected_truncation"]
    return R9Config(n_min=low, n_max=high)


def capture_history():
    path = RAW/"protected_history.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    files = []
    for folder in ("src", "tests", "docs", "results"):
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and "r9_wavepacket_periodic_potential" not in p.parts and "r9" not in p.name.lower():
                files.append(p)
    files += [p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith("R9_")]
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(files))})
    write_json(path, record)
    return record


def verify_history():
    record = json.loads((RAW/"protected_history.json").read_text(encoding="utf-8"))
    changed = [p for p, digest in record["files"].items() if not (ROOT/p).is_file() or sha256(ROOT/p) != digest]
    return dict(pass_check=not changed, checked_files=len(record["files"]), changed_or_missing=changed)


def science_files():
    names = ["r9_config.py", "r9_models.py", "r9_loss.py", "r9_time_derivative.py", "r9_reference.py", "r9_evaluate.py", "r9_decision.py",
             "train_r9_wavepacket.py", "freeze_r9_wavepacket.py", "run_r9_wavepacket.py", "r9_runtime.py",
             "r8_models.py", "r8_loss.py", "r8_sampling.py", "r8_config.py", "r7_runtime.py",
             "wavepacket_models.py", "wavepacket_physics.py", "wavepacket_config.py", "physics.py",
             "periodic_potential_config.py", "periodic_potential_physics.py", "config.py", "__init__.py",
             "train_wavepacket_shared_spectral_ltpb.py", "train_r7_phase_release.py", "freeze_r7_phase_release.py"]
    return [ROOT/"src"/p for p in names]+[ROOT/"tests/test_r9_wavepacket.py", DOCS/"preregistration.md",
        DOCS/"theory_check.md", DOCS/"figure_contract.md", ROOT/"requirements.txt", RAW/"reference_validation.json",
        RAW/"protocol.json", RAW/"smoke_validation.json", RAW/"pre_freeze_test_result.json"]


def freeze():
    path = RAW/"frozen_protocol_and_code.json"
    if path.exists():
        return verify_freeze()
    if list((RAW/"formal").rglob("model_state.pt")):
        raise RuntimeError("Freeze must precede formal training")
    selected_config()
    for name in ("smoke_validation.json", "pre_freeze_test_result.json"):
        if json.loads((RAW/name).read_text(encoding="utf-8"))["status"] != "PASS":
            raise RuntimeError(f"Engineering check failed: {name}")
    record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in science_files()})
    write_json(path, record)
    return sha256(path)


def verify_freeze():
    path = RAW/"frozen_protocol_and_code.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in record["files"].items():
        if not (ROOT/relative).is_file() or sha256(ROOT/relative) != digest:
            raise RuntimeError(f"Frozen scientific file changed: {relative}")
    return sha256(path)
